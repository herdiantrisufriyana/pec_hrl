"""Resume EOPE_T1 final training from checkpoint, then continue ablation."""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pydivnn', 'src'))
from pydivnn.tidyset import TidySet
from pydivnn.ontonet import Ontonet
from pydivnn.dataset import OntoDataset
from pydivnn.utils import get_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 33
BATCH_SIZE = 32
LR = 2e-6
FINAL_EPOCHS = 500
FINAL_PATIENCE = 250


def resume_eope_t1():
    """Resume EOPE_T1 final training from epoch 1 checkpoint."""
    model_dir = os.path.join(BASE_DIR, 'tsne', 'OD_C_EOSPE.P_TE_T1_cx')
    ts_path = os.path.join(model_dir, 'OD_C_EOSPE.P_TE_T1_cx_target.ts.tar.gz')
    checkpoint_path = os.path.join(model_dir, 'final', '1.pt')
    final_dir = os.path.join(model_dir, 'final')

    print('=== Resuming EOPE_T1 final training from epoch 1 ===')
    ts = TidySet.read(ts_path)
    print(f'  TidySet: ontomap={ts.ontomap.shape}, genes={len(ts.gene_names)}')

    n_genes = len(ts.gene_names)
    # EOPE_T1 has 353 genes — OOM on MPS at 17.4 GiB
    device = torch.device('cpu')
    print(f'  Using device: {device} (forced for 353 genes)')

    model = Ontonet(ts, device, init_seed=SEED, init2_seed=SEED,
                    l2_norm=0.01, output_unit=1, output_activation='sigmoid')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    print(f'  Loaded checkpoint: {checkpoint_path}')

    indices = np.arange(ts.ontomap.shape[0])
    dataset = OntoDataset(ts, indices)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(SEED))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Outcome weights
    unique, counts = np.unique(ts.outcome, return_counts=True)
    total = len(ts.outcome)
    weight_map = {label: 1 / (count / total * 0.5) for label, count in zip(unique, counts)}
    weights = torch.Tensor([weight_map[k] for k in sorted(weight_map.keys())]).to(device)

    def criterions(pred, target):
        wm = torch.ones_like(target)
        for outcome_val, w in enumerate(weights):
            wm[target == outcome_val] = w
        return torch.sum(wm * (pred - target) ** 2) / torch.sum(wm)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.96, patience=1, mode='max', threshold=0.01,
        cooldown=0, min_lr=LR / 32)

    # Start from epoch 2 (epoch 1 was the checkpoint)
    # The old run did ~50 epochs after epoch 1 without improvement
    # We restart patience from 0 since we're resuming fresh
    best_val_auc = -np.inf
    patience_counter = 0
    best_epoch = 1  # from checkpoint
    history = []

    start_epoch = 2
    t0 = time.time()

    for epoch in range(start_epoch, FINAL_EPOCHS + 1):
        model.train()
        train_true, train_pred = [], []
        for batch_x, batch_y in train_dl:
            optimizer.zero_grad()
            inputs = batch_x['ontoarray'].float().to(device)
            targets = {k: v.float().to(device) for k, v in batch_y.items()}
            outputs = model(inputs)

            w_total = 0.3 * (len(outputs) - 1) + 1
            w_nonroot = 0.3 / w_total
            w_root = 1 / w_total
            total_loss = sum(
                (w_root if k == 'root' else w_nonroot) * criterions(outputs[k], targets[k])
                for k in outputs
            ) / w_total
            total_loss.backward()
            optimizer.step()

            train_true.append(targets['root'].detach().cpu().numpy().flatten())
            train_pred.append(outputs['root'].detach().cpu().numpy().flatten())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                inputs = batch_x['ontoarray'].float().to(device)
                targets = {k: v.float().to(device) for k, v in batch_y.items()}
                outputs = model(inputs)
                val_true.append(targets['root'].detach().cpu().numpy().flatten())
                val_pred.append(outputs['root'].detach().cpu().numpy().flatten())

        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)

        try:
            train_auc = roc_auc_score(train_true, train_pred)
        except ValueError:
            train_auc = 0.5
        try:
            val_auc = roc_auc_score(val_true, val_pred)
        except ValueError:
            val_auc = 0.5

        scheduler.step(val_auc)
        history.append({
            'epoch': epoch, 'train_auc': train_auc, 'val_auc': val_auc,
            'lr': optimizer.param_groups[0]['lr']
        })

        if val_auc > best_val_auc + 0.001:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(final_dir, f'{epoch}.pt'))
            print(f'    Epoch {epoch}: val_auc={val_auc:.4f} (NEW BEST)')
        else:
            patience_counter += 1
            if patience_counter >= FINAL_PATIENCE:
                print(f'    Early stopping at epoch {epoch} (patience={FINAL_PATIENCE})')
                break

        if epoch % 50 == 0:
            elapsed = time.time() - t0
            print(f'    Epoch {epoch}: train={train_auc:.4f} val={val_auc:.4f} patience={patience_counter}/{FINAL_PATIENCE} time={elapsed:.0f}s')

    elapsed = time.time() - t0
    # Final eval
    try:
        final_train_auc = roc_auc_score(train_true, train_pred)
    except:
        final_train_auc = 0.5
    try:
        final_val_auc = roc_auc_score(val_true, val_pred)
    except:
        final_val_auc = 0.5

    result = {
        'model_id': 'EOPE_T1',
        'model_prefix': 'OD_C_EOSPE.P_TE_T1_cx',
        'method': 'tsne',
        'best_l2': 0.01,
        'train_auc': float(final_train_auc),
        'val_auc': float(best_val_auc if best_val_auc > -np.inf else final_val_auc),
        'best_epoch': best_epoch,
        'n_genes': 353,
        'n_samples': 33,
    }
    with open(os.path.join(model_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(history).to_csv(os.path.join(model_dir, 'history.csv'), index=False)
    print(f'  DONE: train={final_train_auc:.4f} val={best_val_auc:.4f} best_ep={best_epoch} time={elapsed:.0f}s')
    return result


if __name__ == '__main__':
    # Step 1: Resume EOPE_T1
    resume_eope_t1()

    # Step 2: Continue with the main ablation (will skip all completed)
    print('\n=== Continuing main ablation ===\n')
    exec(open(os.path.join(BASE_DIR, 'run_ablation.py')).read())

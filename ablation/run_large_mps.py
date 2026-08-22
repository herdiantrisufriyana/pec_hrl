"""Run large models on MPS with reduced batch_size.
Handles EOPE_HELLP (283), EOPE_T1 (353), EOPE_LSE (405) for PCA and UMAP.
Also EOPE_HELLP t-SNE if not done.
Skips any model/method with existing result.json.
Uses batch_size=4 to avoid MPS OOM."""

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
from pydivnn.dim_reduction import reduce_dimensions
from pydivnn.utils import best_configuration, get_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIDYSET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'tidyset')
COR_DIR = BASE_DIR

LARGE_MODELS = [
    ('EOPE_HELLP', 'OD_C_EOSPE.H3_NH3_HELLP3_cx', 'OD_C_EOSPE.H3_NH3_HELLP3', 283),
    ('EOPE_T1', 'OD_C_EOSPE.P_TE_T1_cx', 'OD_C_EOSPE.P_TE_T1', 353),
    ('EOPE_LSE', 'OD_C_EOSPE.E_PRE_LSE_cx', 'OD_C_EOSPE.E_PRE_LSE', 405),
]

# Methods to run — skip t-SNE for models that already have it
METHODS = ['tsne', 'pca', 'umap']
L2_NORMS = [0.01, 0.1, 1, 10]
TUNING_EPOCHS = 30
TUNING_PATIENCE = 15
FINAL_EPOCHS = 500
FINAL_PATIENCE = 250
LR = 2e-6
SEED = 33
DIMS = 224

# Reduced batch sizes by gene count
def get_batch_size(n_genes):
    if n_genes > 350:
        return 4
    elif n_genes > 250:
        return 8
    else:
        return 16


def train_one(tidy_set, l2_norm, epochs, patience, batch_size, lr, seed, save_dir):
    device = get_device()
    print(f'    (device={device}, batch_size={batch_size})')

    model = Ontonet(tidy_set, device, init_seed=seed, init2_seed=seed,
                    l2_norm=l2_norm, output_unit=1, output_activation='sigmoid')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, 'null.pt'))

    indices = np.arange(tidy_set.ontomap.shape[0])
    dataset = OntoDataset(tidy_set, indices)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(seed))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    unique, counts = np.unique(tidy_set.outcome, return_counts=True)
    total = len(tidy_set.outcome)
    weight_map = {label: 1 / (count / total * 0.5) for label, count in zip(unique, counts)}
    weights = torch.Tensor([weight_map[k] for k in sorted(weight_map.keys())]).to(device)

    def criterions(pred, target):
        wm = torch.ones_like(target)
        for outcome_val, w in enumerate(weights):
            wm[target == outcome_val] = w
        return torch.sum(wm * (pred - target) ** 2) / torch.sum(wm)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.96, patience=1, mode='max', threshold=0.01,
        cooldown=0, min_lr=lr / 32)

    best_val_auc = -np.inf
    patience_counter = 0
    best_epoch = 0
    history = []

    for epoch in range(epochs):
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
        history.append({'epoch': epoch + 1, 'train_auc': train_auc, 'val_auc': val_auc,
                        'lr': optimizer.param_groups[0]['lr']})

        if val_auc > best_val_auc + 0.001:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            if save_dir:
                torch.save(model.state_dict(), os.path.join(save_dir, f'{epoch + 1}.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Clear MPS cache after each training run
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return train_auc, best_val_auc, history, best_epoch


def run_model_method(model_id, model_prefix, cor_name, n_genes, method):
    out_dir = os.path.join(BASE_DIR, method, model_prefix)
    result_path = os.path.join(out_dir, 'result.json')
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        print(f'SKIP {method}/{model_id}: already completed (val_auc={result["val_auc"]:.4f})')
        return result

    print(f'\n{"="*60}')
    print(f'Model: {model_id}, Method: {method}, Genes: {n_genes}')
    print(f'{"="*60}')

    batch_size = get_batch_size(n_genes)
    print(f'  batch_size={batch_size} (reduced for MPS)')

    ts_path = os.path.join(TIDYSET_DIR, f'{model_prefix}_target.ts.tar.gz')
    ts = TidySet.read(ts_path)
    value = ts.pdata.drop(columns=['outcome'])
    outcome = ts.outcome
    ontology = ts.ontology

    cor_path = os.path.join(COR_DIR, f'pear_cor_{cor_name}.csv')
    cor_mat = pd.read_csv(cor_path, index_col=0).values

    print(f'  Applying {method} to {cor_mat.shape[0]} genes...')
    mapping = reduce_dimensions(cor_mat, n_components=3, method=method, seed=SEED)

    print(f'  Compiling TidySet (dims={DIMS})...')
    ts_new = TidySet.compile(value=value, outcome=outcome, similarity=cor_mat,
                             mapping=mapping, ontology=ontology, ranked=True,
                             dims=DIMS, seed_num=SEED)
    print(f'  ontomap: {ts_new.ontomap.shape}, ontotype: {len(ts_new.ontotype)} nodes')

    os.makedirs(out_dir, exist_ok=True)
    ts_new.write(os.path.join(out_dir, f'{model_prefix}_target'))

    print(f'  Tuning ({len(L2_NORMS)} L2 norms x {TUNING_EPOCHS} epochs)...')
    train_aucs, val_aucs = [], []
    for l2 in L2_NORMS:
        tune_dir = os.path.join(out_dir, f'tuning_l2_{l2}')
        t0 = time.time()
        tr_auc, vl_auc, hist, best_ep = train_one(
            ts_new, l2, TUNING_EPOCHS, TUNING_PATIENCE, batch_size, LR, SEED, tune_dir)
        elapsed = time.time() - t0
        train_aucs.append(tr_auc)
        val_aucs.append(vl_auc)
        print(f'    L2={l2}: train_auc={tr_auc:.4f}, val_auc={vl_auc:.4f}, '
              f'best_epoch={best_ep}, time={elapsed:.0f}s')

    best_idx = best_configuration(train_aucs, val_aucs)
    best_l2 = L2_NORMS[best_idx]
    print(f'  Best L2: {best_l2}')

    print(f'  Final training (L2={best_l2}, {FINAL_EPOCHS} epochs max)...')
    final_dir = os.path.join(out_dir, 'final')
    t0 = time.time()
    tr_auc, vl_auc, hist, best_ep = train_one(
        ts_new, best_l2, FINAL_EPOCHS, FINAL_PATIENCE, batch_size, LR, SEED, final_dir)
    elapsed = time.time() - t0
    print(f'  Final: train_auc={tr_auc:.4f}, val_auc={vl_auc:.4f}, '
          f'best_epoch={best_ep}, time={elapsed:.0f}s')

    result = {
        'model_id': model_id, 'model_prefix': model_prefix, 'method': method,
        'best_l2': best_l2, 'train_auc': float(tr_auc), 'val_auc': float(vl_auc),
        'best_epoch': best_ep, 'n_genes': n_genes, 'n_samples': len(outcome),
        'batch_size': batch_size,
    }
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, 'history.csv'), index=False)
    return result


if __name__ == '__main__':
    print('=== Large models on MPS (reduced batch_size) ===')
    start = time.time()

    # Start with smallest large model to test MPS stability
    for model_id, model_prefix, cor_name, n_genes in LARGE_MODELS:
        for method in METHODS:
            try:
                run_model_method(model_id, model_prefix, cor_name, n_genes, method)
            except RuntimeError as e:
                if 'MPS' in str(e) or 'out of memory' in str(e):
                    print(f'  MPS OOM: {e}')
                    print(f'  SKIPPING {method}/{model_id} — needs CPU')
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                else:
                    raise
            except Exception as e:
                print(f'  ERROR: {e}')
                import traceback
                traceback.print_exc()

    total = time.time() - start
    print(f'\nDone. Total: {total/3600:.1f}h')

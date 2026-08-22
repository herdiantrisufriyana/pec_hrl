"""Ablation study: t-SNE vs PCA vs UMAP for DI-VNN feature maps.

Runs all 7 pec_hrl models with each dimensionality reduction method.
For each model × method combination:
  1. Load correlation matrix and existing TidySet data
  2. Apply dim reduction to get 3D mapping
  3. Compile new TidySet
  4. Train DI-VNN (tuning: 4 L2 norms × 30 epochs; final: best L2 × 500 epochs)
  5. Evaluate AUC-ROC on training set

Results saved to ablation/{method}/{model_prefix}/ folders.
"""

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
from sklearn.utils import resample

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pydivnn', 'src'))
from pydivnn.tidyset import TidySet
from pydivnn.ontonet import Ontonet
from pydivnn.dataset import OntoDataset
from pydivnn.dim_reduction import reduce_dimensions
from pydivnn.utils import best_configuration, get_device

# Configuration
MODELS = [
    ('LOPE_MSE', 'OD_C_LOSPE.E_PRE_MSE_cx', 'OD_C_LOSPE.E_PRE_MSE'),
    ('LOPE_CHOR', 'OD_C_LOSPE.C3_NC3_CHOR3_cx', 'OD_C_LOSPE.C3_NC3_CHOR3'),
    ('LOPE_HELLP', 'OD_C_LOSPE.H3_NH3_HELLP3_cx', 'OD_C_LOSPE.H3_NH3_HELLP3'),
    ('EOPE_LSE', 'OD_C_EOSPE.E_PRE_LSE_cx', 'OD_C_EOSPE.E_PRE_LSE'),
    ('EOPE_T1', 'OD_C_EOSPE.P_TE_T1_cx', 'OD_C_EOSPE.P_TE_T1'),
    ('EOPE_T2', 'OD_C_EOSPE.P_TE_T2_pea', 'OD_C_EOSPE.P_TE_T2'),
    ('EOPE_HELLP', 'OD_C_EOSPE.H3_NH3_HELLP3_cx', 'OD_C_EOSPE.H3_NH3_HELLP3'),
]

METHODS = ['tsne', 'pca', 'umap']
L2_NORMS = [0.01, 0.1, 1, 10]
TUNING_EPOCHS = 30
TUNING_PATIENCE = 15
FINAL_EPOCHS = 500
FINAL_PATIENCE = 250
BATCH_SIZE = 32
LR = 2e-6
SEED = 33
DIMS = 224

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIDYSET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'tidyset')
COR_DIR = BASE_DIR


def train_one(tidy_set, l2_norm, epochs, patience, batch_size, lr, seed, save_dir):
    """Train a single DI-VNN model. Returns (train_auc, val_auc, history)."""
    n_genes = len(tidy_set.gene_names) if tidy_set.gene_names else 0
    if n_genes > 280:
        device = torch.device('cpu')
        print(f'    (Using CPU for {n_genes} genes — MPS OOM above ~280)')
    else:
        device = get_device()
        print(f'    (Using device: {device}, {n_genes} genes)')

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

    # Outcome weights
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
        # Train
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

        # Validate
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

        # AUC-ROC (bootstrap)
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
            'epoch': epoch + 1, 'train_auc': train_auc, 'val_auc': val_auc,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Early stopping
        if val_auc > best_val_auc + 0.001:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            if save_dir:
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f'{epoch + 1}.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return train_auc, best_val_auc, history, best_epoch


def run_model_method(model_id, model_prefix, cor_name, method):
    """Run ablation for one model × method combination."""
    # Skip if already completed
    out_dir = os.path.join(BASE_DIR, method, model_prefix)
    result_path = os.path.join(out_dir, 'result.json')
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        print(f'\nSKIP {method}/{model_id}: already completed (val_auc={result["val_auc"]:.4f})')
        return result

    print(f'\n{"="*60}')
    print(f'Model: {model_id}, Method: {method}')
    print(f'{"="*60}')

    # Load existing TidySet for value/outcome/ontology
    ts_path = os.path.join(TIDYSET_DIR, f'{model_prefix}_target.ts.tar.gz')
    ts = TidySet.read(ts_path)
    value = ts.pdata.drop(columns=['outcome'])
    outcome = ts.outcome
    ontology = ts.ontology

    # Load correlation matrix
    cor_path = os.path.join(COR_DIR, f'pear_cor_{cor_name}.csv')
    cor_mat = pd.read_csv(cor_path, index_col=0).values

    # Apply dimensionality reduction
    print(f'  Applying {method} to {cor_mat.shape[0]} genes...')
    mapping = reduce_dimensions(cor_mat, n_components=3, method=method, seed=SEED)

    # Compile TidySet with new mapping
    print(f'  Compiling TidySet (dims={DIMS})...')
    ts_new = TidySet.compile(
        value=value, outcome=outcome, similarity=cor_mat,
        mapping=mapping, ontology=ontology, ranked=True,
        dims=DIMS, seed_num=SEED)
    print(f'  ontomap: {ts_new.ontomap.shape}, ontotype: {len(ts_new.ontotype)} nodes')

    # Save compiled TidySet
    out_dir = os.path.join(BASE_DIR, method, model_prefix)
    os.makedirs(out_dir, exist_ok=True)
    ts_new.write(os.path.join(out_dir, f'{model_prefix}_target'))

    # Tuning: try 4 L2 norms
    print(f'  Tuning ({len(L2_NORMS)} L2 norms × {TUNING_EPOCHS} epochs)...')
    train_aucs, val_aucs = [], []
    for l2 in L2_NORMS:
        tune_dir = os.path.join(out_dir, f'tuning_l2_{l2}')
        t0 = time.time()
        tr_auc, vl_auc, hist, best_ep = train_one(
            ts_new, l2, TUNING_EPOCHS, TUNING_PATIENCE,
            BATCH_SIZE, LR, SEED, tune_dir)
        elapsed = time.time() - t0
        train_aucs.append(tr_auc)
        val_aucs.append(vl_auc)
        print(f'    L2={l2}: train_auc={tr_auc:.4f}, val_auc={vl_auc:.4f}, '
              f'best_epoch={best_ep}, time={elapsed:.0f}s')

    # Select best L2
    best_idx = best_configuration(train_aucs, val_aucs)
    best_l2 = L2_NORMS[best_idx]
    print(f'  Best L2: {best_l2}')

    # Final training with best L2
    print(f'  Final training (L2={best_l2}, {FINAL_EPOCHS} epochs max)...')
    final_dir = os.path.join(out_dir, 'final')
    t0 = time.time()
    tr_auc, vl_auc, hist, best_ep = train_one(
        ts_new, best_l2, FINAL_EPOCHS, FINAL_PATIENCE,
        BATCH_SIZE, LR, SEED, final_dir)
    elapsed = time.time() - t0
    print(f'  Final: train_auc={tr_auc:.4f}, val_auc={vl_auc:.4f}, '
          f'best_epoch={best_ep}, time={elapsed:.0f}s')

    # Save results
    result = {
        'model_id': model_id,
        'model_prefix': model_prefix,
        'method': method,
        'best_l2': best_l2,
        'train_auc': float(tr_auc),
        'val_auc': float(vl_auc),
        'best_epoch': best_ep,
        'n_genes': cor_mat.shape[0],
        'n_samples': len(outcome),
    }
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    pd.DataFrame(hist).to_csv(os.path.join(out_dir, 'history.csv'), index=False)

    return result


def main():
    print('Ablation Study: t-SNE vs PCA vs UMAP for DI-VNN')
    print(f'Models: {len(MODELS)}, Methods: {METHODS}')
    print(f'Tuning: {len(L2_NORMS)} L2 × {TUNING_EPOCHS} epochs')
    print(f'Final: {FINAL_EPOCHS} epochs (patience={FINAL_PATIENCE})')
    print()

    all_results = []
    start = time.time()

    for method in METHODS:
        for model_id, model_prefix, cor_name in MODELS:
            try:
                result = run_model_method(model_id, model_prefix, cor_name, method)
                all_results.append(result)
            except Exception as e:
                print(f'  ERROR: {e}')
                import traceback
                traceback.print_exc()
                all_results.append({
                    'model_id': model_id, 'method': method,
                    'error': str(e)
                })

    # Summary table
    total_time = time.time() - start
    print(f'\n{"="*60}')
    print(f'SUMMARY (total time: {total_time/3600:.1f}h)')
    print(f'{"="*60}')

    df = pd.DataFrame(all_results)
    summary_path = os.path.join(BASE_DIR, 'comparison.csv')
    df.to_csv(summary_path, index=False)
    print(df.to_string(index=False))
    print(f'\nSaved to: {summary_path}')


if __name__ == '__main__':
    main()

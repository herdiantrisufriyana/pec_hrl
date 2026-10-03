"""Ablation study v2: proper external validation.

For each model x method:
  1. Load training TidySet (OD_) and validation TidySet (OV_)
  2. Apply dim reduction to correlation matrix → new 3D mapping
  3. Recompile BOTH TidySets with new mapping (same ontology, same genes)
  4. Train on training TidySet
  5. Predict on validation TidySet using model_predict()
  6. Compute AUC-ROC from external validation predictions
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pydivnn', 'src'))
from pydivnn.tidyset import TidySet
from pydivnn.ontonet import Ontonet
from pydivnn.dataset import OntoDataset
from pydivnn.dim_reduction import reduce_dimensions
from pydivnn.utils import best_configuration, get_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_TIDYSET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'tidyset')
VAL_TIDYSET_DIR = '/Users/herdiantrisufriyana/Documents/lab/pec/data/divnn_models/validation'
COR_DIR = BASE_DIR

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
OUT_DIR_NAME = 'v2'


def get_batch_size(n_genes):
    if n_genes > 350:
        return 4
    elif n_genes > 250:
        return 8
    else:
        return 32


def train_one(tidy_set, l2_norm, epochs, patience, batch_size, lr, seed, save_dir):
    """Train a single DI-VNN model. Returns (train_auc, val_auc_internal, history)."""
    device = get_device()

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

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return train_auc, best_val_auc, history, best_epoch


def predict_on_val(train_ts, val_ts, model_weight_dir, l2_norm, batch_size, seed):
    """Load trained model and predict on external validation TidySet."""
    device = get_device()

    model = Ontonet(train_ts, device, init_seed=seed, init2_seed=seed,
                    l2_norm=l2_norm, output_unit=1, output_activation='sigmoid')

    # Find latest checkpoint
    files = [f for f in os.listdir(model_weight_dir) if f.endswith('.pt') and f.split('.')[0].isdigit()]
    if not files:
        print('    No .pt checkpoints found')
        return None, None
    epochs = [int(f.split('.')[0]) for f in files]
    latest = files[epochs.index(max(epochs))]
    model_path = os.path.join(model_weight_dir, latest)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    val_indices = np.arange(val_ts.ontomap.shape[0])
    val_dataset = OntoDataset(val_ts, val_indices)
    # Use batch_size that avoids remainder of 1 (BatchNorm fails on single sample)
    n_val = len(val_dataset)
    pred_batch = batch_size
    while n_val % pred_batch == 1 and pred_batch > 1:
        pred_batch -= 1
    val_dl = DataLoader(val_dataset, batch_size=pred_batch, shuffle=False)

    val_true, val_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in val_dl:
            inputs = batch_x['ontoarray'].float().to(device)
            outputs = model(inputs)
            val_true.append(batch_y['root'].numpy().flatten())
            val_pred.append(outputs['root'].detach().cpu().numpy().flatten())

    val_true = np.concatenate(val_true)
    val_pred = np.concatenate(val_pred)

    try:
        ext_auc = roc_auc_score(val_true, val_pred)
    except ValueError:
        ext_auc = 0.5

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return ext_auc, pd.DataFrame({'outcome': val_true, 'prob': val_pred})


def run_model_method(model_id, model_prefix, cor_name, method):
    out_dir = os.path.join(BASE_DIR, OUT_DIR_NAME, method, model_prefix)
    result_path = os.path.join(out_dir, 'result.json')
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        print(f'SKIP {method}/{model_id}: already completed (ext_val_auc={result["ext_val_auc"]:.4f})')
        return result

    print(f'\n{"="*60}')
    print(f'Model: {model_id}, Method: {method}')
    print(f'{"="*60}')

    # Load BOTH TidySets
    train_ts_path = os.path.join(TRAIN_TIDYSET_DIR, f'{model_prefix}_target.ts.tar.gz')
    val_ts_path = os.path.join(VAL_TIDYSET_DIR, f'{model_prefix}_target.ts.tar.gz')
    train_ts = TidySet.read(train_ts_path)
    val_ts = TidySet.read(val_ts_path)

    train_value = train_ts.pdata.drop(columns=['outcome'])
    train_outcome = train_ts.outcome
    val_value = val_ts.pdata.drop(columns=['outcome'])
    val_outcome = val_ts.outcome
    ontology = train_ts.ontology

    print(f'  Train: n={len(train_outcome)} ({int(sum(train_outcome==1))}+/{int(sum(train_outcome==0))}-)')
    print(f'  Val:   n={len(val_outcome)} ({int(sum(val_outcome==1))}+/{int(sum(val_outcome==0))}-)')

    # Load correlation matrix
    cor_path = os.path.join(COR_DIR, f'pear_cor_{cor_name}.csv')
    cor_mat = pd.read_csv(cor_path, index_col=0).values
    n_genes = cor_mat.shape[0]
    batch_size = get_batch_size(n_genes)

    # Apply dimensionality reduction (ONCE — reuse for both TidySets)
    print(f'  Applying {method} to {n_genes} genes...')
    mapping = reduce_dimensions(cor_mat, n_components=3, method=method, seed=SEED)

    # Compile TRAINING TidySet with new mapping
    print(f'  Compiling training TidySet (dims={DIMS})...')
    train_ts_new = TidySet.compile(
        value=train_value, outcome=train_outcome, similarity=cor_mat,
        mapping=mapping, ontology=ontology, ranked=True,
        dims=DIMS, seed_num=SEED)

    # Compile VALIDATION TidySet with SAME mapping and ontology
    print(f'  Compiling validation TidySet (dims={DIMS})...')
    val_ts_new = TidySet.compile(
        value=val_value, outcome=val_outcome, similarity=cor_mat,
        mapping=mapping, ontology=ontology, ranked=True,
        dims=DIMS, seed_num=SEED)

    print(f'  Train ontomap: {train_ts_new.ontomap.shape}, Val ontomap: {val_ts_new.ontomap.shape}')

    os.makedirs(out_dir, exist_ok=True)
    train_ts_new.write(os.path.join(out_dir, f'{model_prefix}_train'))
    val_ts_new.write(os.path.join(out_dir, f'{model_prefix}_val'))

    # Tuning
    print(f'  Tuning ({len(L2_NORMS)} L2 norms x {TUNING_EPOCHS} epochs, batch={batch_size})...')
    train_aucs, val_aucs = [], []
    for l2 in L2_NORMS:
        tune_dir = os.path.join(out_dir, f'tuning_l2_{l2}')
        t0 = time.time()
        tr_auc, vl_auc, hist, best_ep = train_one(
            train_ts_new, l2, TUNING_EPOCHS, TUNING_PATIENCE, batch_size, LR, SEED, tune_dir)
        elapsed = time.time() - t0
        train_aucs.append(tr_auc)
        val_aucs.append(vl_auc)
        print(f'    L2={l2}: train={tr_auc:.4f}, int_val={vl_auc:.4f}, ep={best_ep}, time={elapsed:.0f}s')

    best_idx = best_configuration(train_aucs, val_aucs)
    best_l2 = L2_NORMS[best_idx]
    print(f'  Best L2: {best_l2}')

    # Final training
    print(f'  Final training (L2={best_l2}, {FINAL_EPOCHS} epochs max, batch={batch_size})...')
    final_dir = os.path.join(out_dir, 'final')
    t0 = time.time()
    tr_auc, int_val_auc, hist, best_ep = train_one(
        train_ts_new, best_l2, FINAL_EPOCHS, FINAL_PATIENCE, batch_size, LR, SEED, final_dir)
    elapsed = time.time() - t0
    print(f'  Final: train={tr_auc:.4f}, int_val={int_val_auc:.4f}, ep={best_ep}, time={elapsed:.0f}s')

    # Predict on EXTERNAL validation set
    print(f'  Predicting on external validation (n={len(val_outcome)})...')
    ext_val_auc, pred_df = predict_on_val(
        train_ts_new, val_ts_new, final_dir, best_l2, batch_size, SEED)
    print(f'  EXTERNAL VAL AUC-ROC: {ext_val_auc:.4f}')

    # Save predictions
    if pred_df is not None:
        pred_df.to_csv(os.path.join(out_dir, f'prob_{model_prefix}.csv'), index=False)

    result = {
        'model_id': model_id, 'model_prefix': model_prefix, 'method': method,
        'best_l2': best_l2, 'train_auc': float(tr_auc),
        'int_val_auc': float(int_val_auc), 'ext_val_auc': float(ext_val_auc),
        'best_epoch': best_ep, 'n_genes': n_genes,
        'n_train': len(train_outcome), 'n_val': len(val_outcome),
        'n_train_pos': int(sum(train_outcome == 1)),
        'n_val_pos': int(sum(val_outcome == 1)),
        'batch_size': batch_size,
    }
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, 'history.csv'), index=False)
    return result


if __name__ == '__main__':
    print('Ablation Study v2: External Validation')
    print(f'Models: {len(MODELS)}, Methods: {METHODS}')
    print(f'Output: {OUT_DIR_NAME}/')
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
                all_results.append({'model_id': model_id, 'method': method, 'error': str(e)})

    total = time.time() - start
    print(f'\n{"="*60}')
    print(f'DONE (total: {total/3600:.1f}h)')
    print(f'{"="*60}')

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(BASE_DIR, OUT_DIR_NAME, 'comparison_v2.csv'), index=False)
    print(df.to_string(index=False))

"""Evaluate R2 ablation models on external validation set.

Loads trained models from ablation/{method}/{model}/final/
and predicts on external validation TidySets (OV_).
Reports ext_val_auc for comparison with v2 results.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pydivnn', 'src'))
from pydivnn.tidyset import TidySet
from pydivnn.ontonet import Ontonet
from pydivnn.dataset import OntoDataset
from pydivnn.utils import get_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
R2_DIR = BASE_DIR  # R2 models in ablation/{method}/{model}/
VAL_TIDYSET_DIR = '/Users/herdiantrisufriyana/Documents/lab/pec/data/divnn_models/validation'
TRAIN_TIDYSET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'tidyset')

MODELS = [
    ('LOPE_MSE', 'OD_C_LOSPE.E_PRE_MSE_cx'),
    ('LOPE_CHOR', 'OD_C_LOSPE.C3_NC3_CHOR3_cx'),
    ('LOPE_HELLP', 'OD_C_LOSPE.H3_NH3_HELLP3_cx'),
    ('EOPE_LSE', 'OD_C_EOSPE.E_PRE_LSE_cx'),
    ('EOPE_T1', 'OD_C_EOSPE.P_TE_T1_cx'),
    ('EOPE_T2', 'OD_C_EOSPE.P_TE_T2_pea'),
    ('EOPE_HELLP', 'OD_C_EOSPE.H3_NH3_HELLP3_cx'),
]

METHODS = ['tsne', 'pca', 'umap']
SEED = 33


def get_batch_size(n_genes):
    if n_genes > 350:
        return 4
    elif n_genes > 250:
        return 8
    else:
        return 32


def eval_model(model_id, model_prefix, method):
    """Load R2 trained model and evaluate on external validation."""
    r2_model_dir = os.path.join(R2_DIR, method, model_prefix)
    final_dir = os.path.join(r2_model_dir, 'final')
    r2_result_path = os.path.join(r2_model_dir, 'result.json')
    out_path = os.path.join(R2_DIR, 'r2_ext_val', method, f'{model_prefix}.json')

    if os.path.exists(out_path):
        with open(out_path) as f:
            result = json.load(f)
        print(f'SKIP {method}/{model_id}: already evaluated (ext_val={result["ext_val_auc"]:.4f})')
        return result

    if not os.path.exists(final_dir):
        print(f'MISS {method}/{model_id}: no final/ dir')
        return None

    # Find best checkpoint
    files = [f for f in os.listdir(final_dir) if f.endswith('.pt') and f.split('.')[0].isdigit()]
    if not files:
        print(f'MISS {method}/{model_id}: no .pt checkpoints')
        return None

    # Load R2 result to get best_l2
    if os.path.exists(r2_result_path):
        with open(r2_result_path) as f:
            r2_result = json.load(f)
        best_l2 = r2_result.get('best_l2', 0.01)
    else:
        best_l2 = 0.01

    # Load training TidySet (to reconstruct model architecture)
    train_ts = TidySet.read(os.path.join(TRAIN_TIDYSET_DIR, f'{model_prefix}_target.ts.tar.gz'))
    n_genes = len(train_ts.gene_names)
    batch_size = get_batch_size(n_genes)

    # Load external validation TidySet (original, with t-SNE mapping)
    val_ts = TidySet.read(os.path.join(VAL_TIDYSET_DIR, f'{model_prefix}_target.ts.tar.gz'))

    print(f'{method}/{model_id}: n_genes={n_genes}, val_n={len(val_ts.outcome)}, L2={best_l2}')

    device = get_device()
    model = Ontonet(train_ts, device, init_seed=SEED, init2_seed=SEED,
                    l2_norm=best_l2, output_unit=1, output_activation='sigmoid')

    # Load weights
    epochs = [int(f.split('.')[0]) for f in files]
    latest = files[epochs.index(max(epochs))]
    model_path = os.path.join(final_dir, latest)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # Predict on external validation
    val_indices = np.arange(val_ts.ontomap.shape[0])
    val_dataset = OntoDataset(val_ts, val_indices)

    n_val = len(val_dataset)
    pred_batch = batch_size
    while n_val % pred_batch == 1 and pred_batch > 1:
        pred_batch -= 1

    val_dl = DataLoader(val_dataset, batch_size=pred_batch, shuffle=False)

    val_true, val_pred = [], []
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

    print(f'  ext_val_auc={ext_auc:.4f}')

    result = {
        'model_id': model_id, 'model_prefix': model_prefix, 'method': method,
        'best_l2': best_l2, 'ext_val_auc': float(ext_auc),
        'n_genes': n_genes, 'n_val': len(val_ts.outcome),
        'source': 'r2_model_with_original_val_tidyset',
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == '__main__':
    print('Evaluating R2 models on external validation\n')
    all_results = []

    for method in METHODS:
        for model_id, model_prefix in MODELS:
            try:
                result = eval_model(model_id, model_prefix, method)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f'  ERROR {method}/{model_id}: {e}')
                import traceback
                traceback.print_exc()

    print(f'\n{"="*60}')
    print('R2 models external validation results:')
    for r in sorted(all_results, key=lambda x: f'{x["method"]}/{x["model_id"]}'):
        print(f'  {r["method"]}/{r["model_id"]}: ext_val={r["ext_val_auc"]:.4f}')

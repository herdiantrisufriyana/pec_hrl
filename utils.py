import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import roc_auc_score, r2_score, make_scorer
import os
import csv

def train_and_estimate_sample_size(prefix, model_details, save_dir='default', cv=5,
                                   scoring=None, task='classification', seed=42):
    np.random.seed(seed)

    model = model_details['model']
    param_grid = model_details['param_grid']

    save_path = f'inst/extdata/{save_dir}'
    os.makedirs(save_path, exist_ok=True)

    data = pd.read_csv(f'inst/extdata/{prefix}.csv')
    X = data.drop(columns=['outcome', 'id'])
    y = data['outcome'].astype(float if task == 'regression' else int)

    num_predictors = X.shape[1]
    proportions = np.linspace(0.1, 0.9, 9)
    results = []

    # Choose correct splitter
    if task == 'classification':
        splitter = StratifiedShuffleSplit(n_splits=1, random_state=seed)
    else:
        splitter = ShuffleSplit(n_splits=1, random_state=seed)

    # Default scoring
    if scoring is None:
        scoring = 'roc_auc' if task == 'classification' else 'r2'

    # Custom scorer setup
    if task == 'classification':
        scorer = scoring
    else:
        scorer = make_scorer(r2_score)

    for prop in proportions:
        splitter.test_size = prop
        for train_index, test_index in splitter.split(X, y):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]

            # EPV for classification, N/predictors for regression
            if task == 'classification':
                event_count = min(y_train.sum(), len(y_train) - y_train.sum())
                epv = event_count / num_predictors
            else:
                epv = len(y_train) / num_predictors

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = model_details['model']
            if hasattr(model, 'random_state'):
                model.set_params(random_state=seed)

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=scorer,
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)

            best_model = grid_search.best_estimator_

            # Prediction and metric
            if task == 'classification':
                if hasattr(best_model, "predict_proba"):
                    y_scores = best_model.predict_proba(X_train_scaled)[:, 1]
                else:
                    y_scores = best_model.decision_function(X_train_scaled)
                score = roc_auc_score(y_train, y_scores)
            else:
                y_pred = best_model.predict(X_train_scaled)
                score = r2_score(y_train, y_pred)

            results.append((prop, epv, score))

    results_path = os.path.join(save_path, 'sample_size_estimation.csv')
    with open(results_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['proportion', 'epv', 'score'])
        writer.writerows(results)
    print(f"Results saved at {results_path}")



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend, dump
from sklearn.metrics import make_scorer, precision_score
import os

def train_and_save_model(prefix, model_details, save_dir='default', cv=5, scoring='precision',
                         task='classification', seed=42):
    np.random.seed(seed)

    model = model_details['model']
    param_grid = model_details['param_grid']

    os.makedirs(f'inst/extdata/{save_dir}', exist_ok=True)

    data = pd.read_csv(f'inst/extdata/{prefix}.csv')
    X = data.drop(columns=['outcome', 'id'])
    y = data['outcome'].astype(float if task == 'regression' else int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dump(scaler, f'inst/extdata/{save_dir}/scaler.joblib')

    if hasattr(model, 'random_state'):
        model.set_params(random_state=seed)

    # Handle undefined precision
    if task == 'classification' and scoring == 'precision':
        scorer = make_scorer(precision_score, zero_division=0)
    else:
        scorer = scoring

    with parallel_backend('loky', inner_max_num_threads=1):
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scorer,
            n_jobs=-1
        )
        grid_search.fit(X_scaled, y)

    model = grid_search.best_estimator_
    dump(model, f'inst/extdata/{save_dir}/model.joblib')



import pandas as pd
from joblib import load
import os

def load_and_predict(prefix, prediction_type='label', save_dir='default', suffix=''):
    # Load the scaler and model
    scaler = load(f'inst/extdata/{save_dir}/scaler.joblib')
    model = load(f'inst/extdata/{save_dir}/model.joblib')
    
    # Load the new dataset for prediction
    X_new = pd.read_csv(f'inst/extdata/{prefix}.csv')
    
    # Assume the new data also excludes 'id' and 'outcome' columns
    X_new = X_new.drop(columns=['id', 'outcome'], errors='ignore')
    
    # Standardize the new dataset using the same scaler
    X_new_scaled = scaler.transform(X_new)

    # Determine type of prediction and process accordingly
    if prediction_type == 'label':
        predictions = model.predict(X_new_scaled)
        result_df = pd.DataFrame(predictions, columns=['prediction'])
        filename = f'predictions_{suffix}.csv'
    elif prediction_type == 'probability':
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_new_scaled)[:, 1]  # Assumes binary classification
            result_df = pd.DataFrame(probabilities, columns=['predicted_probability'])
            filename = f'prob_{suffix}.csv'
        else:
            raise ValueError("This model does not support probability predictions.")
    else:
        raise ValueError("Invalid prediction type specified. Use 'label' or 'probability'.")

    # Save the results to a CSV file
    results_path = f'inst/extdata/{save_dir}/{filename}'
    result_df.to_csv(results_path, index=False)
    print(f"Results saved at {results_path}")



import shap
import pandas as pd
from joblib import load
import os

def compute_shap_values(prefix, explainer_type, save_dir='default', samp_size=100):
    # Load the scaler and model
    scaler_path = f'inst/extdata/{save_dir}/scaler.joblib'
    model_path = f'inst/extdata/{save_dir}/model.joblib'
    scaler = load(scaler_path)
    model = load(model_path)
    
    # Load the new dataset for prediction
    X_new = pd.read_csv(f'inst/extdata/{prefix}.csv')
    X_new = X_new.drop(columns=['id', 'outcome'], errors='ignore')
    X_new_scaled = scaler.transform(X_new)

    # Background data for explainers
    background_data = shap.sample(X_new_scaled, samp_size)  # Adjust as needed

    # Initialize the SHAP explainer
    if explainer_type == shap.LinearExplainer:
        # For LinearExplainer, a masker is required
        masker = shap.maskers.Independent(data=background_data)
        explainer = explainer_type(model, masker)
    elif explainer_type == shap.TreeExplainer:
        explainer = explainer_type(model)
    elif explainer_type == shap.KernelExplainer:
        # KernelExplainer uses the model prediction function and background data
        explainer = explainer_type(model.predict, background_data)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_new_scaled)

    # Handle multiple classes for classifiers
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # Assuming interest in the first class

    # Convert SHAP values to DataFrame and save
    shap_df = pd.DataFrame(shap_values, columns=X_new.columns)
    shap_csv_filename = os.path.join(f'inst/extdata/{save_dir}', f'shap_values.csv')
    shap_df.to_csv(shap_csv_filename, index=False)
    print(f"Results saved at {shap_csv_filename}")



import torch
from torch import nn

class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(64, 32), dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



import numpy as np
import torch

def _set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



import torch

def _get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



import numpy as np

def _prepare_X_y(data, task="classification", outcome_cols=None):
    """
    outcome_cols:
      - None: use all columns starting with 'outcome', or 'outcome' if present.
      - For multi-output, create multiple outcome columns (e.g., outcome1, outcome2).
    """
    if outcome_cols is None:
        outcome_cols = [c for c in data.columns if c.startswith("outcome")]
        if not outcome_cols and "outcome" in data.columns:
            outcome_cols = ["outcome"]

    X = data.drop(columns=outcome_cols + ["id"], errors="ignore")
    y = data[outcome_cols]
    y_arr = y.values
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    y_arr = y_arr.astype(np.float32)
    return X, y_arr, outcome_cols



import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, r2_score


def _train_one_model(X, y, params, task, device, max_epochs=50, score_index=None):
	"""
	Train a single PyTorch model with given params on (X, y) and return (model, score).

	Score:
	  - classification: ROC AUC
	  - regression: R²

	If score_index is not None and there are multiple outputs,
	the metric is computed only on that output.

	Supports mixed-head classification when y[:,0] is multiclass labels (0..K-1)
	and remaining columns y[:,1:] are binary (0/1). In this case model output is:
	  - dict: {"multiclass": logits [N,K], "binary": logits [N,B]}
	  - OR matrix: logits [N, K+B] where first K columns are multiclass logits.
	"""
	input_dim = X.shape[1]

	# split params into model vs. training
	train_param_keys = {"lr", "batch_size", "epochs", "weight_decay", "model_class", "k_classes"}
	model_params = {k: v for k, v in params.items() if k not in train_param_keys}
	lr = params.get("lr", 1e-3)
	batch_size = params.get("batch_size", 32)
	epochs = params.get("epochs", max_epochs)
	weight_decay = params.get("weight_decay", 0.0)

	# mixed-head detection: multiclass labels in first column
	y0 = y[:, 0].astype(float)
	is_mixed = (task == "classification") and (y.shape[1] >= 2) and (np.max(y0) > 1.0)

	if is_mixed:
		k_classes = int(params.get("k_classes", 0))
		if k_classes < 2:
			raise ValueError("Mixed-head training requires params['k_classes'] >= 2.")
		bin_dim = int(y.shape[1] - 1)
		output_dim = k_classes + bin_dim
	else:
		output_dim = y.shape[1]

	model_class = params.get("model_class", MultiOutputMLP)
	model = model_class(input_dim=input_dim, output_dim=output_dim, **model_params)
	model.to(device)

	if task == "classification":
		if is_mixed:
			criterion_mc = nn.CrossEntropyLoss()
			criterion_bin = nn.BCEWithLogitsLoss()
		else:
			criterion = nn.BCEWithLogitsLoss()
	else:
		criterion = nn.MSELoss()

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

	X_tensor = torch.from_numpy(X.astype(np.float32))
	y_tensor = torch.from_numpy(y.astype(np.float32))
	dataset = TensorDataset(X_tensor, y_tensor)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	for _ in range(epochs):
		model.train()
		for xb, yb in loader:
			xb = xb.to(device)
			yb = yb.to(device)
			optimizer.zero_grad()
			out = model(xb)

			if task == "classification":
				if is_mixed:
					y_mc = yb[:, 0].long()
					y_bin = yb[:, 1:]

					if isinstance(out, dict):
						mc_logits = out.get("multiclass", None)
						bin_logits = out.get("binary", None)
						if mc_logits is None or bin_logits is None:
							raise ValueError("Model output dict must contain 'multiclass' and 'binary'.")
					else:
						if out.shape[1] != output_dim:
							raise ValueError("Model output_dim does not match expected K+B.")
						mc_logits = out[:, :k_classes]
						bin_logits = out[:, k_classes:]

					loss = criterion_mc(mc_logits, y_mc)
					if y_bin.shape[1] > 0:
						loss = loss + criterion_bin(bin_logits, y_bin)
				else:
					loss = criterion(out, yb)
			else:
				loss = criterion(out, yb)

			loss.backward()
			optimizer.step()

	# evaluate on training data
	model.eval()
	with torch.no_grad():
		X_eval = X_tensor.to(device)
		out = model(X_eval)

	if task == "classification":
		if is_mixed:
			if isinstance(out, dict):
				mc_logits = out.get("multiclass", None)
				bin_logits = out.get("binary", None)
				if mc_logits is None or bin_logits is None:
					raise ValueError("Model output dict must contain 'multiclass' and 'binary'.")
				mc_logits_np = mc_logits.detach().cpu().numpy()
				bin_logits_np = bin_logits.detach().cpu().numpy()
			else:
				out_np = out.detach().cpu().numpy()
				mc_logits_np = out_np[:, :k_classes]
				bin_logits_np = out_np[:, k_classes:]

			# softmax for multiclass
			mc_max = np.max(mc_logits_np, axis=1, keepdims=True)
			mc_exp = np.exp(mc_logits_np - mc_max)
			mc_probs = mc_exp / np.sum(mc_exp, axis=1, keepdims=True)

			# sigmoid for binary heads
			if bin_logits_np.size:
				bin_probs = 1 / (1 + np.exp(-bin_logits_np))
				probs = np.concatenate([mc_probs, bin_probs], axis=1)
			else:
				probs = mc_probs

			if score_index is not None:
				if score_index == 0:
					y_true = y[:, 0].astype(int)
					y_score = mc_probs
					score = roc_auc_score(
						y_true,
						y_score,
						multi_class="ovr",
						average="macro"
					)
				else:
					bin_j = score_index - 1
					y_true = y[:, score_index]
					y_score = probs[:, k_classes + bin_j]
					score = roc_auc_score(y_true, y_score)
			else:
				y_true = y[:, 0].astype(int)
				y_score = mc_probs
				score = roc_auc_score(
					y_true,
					y_score,
					multi_class="ovr",
					average="macro"
				)

		else:
			outputs_np = out.detach().cpu().numpy()
			probs = 1 / (1 + np.exp(-outputs_np))

			if probs.shape[1] == 1 or score_index is not None:
				if score_index is None or probs.shape[1] == 1:
					y_true = y.ravel()
					y_score = probs.ravel()
				else:
					y_true = y[:, score_index]
					y_score = probs[:, score_index]
				score = roc_auc_score(y_true, y_score)
			else:
				score = roc_auc_score(y, probs, average="macro")

	else:
		outputs_np = out.detach().cpu().numpy()

		if outputs_np.shape[1] == 1 or score_index is not None:
			if score_index is None or outputs_np.shape[1] == 1:
				y_true = y.ravel()
				y_pred = outputs_np.ravel()
			else:
				y_true = y[:, score_index]
				y_pred = outputs_np[:, score_index]
			score = r2_score(y_true, y_pred)
		else:
			score = r2_score(y, outputs_np, multioutput="variance_weighted")

	return model, score



import os
import csv
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, ParameterGrid

def train_and_estimate_sample_size_torch(prefix, model_details, save_dir="default",
                                         task="classification", seed=42,
                                         proportions=None, device=None,
                                         score_outcome=None):
    """
    Rough sample size trajectory using a PyTorch model (1+ outputs).

    - prefix: path under inst/extdata without .csv
      e.g. 'predmod_data/.../train'
    - model_details:
        {
          "model_class": MultiOutputMLP,
          "param_grid": {
              "hidden_sizes": [(128, 64), (64, 32)],
              "dropout": [0.0, 0.3],
              "lr": [1e-3],
              "batch_size": [32],
              "epochs": [50],
              "weight_decay": [0.0]
          }
        }
    """
    _set_seeds(seed)
    device = _get_device(device)

    save_path = os.path.join("inst", "extdata", save_dir)
    os.makedirs(save_path, exist_ok=True)

    data = pd.read_csv(f"inst/extdata/{prefix}.csv")
    X, y, outcome_cols = _prepare_X_y(data, task=task)

    # choose which outcome to score on (if multi-output)
    if score_outcome is None:
        score_index = None
    else:
        if isinstance(score_outcome, str):
            score_index = outcome_cols.index(score_outcome)
        else:
            score_index = int(score_outcome)
    
    num_predictors = X.shape[1]
    if proportions is None:
        proportions = np.linspace(0.1, 0.9, 9)

    results = []

    # choose splitter
    if task == "classification":
        splitter = StratifiedShuffleSplit(n_splits=1, random_state=seed)
        # use first outcome column for stratification
        strat_y = y[:, 0].astype(int)
    else:
        splitter = ShuffleSplit(n_splits=1, random_state=seed)

    param_grid = model_details["param_grid"]
    model_class = model_details.get("model_class", MultiOutputMLP)
    base_params = {"model_class": model_class}

    for prop in proportions:
        splitter.test_size = prop
        if task == "classification":
            split_iter = splitter.split(X, strat_y)
        else:
            split_iter = splitter.split(X)

        for train_index, test_index in split_iter:
            X_train = X.iloc[train_index].values
            y_train = y[train_index]

            # EPV (multi-output: use worst-case minority count)
            if task == "classification":
                events_per_output = y_train.sum(axis=0)
                non_events_per_output = len(y_train) - events_per_output
                event_count = np.minimum(events_per_output, non_events_per_output).min()
                epv = event_count / num_predictors
            else:
                epv = len(y_train) / num_predictors

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            best_score = -np.inf

            for params in ParameterGrid(param_grid):
                all_params = {**base_params, **params}
                _, score = _train_one_model(
                    X_train_scaled, y_train, all_params,
                    task=task, device=device,
                    score_index=score_index
                )
                if score > best_score:
                    best_score = score
            
            results.append((prop, epv, best_score))

    results_path = os.path.join(save_path, "sample_size_estimation_torch.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["proportion", "epv", "score"])
        writer.writerows(results)

    print(f"Results saved at {results_path}")



import os
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from joblib import dump

def train_and_save_model_torch(prefix, model_details, save_dir="default",
                               task="classification", seed=42,
                               device=None, score_outcome=None):
    """
    Train final PyTorch model (1+ outputs) on full data and save scaler + model.
    """
    _set_seeds(seed)
    device = _get_device(device)

    save_path = os.path.join("inst", "extdata", save_dir)
    os.makedirs(save_path, exist_ok=True)

    data = pd.read_csv(f"inst/extdata/{prefix}.csv")
    X, y, outcome_cols = _prepare_X_y(data, task=task)

    if score_outcome is None:
        score_index = None
    else:
        if isinstance(score_outcome, str):
            score_index = outcome_cols.index(score_outcome)
        else:
            score_index = int(score_outcome)

    # scale on full data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    dump(scaler, os.path.join(save_path, "scaler.joblib"))

    param_grid = model_details["param_grid"]
    model_class = model_details.get("model_class", MultiOutputMLP)
    base_params = {"model_class": model_class}

    best_score = -np.inf
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        all_params = {**base_params, **params}
        model, score = _train_one_model(
            X_scaled, y, all_params,
            task=task, device=device,
            score_index=score_index
        )
        if score > best_score:
            best_score = score
            best_model = model
            best_params = all_params
    
    # save best model (full nn.Module for simplicity)
    model_path = os.path.join(save_path, "model.pt")
    torch.save(best_model.state_dict(), model_path)

    meta = {
        "input_dim": X.shape[1],
        "output_dim": y.shape[1],
        "outcome_cols": outcome_cols,
        "best_params": best_params,
        "task": task,
    }
    dump(meta, os.path.join(save_path, "model_meta.joblib"))

    print(f"Model and scaler saved in {save_path}")



import os
import numpy as np
import pandas as pd
import torch

from joblib import load

def load_and_predict_torch(prefix, prediction_type="label",
                           save_dir="default", suffix="",
                           device=None):
    """
    Load PyTorch model + scaler and generate predictions.

    - prediction_type:
        * 'label' or 'probability' for classification
        * 'value' (or 'label' as alias) for regression
    """
    device = _get_device(device)

    save_path = os.path.join("inst", "extdata", save_dir)
    scaler = load(os.path.join(save_path, "scaler.joblib"))
    meta = load(os.path.join(save_path, "model_meta.joblib"))
    task = meta.get("task", "classification")
    
    best_params = meta["best_params"]
    model_class = best_params.get("model_class", MultiOutputMLP)
    
    # same train_param_keys as in _train_one_model
    train_param_keys = {"lr", "batch_size", "epochs", "weight_decay", "model_class"}
    model_params = {k: v for k, v in best_params.items() if k not in train_param_keys}
    
    model = model_class(
        input_dim=meta["input_dim"],
        output_dim=meta["output_dim"],
        **model_params
    )
    
    state_dict = torch.load(
        os.path.join(save_path, "model.pt"),
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)

    # load new data
    data = pd.read_csv(f"inst/extdata/{prefix}.csv")
    outcome_cols = meta.get("outcome_cols", [])
    X_new = data.drop(columns=outcome_cols + ["id"], errors="ignore")
    X_new_scaled = scaler.transform(X_new.values.astype(np.float32))

    model.to(device)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_new_scaled).to(device)
        outputs = model(X_tensor)
        outputs_np = outputs.detach().cpu().numpy()

    if task == "classification":
        logits = outputs_np
        probs = 1 / (1 + np.exp(-logits))
        if prediction_type == "label":
            labels = (probs >= 0.5).astype(int)
            if labels.shape[1] == 1:
                result_df = pd.DataFrame(labels, columns=["prediction"])
            else:
                cols = [f"prediction_{i+1}" for i in range(labels.shape[1])]
                result_df = pd.DataFrame(labels, columns=cols)
            filename = f"predictions_{suffix}.csv"
        elif prediction_type == "probability":
            if probs.shape[1] == 1:
                result_df = pd.DataFrame(probs.ravel(), columns=["predicted_probability"])
            else:
                cols = [f"predicted_probability_{i+1}" for i in range(probs.shape[1])]
                result_df = pd.DataFrame(probs, columns=cols)
            filename = f"prob_{suffix}.csv"
        else:
            raise ValueError(
                "Invalid prediction_type for classification; use 'label' or 'probability'."
            )
    else:
        # regression
        if prediction_type not in ("value", "label"):
            prediction_type = "value"
        if outputs_np.shape[1] == 1:
            result_df = pd.DataFrame(outputs_np.ravel(), columns=["prediction"])
        else:
            cols = [f"prediction_{i+1}" for i in range(outputs_np.shape[1])]
            result_df = pd.DataFrame(outputs_np, columns=cols)
        filename = f"predictions_{suffix}.csv"

    results_path = os.path.join(save_path, filename)
    result_df.to_csv(results_path, index=False)
    print(f"Results saved at {results_path}")



import os
import numpy as np
import pandas as pd
import torch
from joblib import load
import shap

def compute_shap_values_torch(
	prefix,
	explainer_type,
	save_dir="default",
	samp_size=100,
	device=None,
):
	# paths
	save_path = os.path.join("inst", "extdata", save_dir)
	scaler_path = os.path.join(save_path, "scaler.joblib")
	model_path = os.path.join(save_path, "model.pt")
	meta_path = os.path.join(save_path, "model_meta.joblib")

	# load scaler + meta
	scaler = load(scaler_path)
	meta = load(meta_path)

	input_dim = int(meta["input_dim"])
	output_dim = int(meta.get("output_dim", 1))
	task = meta.get("task", "classification")
	best_params = meta.get("best_params", {}) or {}
	model_class = best_params.get("model_class", MultiOutputMLP)

	# rebuild torch model
	model_param_keys = {"hidden_sizes", "dropout"}
	model_kwargs = {k: best_params[k] for k in model_param_keys if k in best_params}

	model = model_class(
		input_dim=input_dim,
		output_dim=output_dim if task != "classification" else output_dim,
		**model_kwargs,
	)

	# device
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	device = torch.device(device)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()

	# load data
	X_new = pd.read_csv(f"inst/extdata/{prefix}.csv")
	X_new = X_new.drop(columns=["id"], errors="ignore")
	X_new = X_new.drop(columns=[c for c in X_new.columns if c.startswith("outcome")], errors="ignore")

	X_new_scaled = scaler.transform(X_new.values).astype(np.float32)

	# background
	if samp_size is None:
		bg = X_new_scaled
	else:
		bg = shap.sample(
			X_new_scaled,
			samp_size,
			random_state=0,
		)

	bg_t = torch.from_numpy(bg).to(device)
	X_t = torch.from_numpy(X_new_scaled).to(device)

	# model wrapper for KernelExplainer
	def _predict_np(x_np):
		x_t = torch.from_numpy(np.asarray(x_np, dtype=np.float32)).to(device)
		with torch.no_grad():
			out = model(x_t).detach().cpu().numpy()
		if task == "classification":
			# keep logits -> SHAP can explain logits; if you prefer probs, apply sigmoid outside.
			return out
		return out

	# explainer init
	if explainer_type in (shap.DeepExplainer, shap.GradientExplainer):
		explainer = explainer_type(model, bg_t)
		shap_values = explainer.shap_values(X_t)
	elif explainer_type == shap.KernelExplainer:
		explainer = explainer_type(_predict_np, bg)
		shap_values = explainer.shap_values(X_new_scaled)
	else:
		raise ValueError("Unsupported explainer_type for torch model.")

	# match your previous behavior: if list (multi-output), take the first output
	if isinstance(shap_values, list):
		shap_values = shap_values[0]

	# convert + save
	# keep id for merge
	data_full = pd.read_csv(
		f"inst/extdata/{prefix}.csv"
	)

	id_col = data_full["id"] \
		if "id" in data_full.columns \
		else None

	shap_df = pd.DataFrame(
		np.asarray(shap_values),
		columns=X_new.columns,
	)

	if id_col is not None:
		shap_df.insert(
			0,
			"id",
			id_col.values,
		)

	outfile = os.path.join(
		save_path,
		"shap_values.csv",
	)

	shap_df.to_csv(
		outfile,
		index=False,
	)

	print(f"Results saved at {outfile}")



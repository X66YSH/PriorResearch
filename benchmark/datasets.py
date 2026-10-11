"""The two test suites for RQ1: finance tables and general tables, both binary tasks from TabArena v0.1.

Finance = credit risk, default or bankruptcy, i.e. what the FinTFM finance prior simulates.
Banking tables that are not credit risk (churn, marketing, insurance) go into neither suite, so the
general suite does not quietly contain finance-like tables. Moving them is a one-line change below.

Protocol (adapted from the nanoTabPFN paper's experiment.ipynb):
  - per dataset, N_REPEATS stratified subsamples of N_ROWS rows (the size the model is pretrained on)
  - categorical columns ordinal-encoded, constant columns dropped
  - more than MAX_FEATURES features: a random subset of MAX_FEATURES, redrawn per repeat
  - missing values are kept as NaN here and imputed per fold in evaluate.py (train-row means only)
  - datasets whose minority class is under 2.5% are skipped, as in the notebook: 256 rows would hold
    too few positives for 5-fold CV
Prepared subsamples are cached in benchmark/cache/ so every checkpoint sees exactly the same tables.
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder

N_ROWS = 256
MAX_FEATURES = 24
N_REPEATS = 3
MIN_MINORITY_PERCENT = 2.5
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# OpenML task ids, TabArena v0.1 (suite 457), binary classification only
FINANCE = {
    363694: "polish_companies_bankruptcy",
    363706: "taiwanese_bankruptcy_prediction",
    363626: "credit-g",
    363627: "credit_card_clients_default",
    363673: "GiveMeSomeCredit",
    363676: "heloc",
    363682: "Is-this-a-good-customer",
}
BANKING_NOT_CREDIT = {  # in neither suite, pending a decision
    363619: "Bank_Customer_Churn",
    363618: "bank-marketing",
    363624: "coil2000_insurance_policies",
    363684: "Marketing_Campaign",
}
GENERAL = {
    363616: "APSFailure",
    363613: "Amazon_employee_access",
    363620: "Bioresponse",
    363630: "Diabetes130US",
    363632: "E-CommereShippingData",
    363671: "Fitness_Club",
    363679: "HR_Analytics_Job_Change_of_Data_Scientists",
    363689: "NATICUSdroid",
    363621: "blood-transfusion-service-center",
    363623: "churn",
    363628: "customer_satisfaction_in_airline",
    363629: "diabetes",
    363674: "hazelnut-spread-contaminant-detection",
    363681: "in_vehicle_coupon_recommendation",
    363712: "jm1",
    363683: "kddcup09_appetency",
    363691: "online_shoppers_intention",
    363696: "qsar-biodeg",
    363700: "seismic-bumps",
}
SUITES = {"finance": FINANCE, "general": GENERAL}


def get_feature_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Copied from experiment.ipynb: ordinal-encodes categorical columns, keeps NaN, drops constant columns."""
    num_mask, cat_mask = [], []
    for col in X:
        if len(X[col].dropna().unique()) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors="coerce").notna().sum()  # numeric columns stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
    num_transformer = Pipeline([
        ("to_numeric", FunctionTransformer(lambda x: pd.DataFrame(x).apply(pd.to_numeric, errors="coerce").to_numpy())),
    ])
    cat_transformer = Pipeline([
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
    ])
    return ColumnTransformer([("num", num_transformer, np.array(num_mask)), ("cat", cat_transformer, np.array(cat_mask))])


def prepare_dataset(task_id: int) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Returns N_REPEATS (X, y) subsamples for one task, or None if its minority class is too rare."""
    import openml  # only needed to build the cache; benchmark/cache is shipped, so most runs never import it

    task = openml.tasks.get_task(task_id, download_splits=False)
    dataset = task.get_dataset(download_data=False)
    if dataset.qualities["MinorityClassPercentage"] < MIN_MINORITY_PERCENT:
        return None
    X_full, y_full, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
    y_full = LabelEncoder().fit_transform(y_full)
    repeats = []
    for repeat in range(N_REPEATS):
        if len(y_full) > N_ROWS:
            _, X, _, y = train_test_split(X_full, y_full, test_size=N_ROWS, stratify=y_full, random_state=repeat)
        else:
            X, y = X_full, y_full
        X = X.reset_index(drop=True)
        X = get_feature_preprocessor(X).fit_transform(X).astype(np.float32)
        if X.shape[1] > MAX_FEATURES:
            columns = np.random.default_rng(repeat).choice(X.shape[1], MAX_FEATURES, replace=False)
            X = X[:, np.sort(columns)]
        repeats.append((X, np.asarray(y, dtype=np.int64)))
    return repeats


def load_suite(suite: str) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """name -> list of (X, y) repeats, read from the cache when available."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    datasets = {}
    for task_id, name in SUITES[suite].items():
        path = os.path.join(CACHE_DIR, f"{task_id}_{name}.npz")
        if not os.path.exists(path):
            repeats = prepare_dataset(task_id)
            if repeats is None:
                np.savez(path, skipped=True)
            else:
                np.savez(path, **{f"X{i}": X for i, (X, _) in enumerate(repeats)}, **{f"y{i}": y for i, (_, y) in enumerate(repeats)})
        data = np.load(path)
        if "skipped" in data:
            continue
        datasets[name] = [(data[f"X{i}"], data[f"y{i}"]) for i in range(N_REPEATS)]
    return datasets

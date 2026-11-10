"""Train the final XGBoost model and save it to model.bin.

Usage:
    uv run python train.py
"""

import pickle

import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import features as F

SEED = 1
OUTPUT_FILE = "model.bin"

# Selected in notebook.ipynb (section 5.4)
XGB_PARAMS = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 2,
    "seed": SEED,
}
NUM_ROUNDS = 207


def train(X_dicts, y):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(X_dicts)
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(dv.get_feature_names_out()))
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_ROUNDS)
    return dv, model


def predict(dv, model, X_dicts):
    X = dv.transform(X_dicts)
    return model.predict(xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out())))


def main():
    print("Loading and preparing data...")
    df = F.clean(F.load_raw())
    X = F.prepare_features(df)
    y = df["readmitted_30"].values
    groups = df["patient_nbr"].values

    # Same split as the notebook: 80% train (train + validation), 20% test, by patient
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))

    print(f"Training on {len(train_idx)} rows, testing on {len(test_idx)} rows...")
    dv, model = train(F.to_dicts(X.iloc[train_idx]), y[train_idx])
    y_pred = predict(dv, model, F.to_dicts(X.iloc[test_idx]))
    print(f"Test ROC AUC: {roc_auc_score(y[test_idx], y_pred):.4f}")
    print(f"Test PR AUC:  {average_precision_score(y[test_idx], y_pred):.4f}")

    print("Training the final model on all data...")
    dv, model = train(F.to_dicts(X), y)

    with open(OUTPUT_FILE, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Model saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

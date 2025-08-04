import pandas as pd, joblib, json, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from pathlib import Path

MODEL_MAP = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

def train_models(csv_path, target, model_names, model_dir: Path, pre):
    df = pd.read_csv(csv_path)
    X, y = df.drop(columns=[target]), df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    for name in model_names:
        clf = MODEL_MAP[name]
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label="Yes", zero_division=0)
        rec  = recall_score(y_test, preds, pos_label="Yes", zero_division=0)
        f1   = f1_score(y_test, preds, pos_label="Yes", zero_division=0)
        
        cm   = confusion_matrix(y_test, preds).tolist()

        # save
        file_name = f"{name}.joblib"
        joblib.dump(clf, model_dir / file_name)
        joblib.dump(pre, model_dir / f"{name}_pipeline.joblib")


        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm
        })
    return results
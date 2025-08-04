import pandas as pd, joblib, os
from pathlib import Path
from .llm import generate_summary

MODEL_DIR = Path("app/models")

def batch_predict(test_df: pd.DataFrame, model_name: str, model_dir: Path):
    model_path = model_dir / f"{model_name}.joblib"
    pipeline_path= model_dir / f"{model_name}_pipeline.joblib"
    
    if not model_path.exists():
        raise ValueError("Model or preprocessing not found")

    model = joblib.load(model_path)
    pre = joblib.load(pipeline_path)
    
    X = test_df.copy()
    X_transformed = pre.transform(X)

    # Basic sanity: ensure same columns (except target)
    # In a real project we would reuse the preprocessing pipeline.
    preds = model.predict(X_transformed)

    test_df["prediction"] = preds
    summaries = [generate_summary(test_df.iloc[i], preds[i]) for i in range(len(preds))]
    test_df["explanation"] = summaries
    return test_df
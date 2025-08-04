from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd, joblib
import os, uuid
from pathlib import Path
from .schemas import *
from .services.cleaning import clean_df
from .services.training import train_models
from .services.inference import batch_predict
from app.utils import latest_session

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("app/models")
MODEL_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Customer-Churn FastAPI + GenAI")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# expose /templates/** as static (for css/js/png etc.)
app.mount("/static", StaticFiles(directory="templates"), name="static")

# serve index.html on root
@app.get("/", response_class=FileResponse)
async def serve_ui():
    return FileResponse("templates/index.html")

# ---------- 1. Upload ----------
@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), target: str = Form(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    session_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{session_id}.csv"
    with open(save_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_csv(save_path)
    if target not in df.columns:
        raise HTTPException(400, f"Target column '{target}' not found")

    meta = {
        "columns": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "rows": len(df),
        "preview": df.head(3).to_dict(orient="records")
    }
    # Persist target name
    (UPLOAD_DIR / f"{session_id}.target").write_text(target)
    return {"session_id": session_id, **meta}

# ---------- 2. Cleaning ----------
@app.post("/data_cleaning", response_model=CleaningResponse)
async def clean(req: CleaningRequest):
    session_id = req.session_id or latest_session()
    csv_path = UPLOAD_DIR / f"{session_id}.csv"
    target_path = UPLOAD_DIR / f"{session_id}.target"
    if not csv_path.exists():
        raise HTTPException(404, "Session not found")
    target = target_path.read_text()

    df = pd.read_csv(csv_path)
    cleaned, pre = clean_df(df, target, req)
    cleaned.columns = cleaned.columns.astype(str) 
    joblib.dump(pre, MODEL_DIR / f"{session_id}_pipeline.joblib")
    cleaned.to_csv(UPLOAD_DIR / f"{session_id}_cleaned.csv", index=False)
    return {"preview": cleaned.head(3).to_dict(orient="records")}

# ---------- 3. Train ----------
@app.post("/train_model", response_model=TrainResponse)
async def train(req: TrainRequest):
    session_id = req.session_id or latest_session()
    csv_path = UPLOAD_DIR / f"{session_id}_cleaned.csv"
    target_path = UPLOAD_DIR / f"{session_id}.target"
    pre_path = MODEL_DIR / f"{session_id}_pipeline.joblib"
    pre = joblib.load(pre_path)
    if not csv_path.exists():
        raise HTTPException(404, "Cleaned dataset not found")
    target = target_path.read_text()

    results = train_models(csv_path, target, req.models, MODEL_DIR, pre)
    return {"results": results}

# ---------- 4. Predict ----------
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    test_df = pd.read_csv(file.file)
    out_df = batch_predict(test_df, model_name, MODEL_DIR)
    outfile = Path(UPLOAD_DIR / f"predictions.csv")
    out_df.to_csv(outfile, index=False)
    return FileResponse(outfile, media_type="text/csv", filename="predictions.csv")
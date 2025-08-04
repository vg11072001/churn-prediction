from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

class UploadResponse(BaseModel):
    session_id: str
    columns: Dict[str, str]
    missing: Dict[str, int]
    rows: int
    preview: List[Dict[str, Any]]

class CleaningRequest(BaseModel):
    session_id: Optional[str] = None
    missing_strategy: Literal["drop", "impute"] = "impute"
    encode: Literal["label", "onehot"] = "label"
    scale: Literal["standard", "minmax", "none"] = "standard"

class CleaningResponse(BaseModel):
    preview: List[Dict[str, Any]]

class TrainRequest(BaseModel):
    session_id: Optional[str] = None
    models: List[Literal["LogisticRegression", "RandomForest", "XGBoost"]]

class TrainResponse(BaseModel):
    results: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions_csv: str
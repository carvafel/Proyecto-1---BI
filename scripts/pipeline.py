# scripts/pipeline.py
from pathlib import Path
import json
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# columnas del proyecto
TEXT_COL = "textos"
LABEL_COL = "labels"

# rutas por defecto
MODEL_PATHS_TRY = [
    "artifacts/model_v2.pkl",     # modelo reentrenado
    "artifacts/best_model.joblib",
    "best_model.joblib"
]
FINAL_MODEL_PATH = "artifacts/model_v2.pkl"

# "almacén" de entrenamiento acumulado
DATA_STORE = "data/training_store.csv"
SEED_DATA = "data/Datos_etapa2_aumentado.csv"  # si existe, lo usamos para inicializar
EXTRA_SOURCES = [
"data/datosetapa2.xlsx",
"data/Datos_proyecto.xlsx",
]

# en scripts/pipeline.py (arriba de refit_existing_pipeline)

def load_model() -> Pipeline:
    for p in MODEL_PATHS_TRY:
        if Path(p).exists():
            return joblib.load(p)
    raise FileNotFoundError("No encontré un modelo. Esperaba uno en artifacts/model_v2.pkl o best_model.joblib")

def save_model(model: Pipeline, path: str = FINAL_MODEL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def ensure_data_store() -> str:
    Path("data").mkdir(parents=True, exist_ok=True)
    if not Path(DATA_STORE).exists():
        pd.DataFrame(columns=[TEXT_COL, LABEL_COL]).to_csv(DATA_STORE, index=False)
    return DATA_STORE

def load_store_df() -> pd.DataFrame:
    ensure_data_store()
    df = pd.read_csv(DATA_STORE)
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"{DATA_STORE} debe tener columnas [{TEXT_COL}, {LABEL_COL}]")
    return df

def append_to_store(df_new: pd.DataFrame) -> pd.DataFrame:
    df_store = load_store_df()
    df_all = pd.concat([df_store, df_new[[TEXT_COL, LABEL_COL]]], ignore_index=True)
    df_all[TEXT_COL] = df_all[TEXT_COL].astype(str).str.strip()
    df_all[LABEL_COL] = df_all[LABEL_COL].astype(str).str.strip()
    df_all = df_all.dropna().drop_duplicates(subset=[TEXT_COL, LABEL_COL])
    df_all.to_csv(DATA_STORE, index=False)
    return df_all

def evaluate(model: Pipeline, df: pd.DataFrame) -> dict:
    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].astype(str)
    y_pred = pd.Series(model.predict(X)).astype(str).values
    return {
        "precision": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, y_pred)),
    }

def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=[TEXT_COL, LABEL_COL])
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        return pd.DataFrame(columns=[TEXT_COL, LABEL_COL])
    
    cols_lower = {c.lower(): c for c in df.columns}
    tcol = cols_lower.get("textos", cols_lower.get("texto", TEXT_COL))
    lcol = cols_lower.get("labels", cols_lower.get("label", LABEL_COL))
    if tcol not in df.columns or lcol not in df.columns:
        return pd.DataFrame(columns=[TEXT_COL, LABEL_COL])

    out = df[[tcol, lcol]].rename(columns={tcol: TEXT_COL, lcol: LABEL_COL}).copy()
    out[TEXT_COL] = out[TEXT_COL].astype(str).str.strip()
    out[LABEL_COL] = out[LABEL_COL].astype(str).str.strip()
    out = out.dropna()
    # descartar textos muy cortos
    out = out[out[TEXT_COL].str.len() > 5]
    return out

def assemble_training_frame(extra_paths: Optional[List[str]] = None, include_store: bool = True) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    if include_store:
        try:
            parts.append(load_store_df())
        except Exception:
            pass
    for path in (extra_paths or EXTRA_SOURCES):
        dfp = _read_any(path)
        if not dfp.empty:
            parts.append(dfp)
    if not parts:
        return pd.DataFrame(columns=[TEXT_COL, LABEL_COL])
    df_all = pd.concat(parts, ignore_index=True)
    df_all[TEXT_COL] = df_all[TEXT_COL].astype(str).str.strip()
    df_all[LABEL_COL] = df_all[LABEL_COL].astype(str).str.strip()
    df_all = df_all.dropna().drop_duplicates(subset=[TEXT_COL, LABEL_COL])
    return df_all

def train_model(df: pd.DataFrame, min_df: int | float = 1, max_df: int | float = 1.0,
                ngram_range: tuple = (1, 2), stop_words=None) -> Pipeline:
    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].astype(str)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range, stop_words=stop_words)),
        ("clf", SVC(kernel="linear", probability=True)),
    ])
    pipe.fit(X, y)
    return pipe

def refit_existing_pipeline(model: Pipeline, df: pd.DataFrame) -> Pipeline:
    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].astype(str)
    model.fit(X, y)
    return model


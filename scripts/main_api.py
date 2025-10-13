from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd


from .pipeline import (
    load_model, save_model, evaluate,
    refit_existing_pipeline, append_to_store, ensure_data_store,
    assemble_training_frame, train_model,
    TEXT_COL, LABEL_COL, FINAL_MODEL_PATH,
)

app = FastAPI(title="ODS Text Classification API", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()
ensure_data_store()

class PredictItem(BaseModel):
    textos: str = Field(..., description="Texto a clasificar")

class RetrainItem(BaseModel):
    textos: str = Field(..., description="Texto etiquetado")
    labels: str = Field(..., description="Etiqueta ODS (1, 3, 4)")

@app.get("/health")
def health() -> Dict[str, Any]:
    classes = getattr(model, "classes_", [])
    return {
        "status": "ok",
        "model_path": FINAL_MODEL_PATH,
        "classes": list(map(str, classes)),
        "text_col": TEXT_COL,
        "label_col": LABEL_COL
    }

@app.post("/predict")
def predict(items: List[PredictItem]):
    textos = [it.textos for it in items]
    preds = model.predict(textos)
    probs = None
    classes = list(getattr(model, "classes_", []))
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(textos)
        except Exception:
            probs = None
    results = []
    for i, t in enumerate(textos):
        row = {"texto": t, "prediccion": str(preds[i])}
        if probs is not None:
            row["probabilidades"] = {str(classes[j]): float(probs[i][j]) for j in range(len(classes))}
        results.append(row)
    return {"resultados": results}

@app.post("/retrain")
def retrain(items: List[RetrainItem]):
    global model
    # 1) guardar los nuevos datos en el store incremental
    data = [{TEXT_COL: it.textos, LABEL_COL: it.labels} for it in items if it.textos and it.labels]
    df_new = pd.DataFrame(data)
    if df_new.empty:
        return {"error": f"Debes enviar al menos un objeto con campos '{TEXT_COL}' y '{LABEL_COL}'."}

    df_new[TEXT_COL] = df_new[TEXT_COL].astype(str).str.strip()
    df_new[LABEL_COL] = df_new[LABEL_COL].astype(str).str.strip()


    # Clases de referencia para "nuevos_clasificados"
    classes_ref = list(map(str, getattr(model, "classes_", []))) or sorted(df_new[LABEL_COL].unique().tolist())

    # Conteo de nuevos por clase (tal cual llegó en el payload)
    vc = df_new[LABEL_COL].astype(str).value_counts()
    nuevos_clasificados = {lbl: int(vc.get(lbl, 0)) for lbl in classes_ref}

    # 2) acumular en store (append + drop_duplicates)
    df_store = append_to_store(df_new)


    # 3) armar el dataset de ENTRENAMIENTO (training_store + datosetapa2 + Datos_proyecto)
    df_train = assemble_training_frame(include_store=True)
    if df_train.empty:
        return {"error": "No hay datos para entrenar (train set vacío)."}

    # 4) reentrenar (refit si se puede; si no, fallback con TF-IDF seguro)
    
    try:
        model = refit_existing_pipeline(model, df_train)
    except ValueError as e:
        msg = str(e)
        if ("max_df corresponds to < documents than min_df" in msg) or ("empty vocabulary" in msg):
            model = train_model(df_train, min_df=1, max_df=1.0)  # stop_words=None por compatibilidad
        else:
            raise e  


    save_model(model)

    # 5) métricas y distribución final sobre el train total
    metrics = evaluate(model, df_train)
    dist = df_train[LABEL_COL].astype(str).value_counts().sort_index().to_dict()


    return {
        "mensaje": "Modelo reentrenado y guardado con training_store + datosetapa2 + Datos_proyecto",
        "metricas_macro": metrics,
        "instancias_entrenamiento": int(len(df_train)),
        "distribucion_clases": {str(k): int(v) for k, v in dist.items()},
        "nuevos_clasificados": nuevos_clasificados
    }




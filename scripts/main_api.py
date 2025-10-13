from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import Request, Form, UploadFile, File, status
from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import json
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

# Sesiones para modo admin
SECRET_KEY = os.environ.get("ADMIN_PASSWORD", "admin")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

templates = Jinja2Templates(directory="app/templates")

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




##############################
# UI (interfaz web básica)
##############################

def _is_admin(request: Request) -> bool:
    return bool(request.session.get("is_admin", False))


@app.get("/", response_class=HTMLResponse)
def ui_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "results": None, "batch_table": None, "error": None})


@app.post("/ui/predict", response_class=HTMLResponse)
def ui_predict(request: Request, texto_single: str = Form("")):
    textos = [texto_single.strip()] if texto_single and texto_single.strip() else []
    results = []
    if textos:
        preds = model.predict(textos)
        probs = None
        classes = list(getattr(model, "classes_", []))
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(textos)
            except Exception:
                probs = None
        row = {"texto": textos[0], "prediccion": str(preds[0])}
        if probs is not None:
            row["probabilidades"] = {str(classes[j]): float(probs[0][j]) for j in range(len(classes))}
        results = [row]
    return templates.TemplateResponse("home.html", {"request": request, "results": results, "batch_table": None, "error": None})


@app.post("/ui/predict-batch", response_class=HTMLResponse)
def ui_predict_batch(request: Request, file: UploadFile = File(...)):
    try:
        if file.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file)
        else:
            df = pd.read_csv(file.file)
    except Exception as e:
        return templates.TemplateResponse("home.html", {"request": request, "error": f"No pude leer el archivo: {e}", "results": None, "batch_table": None})

    if "textos" not in df.columns:
        return templates.TemplateResponse("home.html", {"request": request, "error": "El archivo debe contener la columna 'textos'", "results": None, "batch_table": None})

    textos = df[TEXT_COL].astype(str).fillna("").tolist()
    preds = model.predict(textos)
    classes = list(getattr(model, "classes_", []))
    probas = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(textos)
        except Exception:
            probas = None

    out_rows = []
    for i, t in enumerate(textos):
        row = {"texto": t, "prediccion": str(preds[i])}
        if probas is not None:
            row["probabilidades"] = {str(classes[j]): float(probas[i][j]) for j in range(len(classes))}
        out_rows.append(row)

    batch_table = [{"texto": r["texto"], "prediccion": r["prediccion"], **(r.get("probabilidades") or {})} for r in out_rows]
    return templates.TemplateResponse("home.html", {"request": request, "results": None, "batch_table": batch_table, "error": None})


@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    if _is_admin(request):
        return RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": None})


@app.post("/admin/login")
def admin_login(request: Request, password: str = Form("")):
    if password == SECRET_KEY:
        request.session["is_admin"] = True
        return RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": "Contraseña incorrecta"})


@app.get("/admin/logout")
def admin_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


@app.get("/admin", response_class=HTMLResponse)
def admin_panel(request: Request):
    if not _is_admin(request):
        return RedirectResponse(url="/admin/login", status_code=status.HTTP_302_FOUND)

    health = {
        "classes": list(map(str, getattr(model, "classes_", []))),
        "model_path": FINAL_MODEL_PATH,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
    }

    saved_metrics: Optional[dict] = None
    for p in ["artifacts/aug_metrics.json", "artifacts/baseline_metrics.json"]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    saved_metrics = json.load(f)
                    break
            except Exception:
                pass

    try:
        df_train = assemble_training_frame(include_store=True)
        dist = df_train[LABEL_COL].astype(str).value_counts().sort_index().to_dict()
    except Exception:
        dist = {}

    return templates.TemplateResponse("admin_panel.html", {"request": request, "health": health, "saved_metrics": saved_metrics, "dist": dist, "message": None})


@app.post("/admin/retrain-line", response_class=HTMLResponse)
def admin_retrain_line(request: Request, texto: str = Form(""), label: str = Form("")):
    if not _is_admin(request):
        return RedirectResponse(url="/admin/login", status_code=status.HTTP_302_FOUND)

    data = [{TEXT_COL: texto.strip(), LABEL_COL: str(label).strip()}]
    df_new = pd.DataFrame(data)
    try:
        append_to_store(df_new)
        df_train = assemble_training_frame(include_store=True)
        global model
        try:
            model = refit_existing_pipeline(model, df_train)
        except ValueError as e:
            msg = str(e)
            if ("max_df corresponds to < documents than min_df" in msg) or ("empty vocabulary" in msg):
                model = train_model(df_train, min_df=1, max_df=1.0)
            else:
                raise e
        save_model(model)
        metrics = evaluate(model, df_train)
        dist = df_train[LABEL_COL].astype(str).value_counts().sort_index().to_dict()
        message = "Reentrenamiento completado con 1 ejemplo"
    except Exception as e:
        metrics = None
        dist = {}
        message = f"Error al reentrenar: {e}"

    health = {
        "classes": list(map(str, getattr(model, "classes_", []))),
        "model_path": FINAL_MODEL_PATH,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
    }
    return templates.TemplateResponse("admin_panel.html", {"request": request, "health": health, "saved_metrics": metrics, "dist": dist, "message": message})


@app.post("/admin/retrain-batch", response_class=HTMLResponse)
def admin_retrain_batch(request: Request, file: UploadFile = File(...)):
    if not _is_admin(request):
        return RedirectResponse(url="/admin/login", status_code=status.HTTP_302_FOUND)

    try:
        if file.filename.lower().endswith((".xlsx", ".xls")):
            df_new = pd.read_excel(file.file)
        else:
            df_new = pd.read_csv(file.file)
    except Exception as e:
        return templates.TemplateResponse("admin_panel.html", {"request": request, "message": f"No pude leer el archivo: {e}", "health": {}, "saved_metrics": None, "dist": {}})

    cols_lower = {c.lower(): c for c in df_new.columns}
    tcol = cols_lower.get("textos", cols_lower.get("texto", TEXT_COL))
    lcol = cols_lower.get("labels", cols_lower.get("label", LABEL_COL))
    if tcol not in df_new.columns or lcol not in df_new.columns:
        return templates.TemplateResponse("admin_panel.html", {"request": request, "message": "El archivo debe tener columnas 'textos' y 'labels'", "health": {}, "saved_metrics": None, "dist": {}})

    df_new = df_new[[tcol, lcol]].rename(columns={tcol: TEXT_COL, lcol: LABEL_COL})
    df_new[TEXT_COL] = df_new[TEXT_COL].astype(str).str.strip()
    df_new[LABEL_COL] = df_new[LABEL_COL].astype(str).str.strip()
    df_new = df_new.dropna()

    try:
        append_to_store(df_new)
        df_train = assemble_training_frame(include_store=True)
        global model
        try:
            model = refit_existing_pipeline(model, df_train)
        except ValueError as e:
            msg = str(e)
            if ("max_df corresponds to < documents than min_df" in msg) or ("empty vocabulary" in msg):
                model = train_model(df_train, min_df=1, max_df=1.0)
            else:
                raise e
        save_model(model)
        metrics = evaluate(model, df_train)
        dist = df_train[LABEL_COL].astype(str).value_counts().sort_index().to_dict()
        message = f"Reentrenamiento completado con {len(df_new)} ejemplos"
    except Exception as e:
        metrics = None
        dist = {}
        message = f"Error al reentrenar: {e}"

    health = {
        "classes": list(map(str, getattr(model, "classes_", []))),
        "model_path": FINAL_MODEL_PATH,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
    }
    return templates.TemplateResponse("admin_panel.html", {"request": request, "health": health, "saved_metrics": metrics, "dist": dist, "message": message})
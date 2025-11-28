import os
import glob
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shutil
import joblib  # <-- NUEVO: para guardar modelo localmente sin Model Registry

DATA_DIR = "/opt/airflow/data/processed"
STATE_FILE = "/opt/airflow/data/state.json"

RANDOM_STATE = 31102025


def train_model():
    """Entrena un XGBoostRegressor acumulando todos los chunks procesados."""

    # -----------------------------
    # VALIDAR STATE JSON
    # -----------------------------
    if not os.path.exists(STATE_FILE):
        raise FileNotFoundError("No existe state.json. Corre ingesta + limpieza primero.")

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    current_chunk = state["last_chunk"]

    # -----------------------------
    # CARGAR ARCHIVOS PROCESADOS
    # -----------------------------
    X_train_files = sorted(glob.glob(f"{DATA_DIR}/X_train_*.parquet"))
    y_train_files = sorted(glob.glob(f"{DATA_DIR}/y_train_*.parquet"))
    X_test_files  = sorted(glob.glob(f"{DATA_DIR}/X_test_*.parquet"))
    y_test_files  = sorted(glob.glob(f"{DATA_DIR}/y_test_*.parquet"))

    if not X_train_files:
        raise ValueError("No hay datos procesados para entrenar.")

    # Concatenar datos
    def concat(files_x, files_y):
        X = pd.concat([pd.read_parquet(f) for f in files_x], ignore_index=True)
        y = pd.concat([pd.read_parquet(f).squeeze() for f in files_y], ignore_index=True)
        return X, y

    X_train, y_train = concat(X_train_files, y_train_files)
    X_test,  y_test  = concat(X_test_files,  y_test_files)

    print(f"Entrenando con {len(X_train)} muestras acumuladas...")

    # -----------------------------
    # MODELO XGBoost
    # -----------------------------
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE
    )

    # -----------------------------
    # CONFIGURACIÓN MLFLOW
    # -----------------------------
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("house_price_xgb")

    with mlflow.start_run(run_name=f"chunk_{current_chunk:03d}") as run:

        # =============================
        # ENTRENAR Y PREDECIR
        # =============================
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # =============================
        # MÉTRICAS
        # =============================
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape
        })

        mlflow.log_params({
            "current_chunk": current_chunk,
            "n_train": len(X_train),
            "n_test": len(X_test)
        })

        # ==================================================
        #   ***   GUARDAR MODELO COMO ARTEFACTO SIMPLE   ***
        # ==================================================
        #   - Sin Model Registry
        #   - Compatible 100% con MinIO
        #   - Sin endpoint /logged-models
        # ==================================================

        model_dir = f"/opt/airflow/data/models/chunk_{current_chunk:03d}"
        model_path = os.path.join(model_dir, "model.pkl")

        # Limpia carpeta si existe
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        # Guardar modelo
        joblib.dump(model, model_path)

        # Subir a MLflow como artefacto
        mlflow.log_artifacts(model_dir, artifact_path="model")

        print("Modelo guardado como artefacto simple (sin Model Registry).")

        print(f"Entrenamiento completado para chunk {current_chunk:03d}. Métricas:")
        print(f" - RMSE: {rmse:.4f}")
        print(f" - MAE:  {mae:.4f}")
        print(f" - R2:   {r2:.4f}")
        print(f" - MAPE: {mape:.4f}")

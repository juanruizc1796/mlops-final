import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from process_clean import clean_and_prepare

DATA_DIR = "/opt/airflow/data"
RAW_DIR = f"{DATA_DIR}/raw"
PROC_DIR = f"{DATA_DIR}/processed"
STATE_FILE = f"{DATA_DIR}/state.json"

RANDOM_STATE = 31102025


def clean_and_split():
    """Toma el chunk más reciente, lo limpia y lo divide en train/val/test."""
    os.makedirs(PROC_DIR, exist_ok=True)

    if not os.path.exists(STATE_FILE):
        raise FileNotFoundError("No existe state.json. Ejecuta primero la ingesta.")

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    chunk_id = state["last_chunk"]
    raw_file = os.path.join(RAW_DIR, f"chunk_{chunk_id:03d}.csv")

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"No existe {raw_file}. Ejecuta primero la ingesta.")

    print(f"Procesando {raw_file}...")

    df_raw = pd.read_csv(raw_file)
    X, y = clean_and_prepare(df_raw)

    # split: 70 train / 15 val / 15 test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.1765,
        random_state=RANDOM_STATE
    )

    # guardar todo por chunk
    def save(df, name):
        df.to_parquet(os.path.join(PROC_DIR, f"{name}_{chunk_id:03d}.parquet"))

    save(X_train, "X_train")
    save(X_val,   "X_val")
    save(X_test,  "X_test")
    y_train.to_frame("target").to_parquet(os.path.join(PROC_DIR, f"y_train_{chunk_id:03d}.parquet"))
    y_val.to_frame("target").to_parquet(os.path.join(PROC_DIR, f"y_val_{chunk_id:03d}.parquet"))
    y_test.to_frame("target").to_parquet(os.path.join(PROC_DIR, f"y_test_{chunk_id:03d}.parquet"))

    print(f"Limpieza + split completados para chunk {chunk_id}")

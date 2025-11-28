import os
import json
import pandas as pd
import requests
from datetime import datetime

API_URL = "http://10.43.100.103:8000/data?group_number=1&day=Tuesday"

DATA_DIR = "/opt/airflow/data"
RAW_DIR = f"{DATA_DIR}/raw"
STATE_FILE = f"{DATA_DIR}/state.json"


def fetch_chunk():
    """Descarga un chunk desde la API, lo guarda como CSV y actualiza el estado."""
    os.makedirs(RAW_DIR, exist_ok=True)

    # Si no existe, creamos state.json
    if not os.path.exists(STATE_FILE):
        state = {"last_chunk": 0}
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    else:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)

    chunk_id = state["last_chunk"] + 1

    print(f"Descargando chunk {chunk_id}...")

    try:
        resp = requests.get(API_URL, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error al obtener datos: {e}")
        raise ValueError("La API no respondió correctamente. Pipeline detenido.")

    raw = resp.json()
    df = pd.DataFrame(raw)

    if "data" in df.columns:
        details = pd.json_normalize(df["data"])
        df = pd.concat([df.drop(columns=["data"]), details], axis=1)

    file_path = os.path.join(RAW_DIR, f"chunk_{chunk_id:03d}.csv")
    df.to_csv(file_path, index=False)

    print(f"Guardado en {file_path}")

    # Actualizamos estado
    state["last_chunk"] = chunk_id
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

    print(f"Estado actualizado: chunk actual = {chunk_id}")

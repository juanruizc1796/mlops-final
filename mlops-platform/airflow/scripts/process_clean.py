import pandas as pd
from typing import Tuple

# ===============================
# Configuración de columnas
# ===============================

TARGET_COLUMN = "price"

NUMERIC_FEATURES = [
    "bed",
    "bath",
    "acre_lot",
    "house_size",
]


ALL_FEATURES = NUMERIC_FEATURES 


# ===============================
# Funciones de limpieza
# ===============================

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Verificar columnas requeridas
    required_columns = ALL_FEATURES + [TARGET_COLUMN]
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Convertir numéricos
    for col in NUMERIC_FEATURES + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminar filas sin target
    df = df.dropna(subset=[TARGET_COLUMN])

    # Rellenar numéricos
    for col in NUMERIC_FEATURES:
        if df[col].isna().all():
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].median())
 
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def clean_and_prepare(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Pipeline completa de limpieza → X, y"""
    df_clean = clean_raw_data(df_raw)
    return prepare_features(df_clean)

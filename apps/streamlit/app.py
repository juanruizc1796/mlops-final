import os
import requests
import streamlit as st

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi-service.fastapi.svc.cluster.local")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("House Price Predictor")
st.write("Modelo de estimación del precio de vivienda.")

st.sidebar.header("Parámetros de la vivienda")

bed = st.sidebar.number_input("Número de habitaciones (bed)", min_value=0, max_value=20, value=3)
bath = st.sidebar.number_input("Número de baños (bath)", min_value=0, max_value=20, value=2)
acre_lot = st.sidebar.number_input("Tamaño del lote (acre_lot)", min_value=0.0, max_value=100.0, value=0.5)
house_size = st.sidebar.number_input("Tamaño de la casa (house_size)", min_value=0, max_value=20000, value=1800)

if st.button("Predecir precio"):
    payload = {
        "bed": bed,
        "bath": bath,
        "acre_lot": acre_lot,
        "house_size": house_size,
    }
    try:
        resp = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pred = data.get("prediction")
        st.success(f"Precio estimado: **${pred:,.2f}**")
        st.caption(f"Modelo cargado desde: {data.get('model_path', 'desconocido')}")
    except Exception as e:
        st.error(f"Error llamando a la API de FastAPI: {e}")
        st.code(payload, language="json")

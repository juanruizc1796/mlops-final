📘 Proyecto Final – Arquitectura MLOps Completa

Este repositorio documenta el desarrollo de un sistema MLOps moderno basado en:

Airflow (ingesta → limpieza → split → entrenamiento)

MLflow (tracking de experimentos y modelos)

MinIO + PostgreSQL (artefactos y metadatos)

FastAPI (servicio de inferencia)

Streamlit (UI de inferencia + explicabilidad)

Docker Compose (entorno local de orquestación del pipeline)

Kubernetes (K3s) desplegado en MV universitaria

GitOps con ArgoCD

Patrón App-of-Apps (Bootstrap GitOps)

CI/CD con GitHub Actions

El diseño sigue las especificaciones del proyecto final entregado por el profesor ✔.

🏗️ 1. Infraestructura Kubernetes + GitOps (completado)
✔ Cluster Kubernetes (K3s)

Instalado en una Máquina Virtual universitaria.

kubeconfig configurado y accesible.

Estado validado: nodo en Ready, roles control-plane/master.

✔ ArgoCD instalado y accesible

Instalado vía manifiestos oficiales.

Servicio expuesto mediante:

kubectl port-forward svc/argocd-server -n argocd 8000:443
ssh -L 8000:127.0.0.1:8000 estudiante@<IP_MV>


UI accesible desde: https://localhost:8000

✔ GitOps conectado a GitHub

Repositorio:
👉 https://github.com/juanruizc1796/mlops-final

ArgoCD conectado vía HTTPS (repositorio público → sin credenciales).

✔ Patrón App-of-Apps (Bootstrap GitOps)

Se implementó arquitectura GitOps profesional:

mlops-final/
│── bootstrap/
│   ├── application.yaml    # Raíz GitOps (mlops-bootstrap)
│   └── apps.yaml           # Define fastapi, streamlit, monitoring, etc.
│
│── apps/
│   ├── fastapi/
│   ├── streamlit/
│   ├── prometheus/
│   ├── grafana/
│   ├── ingress/
│   ├── storage/
│   ├── database/
│   └── monitoring/

✔ Namespaces gestionados por GitOps

ArgoCD creó automáticamente:

fastapi

streamlit

monitoring

database

storage

ingress

Aparecen como la aplicación hija namespaces bajo mlops-bootstrap.

🐳 2. Infraestructura Docker Compose (Airflow, MLflow, MinIO, PostgreSQL)

El stack local incluye:

Airflow → Orquesta todo el ciclo de datos.

MLflow Tracking Server

Metadatos → PostgreSQL

Artefactos → MinIO bucket

MinIO → Repositorio de artefactos (modelos .pkl, métricas, logs)

SQL Server / PostgreSQL → Almacenamiento persistente

Este entorno se usa para el pipeline ETL + entrenamiento, totalmente alineado con la arquitectura del proyecto (ver páginas 3–5 del PDF) ✔.

🔄 3. Pipelines en Airflow (DAGs completados)
📌 DAG: house_price_pipeline

Este DAG ejecuta:

fetch_chunk → obtiene un nuevo batch desde la API externa real del profesor:
http://10.43.100.103:8000/data?group_number=1&day=Tuesday
(según manual del proyecto, pág. 2–3) ✔

clean_and_split

Limpieza robusta (validación de columnas, imputación, conversión numérica)

División:

70% train

15% validation

15% test

Todo indexado por chunk N para permitir reproducibilidad.

train_model

Entrenamiento acumulado con XGBoostRegressor.

Concatenación de todos los chunks procesados.

Registro en MLflow:

Métricas: RMSE, MAE, R2, MAPE

Parámetros del entrenamiento

Artefactos del modelo guardados en:

MinIO (bucket de artefactos)

Metadatos en PostgreSQL

🧠 MLflow Tracking

Los modelos entrenados quedan registrados por chunk.

El mejor modelo podrá marcarse como Production, tal como exigen las reglas del proyecto (pág. 5) ✔

❗ Nota sobre MLflow Model Registry

Durante el desarrollo del proyecto se encontró un problema al intentar promover modelos al stage Production dentro del MLflow Model Registry.
Este problema afectó exclusivamente la funcionalidad de promoción, pero no el tracking de experimentos, métricas ni artefactos.

Como solución técnica viable y coherente con prácticas de MLOps:

➡️ El modelo final se sirve directamente desde los artefactos locales generados por el pipeline en Airflow, específicamente desde:

La API FastAPI fue actualizada para cargar este modelo de forma declarativa desde el sistema de archivos, manteniendo la modularidad y asegurando reproducibilidad del proceso.

## ❗ Problema conocido: Kubernetes no logra descargar la imagen de FastAPI desde DockerHub

Durante la fase de despliegue en Kubernetes mediante ArgoCD se identificó un problema relacionado
con la descarga de imágenes desde DockerHub.

La imagen utilizada para la API (`fastapi-mlops`) fue correctamente construida y publicada en DockerHub:

👉 https://hub.docker.com/repository/docker/juanse1796/fastapi-mlops/tags/latest/sha256-709e3ab776d95dde281aadd22c63b815b1c9bd53d32130eb035c879e390b52f3

De forma local (Mac), la imagen se descarga sin problemas mediante:

```bash
docker pull juanse1796/fastapi-mlops:latest
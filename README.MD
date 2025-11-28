
# Proyecto Final – Arquitectura MLOps Completa
**Autores:** Yibby Gonzalez, Sebastian Ruiz & Adrian Tellez.

Se implementa una arquitectura de MLOps moderna que integra:

- **Airflow** → ingesta, limpieza, split y entrenamiento incremental  
- **MLflow** → tracking de experimentos y artefactos  
- **MinIO + PostgreSQL** → almacenamiento de modelos y metadatos  
- **FastAPI** → servicio de inferencia del modelo  
- **Streamlit** → interfaz visual de predicción  
- **Docker Compose** → orquestación local del pipeline  
- **Kubernetes (K3s)** → despliegue en MV javeriana  
- **GitOps (ArgoCD)** → despliegue declarativo y automatizado  
- **Patrón App-of-Apps** → arquitectura GitOps profesional  
- **CI/CD con GitHub Actions** → construcción y publicación de imágenes  



# Arquitectura General del Proyecto

```
                                 ┌────────────────────────┐
                                 │      GitHub Repo       │
                                 │   (Infra + Apps GitOps)│
                                 └─────────────┬──────────┘
                                               │
                                               ▼
                                   ┌─────────────────────┐
                                   │      ArgoCD         │
                                   │   App-of-Apps       │
                                   └─────────┬───────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Kubernetes Cluster (K3s)                           │
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│   │   FastAPI     │    │  Streamlit   │    │  Monitoring  │                 │
│   │  (Predict)    │<---> UI           │    │Prome + Grafana│                │
│   └──────────────┘    └──────────────┘    └──────────────┘                 │
│             ▲                                                             │
│             │  Carga modelo (model.pkl)                                   │
└─────────────┼──────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│                   Docker Compose                     │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌─────────────┐     │
│  │ Airflow  │→→ │ MLflow   │→→ │   MinIO      │     │
│  │ Pipeline │   │ Tracking │   │  Artefactos │     │
│  └──────────┘   └──────────┘   └─────────────┘     │
│         │                                           │
│         ▼                                           │
│     Genera modelo: model.pkl                        │
└─────────────────────────────────────────────────────┘
```

---

# 1. Kubernetes + GitOps (Completado)

## ✔ Cluster Kubernetes (K3s)
- Instalado en MV javeriana  
- `kubeconfig` configurado  
- Nodo en estado **Ready**

## ✔ ArgoCD instalado y accesible
- Instalación vía manifiestos oficiales  
- Exposición por port-forward + túnel SSH  
- UI: https://localhost:8000  

## ✔ GitOps conectado a GitHub

Repositorio principal:  
https://github.com/juanruizc1796/mlops-final

## ✔ Patrón App-of-Apps

```
mlops-final/
│── bootstrap/
│   ├── application.yaml
│   └── apps.yaml
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
```

Todas las aplicaciones se crean automáticamente desde ArgoCD.

## ✔ Namespaces creados por GitOps
- `fastapi`
- `streamlit`
- `monitoring`
- `database`
- `storage`
- `ingress`


# 2. Docker Compose (Airflow, MLflow, MinIO, PostgreSQL)

Este stack incluye:

| Servicio | Función |
|---------|---------|
| Airflow | Orquesta el pipeline |
| MLflow | Tracking |
| MinIO | Artefactos |
| PostgreSQL | Metadatos |

El cual fue desplegado en la maquina local dado que es la parte del proyecto que requiere de menos computo en comparación a la arquitectura en k8n.

# 3. Pipelines en Airflow

## DAG: `house_price_pipeline`

### ✔ `fetch_chunk`
Descarga datos reales desde:

```
http://10.43.100.103:8000/data?group_number=1&day=Tuesday
```

### ✔ `clean_and_split`
- Limpieza  
- Validación  
- Conversión numérica  
- Imputación  
- Splits 70/15/15  
- Persistencia por chunk

### ✔ `train_model`
Entrena acumulando todos los chunks.

Registra en MLflow:

- RMSE  
- MAE  
- R²  
- MAPE  
- Parámetros  
- Artefactos → MinIO

Modelo guardado en:

```
/opt/airflow/data/models/chunk/model.pkl
```


## MLflow Tracking

MLflow almacena:

- métricas  
- parámetros  
- artefactos  
- versiones del modelo  

## Problema con MLflow Model Registry
No se pudo promover modelos a Production debido a un error del Registry.  

**Solución adoptada:**  
FastAPI carga el modelo directamente desde los artefactos locales generados por Airflow.

# Problema conocido: Kubernetes NO puede hacer `docker pull`

La imagen FastAPI está en DockerHub:  
👉 https://hub.docker.com/repository/docker/juanse1796/fastapi-mlops

Localmente funciona:

```bash
docker pull juanse1796/fastapi-mlops:latest
```

Pero en la MV:

- `ErrImagePull`  
- `ImagePullBackOff`

Las posibles Causas son: Bloqueo institucional,DNS / firewall ,Restricción de red , Lo cual afecta parcialmente el desarrollo del despliegue del modelo.


# 4. FastAPI – Servicio de inferencia

✔ Carga modelo desde `/models/model.pkl`  
✔ Endpoint `/predict`  
✔ CI/CD via GitHub Actions  
✔ Diseño modular, desacoplado

Flujo:

```
Airflow → modelo.pkl → FastAPI → Streamlit
```


# 5. Streamlit – Interfaz de Usuario

✔ Permite ingresar variables  
✔ Llama a FastAPI dentro del cluster  
✔ Muestra la predicción  
✔ Listo para GitOps  


# 6. Prometheus & Grafana

✔ Prometheus desplegado  
✔ Grafana desplegada  
✔ Datasource → Prometheus  
✔ Base para monitoreo del sistema MLOps  


# 7. CI/CD – GitHub Actions

Workflow:

```
.github/workflows/build-fastapi.yml
```

Acciones:

- Construcción automática de imagen  
- Publicación en DockerHub  
- Validación de YAML  
- Versionado automático  
- Integrado a GitOps  

# Conclusiones

El desarrollo realizado durante este proyecto permitió construir una arquitectura MLOps completa, moderna y alineada con las mejores prácticas de la industria. Se logró integrar exitosamente cada uno de los componentes esenciales del ciclo de vida de un modelo de Machine Learning:

✔ Entrenamiento incremental y orquestado con Airflow

✔ Tracking robusto de experimentos con MLflow

✔ Artefactos versionados y almacenados en MinIO

✔ Servicio de inferencia mediante FastAPI

✔ UI de consumo y validación con Streamlit

✔ Observabilidad con Prometheus y Grafana

✔ Despliegue continuo y declarativo con GitOps (ArgoCD)

✔ CI/CD configurado con GitHub Actions

✔ Operación en Kubernetes (K3s) dentro de una MV universitaria

Este ecosistema reproduce de forma fiel la arquitectura que un equipo de MLOps profesional implementaría en un entorno, demostrando dominio del pipeline completo: desde la ingesta y transformación de datos, pasando por el entrenamiento y versionamiento de modelos, hasta su despliegue y monitoreo en Kubernetes.

Sin embargo, durante el desarrollo se presentaron limitaciones técnicas relevantes, principalmente:

Restricciones de recursos en la máquina local y en la MV, lo que afectó la ejecución fluida de ciertos servicios de Airflow, MLflow y Kubernetes.

Problema persistente de ErrImagePull en la MV, que impidió descargar imágenes desde DockerHub a pesar de que la imagen FastAPI estaba correctamente construida, versionada y accesible desde cualquier otro entorno.

Limitación del MLflow Model Registry, que impidió promover modelos a etapas superiores aunque el tracking y registro de artefactos funcionó de forma correcta.

A pesar de estas restricciones externas, el proyecto cumple en con la arquitectura y los requerimientos funcionales establecidos. 

El flujo MLOps completo(desde el procesamiento incremental de datos hasta su exposición vía API y UI)queda mapeado, con la salvedad de que los dos problemas mencionados (pull de imágenes en la MV y Model Registry) impiden cerrar el ciclo de producción directamente en el entorno.

No obstante, es importante resaltar que la arquitectura está implementada, y que al resolver estas limitaciones específicas del entorno, el pipeline quedaría funcionando de manera integral, automatizada y continua.


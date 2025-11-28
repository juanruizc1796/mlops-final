from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta

from fetch_chunk import fetch_chunk
from clean_and_split import clean_and_split
from train_model import train_model

default_args = {
    "owner": "juan_ruiz",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="house_price_pipeline",
    description="Pipeline completa: fetch → clean → split → train",
    default_args=default_args,
    start_date=datetime(2025, 11, 1),
    schedule_interval="*/3 * * * *",
    catchup=False,
    tags=["mlops", "pipeline"],
):

    with TaskGroup("data_pipeline") as group:

        t1 = PythonOperator(
            task_id="fetch_chunk",
            python_callable=fetch_chunk
        )

        t2 = PythonOperator(
            task_id="clean_and_split",
            python_callable=clean_and_split
        )

        t1 >> t2

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    group >> train

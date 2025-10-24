# Import necessary libraries
from airflow import DAG
# from airflow.operators.python import PythonOperator
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow
<<<<<<< HEAD
from airflow import configuration as conf

# Enable pickling for XCom
conf.set("core", "enable_xcom_pickling", "True")
=======

# NOTE:
# In Airflow 3.x, enabling XCom pickling should be done via environment variable:
# export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
# The old airflow.configuration API is deprecated.
>>>>>>> ec2eb14780820681766b73e6c4136b3f4fda1d89

# Default DAG arguments
default_args = {
<<<<<<< HEAD
    "owner": "your_name",
    "start_date": datetime(2025, 1, 15),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    "Airflow_Lab1_KMedoids",
    default_args=default_args,
    description="Airflow Lab DAG using K-Medoids clustering",
    schedule_interval=None,
=======
    'owner': 'your_name',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance named 'Airflow_Lab1' with the defined default arguments
with DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='Dag example for Lab 1 of Airflow series',
>>>>>>> ec2eb14780820681766b73e6c4136b3f4fda1d89
    catchup=False,
) as dag:

<<<<<<< HEAD
# Define tasks
load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id="data_preprocessing_task",
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

build_save_model_task = PythonOperator(
    task_id="build_kmedoids_model_task",
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "kmedoids_model.pkl"],
    dag=dag,
)

load_model_task = PythonOperator(
    task_id="load_model_task",
    python_callable=load_model_elbow,
    op_args=["kmedoids_model.pkl", build_save_model_task.output],
    dag=dag,
)

# Set dependencies
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task
=======
    # Task to load data, calls the 'load_data' Python function
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # Task to perform data preprocessing, depends on 'load_data_task'
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # Task to build and save a model, depends on 'data_preprocessing_task'
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
    )

    # Task to load a model using the 'load_model_elbow' function, depends on 'build_save_model_task'
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=["model.sav", build_save_model_task.output],
    )

    # Set task dependencies
    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task
>>>>>>> ec2eb14780820681766b73e6c4136b3f4fda1d89

# CLI compatibility
if __name__ == "__main__":
    dag.test()

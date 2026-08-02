FROM apache/airflow:2.10.0-python3.10
USER airflow
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

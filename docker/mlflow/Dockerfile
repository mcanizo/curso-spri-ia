#FROM python:3.7-slim-buster
FROM python:3.8-slim-buster

RUN pip install --upgrade pip

# Install python packages
RUN pip install boto3 \
 && pip install psycopg2-binary \
 && pip install numpy \
 && pip install pandas \
 && pip install scikit-learn \
 && pip install mlflow==1.23.1 \
 && pip install h5py

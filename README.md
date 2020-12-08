[![MlFLOW](img/mlflow-logo.png)](https://mlflow.org)

# SPRI AI for Industry 4.0. 2020 course
Code for SPRI AI for Industry 4.0. 2020 course.

This code has been updated to use latest versions of all the components on november 2020.

## Build MLflow Tracking Server via docker-compose
The ``docker-compose.yml`` file will build a MLflow Tracking Server with [PostgeSQL](https://www.postgresql.org/) as the metadata store and [MinIO](https://min.io/) as the artifact repository. Note that MinIO is a standalone version of [Amazon S3](https://aws.amazon.com/s3/?nc1=h_ls). Moreover, a [NGINX](https://www.nginx.com/) server is used as a reversed proxy to secure the communications.

Follow these steps to build the MLflow Tracking Server:

1. Install docker (docker & docker-compose commands must be installed).
2. git clone
3. docker-compose up
4. Open MLfLow UI at http://your-docker-machine-ip:80
5. Open MinIO at at http://your-docker-machine-ip:9000
6. Within MinIO, create a new bucket named ``mlflow-bucket``
5. Enjoy!


![MLflow Tracking SErver](img/MLflow_architecture.png)


## Track your models with MLflow
To track your ML models you must have MLflow instaled within your conda environment.

Run this to create a new conda env with the minimum libraries:
```
$ conda create -n mlflowEnv python=3.7
$ conda activate mlflowEnv
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

Set enviromental variables to define the credentials to be able to acces the MinIO repository to store your models:

```
$ sudo nano /etc/environment
```

set the following varibales within the file (the same as in the ``docker-compose.yml``):
- ``AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE``
- ``AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY``
- ``MLFLOW_S3_ENDPOINT_URL=http://localhost:9000``
- ``MLFLOW_TRACKING_URI=http://localhost:80``

**Note**: this credentials are only for the tuorial. You may change them in a production scenario.

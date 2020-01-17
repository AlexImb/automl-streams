FROM jupyter/scipy-notebook:latest

USER root

RUN apt-get update && apt-get install -y swig curl

USER $NB_USER

RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install --default-timeout=100

# Install the reqirements from the build context
COPY requirements.txt .
RUN pip install -r requirements.txt

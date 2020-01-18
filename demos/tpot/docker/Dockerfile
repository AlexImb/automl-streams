FROM jupyter/scipy-notebook:latest

USER root

RUN apt-get update && apt-get install -y swig curl

USER $NB_USER

# Install the reqirements from the build context
COPY requirements.txt .
RUN pip install -r requirements.txt

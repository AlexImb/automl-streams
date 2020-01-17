# AutoML Streams Dockerized Demos

A collection of AutoML Streams demos than can be executed in Docker and can use Kafka as streaming data source.

## Running instructions

### Requirements

Required: `Docker`

Strongly recommended: `Docker Compose, Make`

Useful: `kafkacat`

### Starting the containers

All containers at once:
```bash
make up
```

Individual containers:
```bash
docker-compose up auto-sklearn zookeeper broker
```

### Publishing a dataset to Kafka

OpenML dataset: 

To be added. For now see: https://github.com/AlexImb/automl-experiments

For any other dataset: 

```bash
cat ./_datasets/covtype.csv | kafkacat -P -b localhost -t covtype   
```

### Running an experiment

```bash
docker-compose exec h2o python training/h2o-pretrained.py
```


### Opening Jupyter/JupyterLab

Find the right port for the experiment/service in the `docker-compose.yml`

Navigate to: `localhost:<port>`, for example: `localhost:8887`

Get the Jupyter token by running 

```bash
docker-compose logs <service_name>
```

For example: 

```bash
docker-compose logs auto-sklearn
```

Copy the token and use it to login in Jupyter.

### Stopping the containers

All containers at once:
```bash
make down
```
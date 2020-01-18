# Start all containers: demos + Kafka Cluster
up:
	docker-compose up

# Stop and remove all containers
down:
	docker-compose down

# Publish the files from the _datasets folder to Kafka using kafkacat
publish-datasets:
	cd ./_datasets && ./_publish_to_kafka.sh

# Automl-streams: incl scikit-learn and scikit-multiflow
automl-streams: 
	docker-compose exec automl-streams python $(filepath)
automl-streams-batch: 
	docker-compose exec automl-streams python batch.py
automl-streams-online: 
	docker-compose exec automl-streams python online.py
automl-streams-online-drift: 
	docker-compose exec automl-streams python online_drift.py
automl-streams-meta: 
	docker-compose exec automl-streams python meta.py

# Auto-sklearn
auto-sklearn: 
	docker-compose exec auto-sklearn python $(filepath)
auto-sklearn-batch: 
	docker-compose exec auto-sklearn python batch.py

# H2O.ai
h2o:
	docker-compose exec h2o python $(filepath)
h2o-batch: 
	docker-compose exec h2o python batch.py

# TPOT
tpot:
	docker-compose exec tpot python $(filepath)
tpot-batch: 
	docker-compose exec tpot python batch.py

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
auto-streams: 
	docker-compose exec automl-streams python $(file)
auto-streams-batch: 
	docker-compose exec automl-streams python batch.py

# Auto-sklearn
auto-sklearn: 
	docker-compose exec auto-sklearn python $(file)
auto-sklearn-batch: 
	docker-compose exec auto-sklearn python batch.py

# H2O.ai
h2o:
	docker-compose exec h2o python $(file)
h2o-batch: 
	docker-compose exec h2o python batch.py

# TPOT
tpot:
	docker-compose exec tpot python $(file)
tpot-batch: 
	docker-compose exec tpot python batch.py

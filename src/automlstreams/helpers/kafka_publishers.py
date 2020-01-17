import time
import openml
from kafka import KafkaProducer


def publish_dataframe(data, topic, server='localhost:9092'):
    print(f'Publishing Pandas DataFrame to Kafka topic: {topic}')
    start_time = time.time()
    producer = KafkaProducer(bootstrap_servers=server)
    for i in data.index:
        producer.send(topic, data.iloc[[i]].to_csv(header=False).encode('utf-8'))
        producer.flush()
    producer.close()
    elapsed_time = time.time() - start_time
    print('Published in: ', elapsed_time)


def publish_openml_dataset(dataset_id, topic, server='localhost:9092'):
    print(f'Feching OpenML dataset ID: {dataset_id}')
    dataset = openml.datasets.get_dataset(dataset_id)
    data, _, _ = dataset._load_data()

    print(f'Publishing OpenML dataset to Kafka topic: {topic}')
    start_time = time.time()
    producer = KafkaProducer(bootstrap_servers=server)
    for i in data.index:
        producer.send(topic, data.iloc[[i]].to_csv(header=False).encode('utf-8'))
        producer.flush()
    producer.close()
    elapsed_time = time.time() - start_time
    print('Published in: ', elapsed_time)

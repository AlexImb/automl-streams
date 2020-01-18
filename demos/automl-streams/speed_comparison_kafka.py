from skmultiflow.evaluation.evaluate_stream_gen_speed import EvaluateStreamGenerationSpeed
from automlstreams.streams import KafkaStream
from skmultiflow.data import FileStream

DEFAULT_INPUT_TOPIC = 'sea_gen'
DEFAULT_BROKER = 'broker:29092'


def run(topic=DEFAULT_INPUT_TOPIC, broker=DEFAULT_BROKER):
    print(f'Running demo for topic={topic} and broker={broker}')
    evaluator = EvaluateStreamGenerationSpeed(10000, float("inf"), None, 1)

    kafka_stream = KafkaStream(topic, bootstrap_servers=broker)
    kafka_stream.prepare_for_use()
    evaluator.evaluate(kafka_stream)

    file_stream = FileStream(f'/_datasets/{topic}.csv')
    file_stream.prepare_for_use()
    evaluator.evaluate(file_stream)


if __name__ == "__main__":
    topics = [
        'hyperplane_gen', 'led_gen', 'rbf_gen', 'sea_gen',
        'covtype', 'elec', 'pokerhand', 'weather'
    ]
    for topic in topics:
        run(topic)

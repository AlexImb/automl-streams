from skmultiflow.evaluation.evaluate_stream_gen_speed import EvaluateStreamGenerationSpeed
from automlstreams.streams import KafkaStream

DEFAULT_INPUT_TOPIC = 'elec_4'
DEFAULT_BROKER = 'localhost:9092'


def run(topic=DEFAULT_INPUT_TOPIC, broker=DEFAULT_BROKER):
    print(f'Running demo for topic={topic} and broker={broker}')
    stream = KafkaStream(topic, bootstrap_servers=broker)
    stream.prepare_for_use()
    evaluator = EvaluateStreamGenerationSpeed(10000, float("inf"), None, 10)
    stream = evaluator.evaluate(stream)


if __name__ == "__main__":
    run()

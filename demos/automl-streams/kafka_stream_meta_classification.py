from automlstreams.streams import KafkaStream
from skmultiflow.evaluation import EvaluatePrequential
from automlstreams.meta import MetaClassifier

DEFAULT_INPUT_TOPIC = 'sea_big'
DEFAULT_BROKER = 'localhost:9092'


def run(topic=DEFAULT_INPUT_TOPIC, broker=DEFAULT_BROKER):
    print(f'Running demo for topic={topic} and broker={broker}')
    stream = KafkaStream(topic, bootstrap_servers=broker)
    stream.prepare_for_use()

    classifier = MetaClassifier()
    evaluator = EvaluatePrequential(show_plot=True,
                                    n_wait=200,
                                    batch_size=50,
                                    pretrain_size=500,
                                    max_samples=5000)

    evaluator.evaluate(stream=stream, model=classifier)


if __name__ == "__main__":
    run()

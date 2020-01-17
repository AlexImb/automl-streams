from automlstreams.streams import KafkaStream
from skmultiflow.evaluation import EvaluatePrequential
from automlstreams.meta import MetaClassifier, LastBestClassifier

# from skmultiflow.data import LEDGeneratorDrift
from skmultiflow.meta import LeverageBagging

DEFAULT_INPUT_TOPIC = 'sea_big'
DEFAULT_BROKER = 'localhost:9092'


def run(topic=DEFAULT_INPUT_TOPIC, broker=DEFAULT_BROKER):
    print(f'Running demo for topic={topic} and broker={broker}')
    stream = KafkaStream(topic, bootstrap_servers=broker)
    # stream = LEDGeneratorDrift(random_state=112, noise_percentage=0.28, has_noise=True, n_drift_features=4)
    stream.prepare_for_use()

    classifiers = [
        MetaClassifier(),
        MetaClassifier(active_learning=False),
        LastBestClassifier(),
        LastBestClassifier(active_learning=False),
        LeverageBagging()
    ]

    evaluator = EvaluatePrequential(show_plot=True,
                                    n_wait=200,
                                    batch_size=5,
                                    pretrain_size=1000,
                                    max_samples=10000)

    evaluator.evaluate(stream=stream, model=classifiers)


if __name__ == "__main__":
    run()

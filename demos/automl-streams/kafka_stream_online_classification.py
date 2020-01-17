from skmultiflow.bayes import NaiveBayes
from skmultiflow.lazy import SAMKNN
from skmultiflow.trees import HoeffdingTree, HAT, HATT
from skmultiflow.meta import AccuracyWeightedEnsemble, LeverageBagging, OzaBagging
from skmultiflow.evaluation import EvaluatePrequential
from sklearn.linear_model import SGDClassifier, Perceptron

from automlstreams.streams import KafkaStream

DEFAULT_INPUT_TOPIC = 'sea_big'
DEFAULT_BROKER = 'localhost:9092'
NO_TARGET_CLASSES = 2


def run(topic=DEFAULT_INPUT_TOPIC, broker=DEFAULT_BROKER):
    print(f'Running demo for topic={topic} and broker={broker}')
    stream = KafkaStream(topic, bootstrap_servers=broker)
    stream.prepare_for_use()

    # Wait until data from all the classes is available
    while stream.n_classes < NO_TARGET_CLASSES:
        stream.next_sample()

    clasifiers = [
        NaiveBayes(),
        SAMKNN(),
        HoeffdingTree(),
        HAT(),
        HATT(),
        AccuracyWeightedEnsemble(),
        LeverageBagging(),
        OzaBagging(),
        SGDClassifier(),
        SGDClassifier(average=True),
        Perceptron()
        ]

    evaluator = EvaluatePrequential(show_plot=True,
                                    n_wait=200,
                                    batch_size=1,
                                    pretrain_size=500,
                                    max_samples=5000)

    evaluator.evaluate(stream=stream, model=clasifiers)


if __name__ == "__main__":
    run()

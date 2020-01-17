from skmultiflow.evaluation import EvaluatePrequential
from automlstreams.streams import KafkaStream

from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTree
from skmultiflow.meta import LeverageBagging, OzaBagging
from sklearn.linear_model import SGDClassifier


DEFAULT_INPUT_TOPICS = ['elec', 'sea_big', 'weather']
DEFAULT_ESTIMATORS = [HoeffdingTree(), OzaBagging(), LeverageBagging(), SGDClassifier(), NaiveBayes()]
DEFAULT_BROKER = 'localhost:9092'
MAX_SAMPLES = 5000


def run(topics=DEFAULT_INPUT_TOPICS, broker=DEFAULT_BROKER, estimators=DEFAULT_ESTIMATORS, agg_by='dataset'):
    print(f'Running benchmark for topics={topics} and broker={broker}')

    if agg_by == 'dataset':
        for topic in DEFAULT_INPUT_TOPICS:
            stream = KafkaStream(topic, bootstrap_servers=broker)
            stream.prepare_for_use()
            evaluator = EvaluatePrequential(show_plot=True,
                                            n_wait=200,
                                            batch_size=1,
                                            pretrain_size=200,
                                            max_samples=MAX_SAMPLES,
                                            output_file=None,
                                            metrics=['accuracy', 'kappa'])

            evaluator.evaluate(stream=stream, model=estimators)
    elif agg_by == 'estimator':
        for estimator in DEFAULT_ESTIMATORS:
            stream = KafkaStream(topic, bootstrap_servers=broker)
            stream.prepare_for_use()
            evaluator = EvaluatePrequential(show_plot=True,
                                            n_wait=200,
                                            batch_size=1,
                                            pretrain_size=200,
                                            max_samples=MAX_SAMPLES,
                                            output_file=None,
                                            metrics=['accuracy', 'kappa'])

            evaluator.evaluate(stream=stream, model=estimators)


if __name__ == "__main__":
    run()

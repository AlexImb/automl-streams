from skmultiflow.data import FileStream
from automlstreams.streams import KafkaStream
from skmultiflow.evaluation import EvaluatePrequential

from skmultiflow.lazy import KNN
from skmultiflow.trees import HoeffdingTree
from skmultiflow.neural_networks import PerceptronMask
from sklearn.linear_model import SGDClassifier
from skmultiflow.trees import HAT
from skmultiflow.meta import OzaBaggingAdwin
from skmultiflow.meta import LeverageBagging

USE_KAFKA = False
DEFAULT_INPUT_TOPIC = 'sea_gen'
DEFAULT_BROKER = 'broker:29092'
MAX_SAMPLES = 20000


def run(model=SGDClassifier(), topic=DEFAULT_INPUT_TOPIC, broker=DEFAULT_BROKER):
    if USE_KAFKA:
        print(f'Running demo for topic={topic} and broker={broker}')
        stream = KafkaStream(topic, bootstrap_servers=broker)
    else:
        print(f'Running demo for file=/_datasets/{topic}.csv')
        stream = FileStream(f'/_datasets/{topic}.csv')

    stream.prepare_for_use()
    # Discard first 1k samples to discover classes
    stream.next_sample(1000)

    model_name = model.__class__.__name__
    evaluator = EvaluatePrequential(show_plot=False,
                                    n_wait=200,
                                    batch_size=1,
                                    pretrain_size=200,
                                    max_samples=MAX_SAMPLES,
                                    output_file=f'results/online.{model_name}.{topic}.csv')

    evaluator.evaluate(stream=stream, model=model)


if __name__ == "__main__":

    topics = [
        'agrawal_gen', 'stagger_gen', 'hyperplane_gen', 'led_gen', 'rbf_gen', 'sea_gen',
        'covtype', 'elec', 'pokerhand'
    ]

    models = [HoeffdingTree(), KNN(), PerceptronMask(), SGDClassifier(), HAT(), LeverageBagging(), OzaBaggingAdwin()]
    print([m.__class__.__name__ for m in models])
    for topic in topics:
        for model in models:
            print('\n', model.__class__.__name__, topic, ':\n')
            run(model, topic)

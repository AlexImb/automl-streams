from skmultiflow.data import FileStream
from automlstreams.streams import KafkaStream
from skmultiflow.evaluation import EvaluatePrequential

from automlstreams.meta import MetaClassifier, LastBestClassifier
from skmultiflow.trees import HoeffdingTree

USE_KAFKA = False
DEFAULT_INPUT_TOPIC = 'sea_gen'
DEFAULT_BROKER = 'broker:29092'
MAX_SAMPLES = 20000


def run(model=HoeffdingTree(), topic=DEFAULT_INPUT_TOPIC, broker=DEFAULT_BROKER):
    if USE_KAFKA:
        print(f'Running demo for topic={topic} and broker={broker}')
        stream = KafkaStream(topic, bootstrap_servers=broker)
    else:
        print(f'Running demo for file=/_datasets/{topic}.csv')
        # stream = FileStream(f'/_datasets/{topic}.csv')
        stream = FileStream(f'_datasets/{topic}.csv')

    stream.prepare_for_use()

    model_name = model.__class__.__name__
    # evaluator = EvaluatePrequential(show_plot=False,
    #                                 n_wait=200,
    #                                 batch_size=50,
    #                                 pretrain_size=200,
    #                                 max_samples=MAX_SAMPLES,
    #                                 output_file=f'results/meta.{model_name}.{topic}.csv')

    evaluator = EvaluatePrequential(show_plot=False,
                                    n_wait=200,
                                    batch_size=200,
                                    pretrain_size=500,
                                    max_samples=MAX_SAMPLES,
                                    output_file=f'automl-streams/results/meta.{model_name}.{topic}.csv')

    evaluator.evaluate(stream=stream, model=model)


if __name__ == "__main__":
    topics = [
        'agrawal_gen', 'stagger_gen', 'hyperplane_gen', 'led_gen', 'rbf_gen', 'sea_gen',
        'covtype', 'elec', 'pokerhand'
    ]

    models = [
        # MetaClassifier,
        LastBestClassifier
    ]
    print([m.__name__ for m in models])
    for topic in topics:
        for model in models:
            print('\n', model.__name__, topic, ':\n')
            run(model(), topic)

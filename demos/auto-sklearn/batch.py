from automlstreams.streams import KafkaStream
from automlstreams.evaluators import EvaluatePretrained
from autosklearn.classification import AutoSklearnClassifier
from skmultiflow.data import FileStream

USE_KAFKA = False
DEFAULT_INPUT_TOPIC = 'sea_gen'
DEFAULT_BROKER = 'broker:29092'
BATCH_SIZE = 8000
MAX_SAMPLES = 10000


def run(topic=DEFAULT_INPUT_TOPIC, broker=DEFAULT_BROKER):

    if USE_KAFKA:
        print(f'Running demo for topic={topic} and broker={broker}')
        stream = KafkaStream(topic, bootstrap_servers=broker)
    else:
        print(f'Running demo for file=/_datasets/{topic}.csv')
        stream = FileStream(f'/_datasets/{topic}.csv')
    stream.prepare_for_use()

    # Get a batch of BATCH_SIZE samples
    X, y = stream.next_sample(BATCH_SIZE)
    print('Sampled batch shape: ', X.shape)

    model = AutoSklearnClassifier(
        time_left_for_this_task=180,
        per_run_time_limit=30
    )

    model.fit(X, y)

    evaluator = EvaluatePretrained(show_plot=False,
                                   n_wait=200,
                                   batch_size=100,
                                   max_samples=MAX_SAMPLES,
                                   output_file=f'results/batch.AutoSklearnClassifier.{topic}.csv')

    evaluator.evaluate(stream=stream, model=model)


if __name__ == "__main__":
    topics = [
        'agrawal_gen', 'stagger_gen', 'hyperplane_gen', 'led_gen', 'rbf_gen', 'sea_gen',
        'covtype', 'elec', 'pokerhand'
    ]
    for topic in topics:
        run(topic)

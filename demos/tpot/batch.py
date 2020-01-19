from automlstreams.streams import KafkaStream
from automlstreams.evaluators import EvaluatePretrained
from tpot import TPOTClassifier
from skmultiflow.data import FileStream

USE_KAFKA = False
DEFAULT_INPUT_TOPIC = 'sea_gen'
DEFAULT_BROKER = 'broker:29092'
BATCH_SIZE = 5000
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

    model = TPOTClassifier(generations=5, population_size=20, max_time_mins=3, verbosity=2)

    model.fit(X, y)

    print('Fitted pipeline:')
    print(model.fitted_pipeline_)
    model.export(f'results/batch_pipeline_{topic}.py')

    evaluator = EvaluatePretrained(show_plot=False,
                                   n_wait=200,
                                   batch_size=1,
                                   max_samples=MAX_SAMPLES,
                                   output_file=f'results/batch.TPOTClassifier.{topic}.csv')

    evaluator.evaluate(stream=stream, model=model)


if __name__ == "__main__":
    topics = [
        'hyperplane_gen', 'led_gen', 'rbf_gen', 'sea_gen',
        'covtype', 'elec', 'pokerhand', 'weather'
    ]
    for topic in topics:
        run(topic)

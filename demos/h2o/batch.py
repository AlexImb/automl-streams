import h2o
from h2o.frame import H2OFrame
from h2o.automl import H2OAutoML
from automlstreams.streams import KafkaStream
from automlstreams.evaluators import EvaluatePretrained
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

    # Initialize a local H2O Cluster
    h2o.init()

    # Merge it into a single DF as required by H@O
    X = H2OFrame(X)
    y = H2OFrame(y)
    df = X.concat([y], axis=1)

    model = H2OAutoML(max_runtime_secs=180, seed=1)
    model.train(y=-1, training_frame=df)

    evaluator = EvaluatePretrained(show_plot=False,
                                   n_wait=200,
                                   batch_size=100,
                                   max_samples=MAX_SAMPLES,
                                   output_file=f'results/batch_{topic}.csv')

    evaluator.evaluate(stream=stream, model=model)


if __name__ == "__main__":
    topics = [
        'agrawal_gen', 'stagger_gen', 'hyperplane_gen', 'led_gen', 'rbf_gen', 'sea_gen',
        'covtype', 'elec', 'pokerhand'
    ]
    for topic in topics:
        run(topic)

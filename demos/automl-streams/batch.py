from skmultiflow.data import FileStream
from automlstreams.streams import KafkaStream
from automlstreams.evaluators import EvaluatePretrained
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

USE_KAFKA = False
DEFAULT_INPUT_TOPIC = 'sea_gen'
DEFAULT_BROKER = 'broker:29092'
BATCH_SIZE = 8000
MAX_SAMPLES = 10000


def run(model=RandomForestClassifier(), topic=DEFAULT_INPUT_TOPIC, broker=DEFAULT_BROKER):
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

    model.fit(X, y)

    model_name = model.__class__.__name__
    evaluator = EvaluatePretrained(show_plot=False,
                                   n_wait=200,
                                   batch_size=1,
                                   max_samples=MAX_SAMPLES,
                                   output_file=f'results/batch.{model_name}.{topic}.csv')

    evaluator.evaluate(stream=stream, model=model)


if __name__ == "__main__":
    topics = [
        'agrawal_gen', 'stagger_gen', 'hyperplane_gen', 'led_gen', 'rbf_gen', 'sea_gen',
        'covtype', 'elec', 'pokerhand'
    ]

    models = [RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(), LinearSVC()]
    print([m.__class__.__name__ for m in models])
    for topic in topics:
        for model in models:
            run(model, topic)

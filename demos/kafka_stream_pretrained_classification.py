from automlstreams.streams import KafkaStream
from automlstreams.evaluators import EvaluatePretrained
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_INPUT_TOPIC = 'sea_big'
DEFAULT_BROKER = 'localhost:9092'
BATCH_SIZE = 5000


def run(topic=DEFAULT_INPUT_TOPIC, broker=DEFAULT_BROKER):
    print(f'Running demo for topic={topic} and broker={broker}')
    stream = KafkaStream(topic, bootstrap_servers=broker)
    stream.prepare_for_use()

    # Get a batch of BATCH_SIZE samples
    X, y = stream.next_sample(BATCH_SIZE)
    print('Sampled batch shape: ', X.shape)

    # Fit a list of clasifiers on the batch
    classifiers = [
        GradientBoostingClassifier(),
        KNeighborsClassifier(),
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression())
    ]

    models = [clf.fit(X, y) for clf in classifiers]

    evaluator = EvaluatePretrained(show_plot=True,
                                   n_wait=200,
                                   batch_size=10,
                                   max_samples=10000)

    evaluator.evaluate(stream=stream, model=models)


if __name__ == "__main__":
    run()

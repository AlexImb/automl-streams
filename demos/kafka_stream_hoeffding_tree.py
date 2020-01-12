from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential
from automlstreams.streams import KafkaStream

DEFAULT_INPUT_TOPIC = 'elec_4'
DEFAULT_BROKER = 'localhost:9092'


def run(topic=DEFAULT_INPUT_TOPIC, broker=DEFAULT_BROKER):
    print(f'Running demo for topic={topic} and broker={broker}')
    stream = KafkaStream(topic, bootstrap_servers=broker)
    stream.prepare_for_use()
    ht = HoeffdingTree()
    evaluator = EvaluatePrequential(show_plot=True,
                                    pretrain_size=200,
                                    max_samples=3000)

    evaluator.evaluate(stream=stream, model=[ht], model_names=['HT'])


if __name__ == "__main__":
    run()

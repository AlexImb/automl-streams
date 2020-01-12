
# AutoML Streams

An AutoML framework for implementing automated machine learning on data streams 
architectures in production environments.

# Installation

From `pip`

```shell
pip install -U automl-streams
```

or `conda`:

```shell
conda install automl-streams
```

# Usage

```py
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential
from automlstreams.streams import KafkaStream

stream = KafkaStream(topic, bootstrap_servers=broker)
stream.prepare_for_use()
ht = HoeffdingTree()
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=200,
                                max_samples=3000)

evaluator.evaluate(stream=stream, model=[ht], model_names=['HT'])
```

More demonstrations available in the [demos](./demos) directory.

# Development

Create and activate a `virtualenv` for the project:

```shell
$ virtualenv .venv
$ source .venv/bin/activate
```

Install the `development` dependencies:

```shell
$ pip install -e . 
```

Install the app in "development" mode:
```shell
$ python setup.py develop  
```





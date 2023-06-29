
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


# Paper

https://arxiv.org/abs/2106.07317

```bibtex
@article{DBLP:journals/corr/abs-2106-07317,
  author       = {Alexandru{-}Ionut Imbrea},
  title        = {Automated Machine Learning Techniques for Data Streams},
  journal      = {CoRR},
  volume       = {abs/2106.07317},
  year         = {2021},
  url          = {https://arxiv.org/abs/2106.07317},
  eprinttype    = {arXiv},
  eprint       = {2106.07317},
  timestamp    = {Wed, 16 Jun 2021 10:42:19 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2106-07317.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


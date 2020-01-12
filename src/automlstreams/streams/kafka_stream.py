import pandas as pd
import numpy as np
from skmultiflow.data.base_stream import Stream
from kafka import KafkaConsumer, TopicPartition
from io import StringIO


class KafkaStream(Stream):
    """ Creates a stream from a Kafka topic

    Parameters
    ----------
    topic: str
        Kafka topic to consume data from.

    bootstrap_servers: str, list
        'host[:port]' string (or list of 'host[:port]'
        strings) that the consumer should contact to bootstrap initial
        cluster metadata. This does not have to be the full node list.
        It just needs to have at least one broker that will respond to a
        Metadata API Request. Default port is 9092. If no servers are
        specified, will default to localhost:9092.

    target_idx: int, optional (default=-1)
        The column index from which the targets start.

    n_targets: int, optional (default=1)
        The number of targets.

    cat_features: list, optional (default=None)
        A list of indices corresponding to the location of
        categorical features.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.kafka_stream import KafkaStream
    >>> # Setup the stream
    >>> stream = KafkaStream('sea_stream', bootstrap_servers='broker:29092')
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[0.080429, 8.397187, 7.074928]]), array([0]))
    >>> # Retrieving 10 samples
    >>> stream.next_sample(10)
    (array([[1.42074 , 7.504724, 6.764101],
        [0.960543, 5.168416, 8.298959],
        [3.367279, 6.797711, 4.857875],
        [9.265933, 8.548432, 2.460325],
        [7.295862, 2.373183, 3.427656],
        [9.289001, 3.280215, 3.154171],
        [0.279599, 7.340643, 3.729721],
        [4.387696, 1.97443 , 6.447183],
        [2.933823, 7.150514, 2.566901],
        [4.303049, 1.471813, 9.078151]]),
        array([0, 0, 1, 1, 1, 1, 0, 0, 1, 0]))
    >>> stream.n_remaining_samples()
    39989
    >>> stream.has_more_samples()
    True

"""
    _CLASSIFICATION = 'classification'
    _REGRESSION = 'regression'

    def __init__(self, topic, bootstrap_servers, target_idx=-1, n_targets=1, cat_features=None):
        super().__init__()

        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.n_targets = n_targets
        self.target_idx = target_idx
        self.cat_features = cat_features
        self.cat_features_idx = [] if self.cat_features is None else self.cat_features

        self.task_type = None
        # TODO: Fix classes for multilabel
        self.classes = np.array([]).astype(int)
        self.n_classes = 0

        # Automatically infer target_idx if not passed in multi-output problems
        if self.n_targets > 1 and self.target_idx == -1:
            self.target_idx = -self.n_targets

        self.__configure()

    def __configure(self):
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=None,
            auto_offset_reset='earliest',
            value_deserializer=lambda x: x.decode('utf-8')
        )

    @property
    def target_idx(self):
        """
        Get the number of the column where Y begins.

        Returns
        -------
        int:
            The number of the column where Y begins.
        """
        return self._target_idx

    @target_idx.setter
    def target_idx(self, target_idx):
        """
        Sets the number of the column where Y begins.

        Parameters
        ----------
        target_idx: int
        """

        self._target_idx = target_idx

    @property
    def n_targets(self):
        """
         Get the number of targets.

        Returns
        -------
        int:
            The number of targets.
        """
        return self._n_targets

    @n_targets.setter
    def n_targets(self, n_targets):
        """
        Sets the number of targets.

        Parameters
        ----------
        n_targets: int
        """

        self._n_targets = n_targets

    @property
    def cat_features_idx(self):
        """
        Get the list of the categorical features index.

        Returns
        -------
        list:
            List of categorical features index.

        """
        return self._cat_features_idx

    @cat_features_idx.setter
    def cat_features_idx(self, cat_features_idx):
        """
        Sets the list of the categorical features index.

        Parameters
        ----------
        cat_features_idx:
            List of categorical features index.
        """

        self._cat_features_idx = cat_features_idx

    def prepare_for_use(self):
        """ prepare_for_use

        Prepares the stream for use.

        Notes
        -----
        This functions should always be called after the stream initialization.

        """
        self._find_n_samples()
        # Call next sample once to initialize data info
        self.next_sample()
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None

    def _find_n_samples(self):
        """ Finds the number of samples by getting the latest offset (highwater) from Kafka
        """
        partitions = [TopicPartition(
            self.topic, p) for p in self.consumer.partitions_for_topic(self.topic)]
        offsets = self.consumer.end_offsets(partitions)
        # TODO: Check if first partition per topic is corrent for max offset
        offset = offsets[partitions[0]]
        self.n_samples = offset - 1

    def restart(self):
        """ restart

        Restarts the stream's sample feeding, while keeping all of its
        parameters.

        It basically server the purpose of reinitializing the stream to
        its initial state.

        """
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None

    def next_sample(self, batch_size=1):
        """ next_sample

        If there is enough instances to supply at least batch_size samples, those
        are returned. If there aren't a tuple of (None, None) is returned.

        Parameters
        ----------
        batch_size: int
            The number of instances to return.

        Returns
        -------
        tuple or tuple list
            Returns the next batch_size instances.
            For general purposes the return can be treated as a numpy.ndarray.

        """
        self.sample_idx += batch_size
        try:
            i = 0
            for message in self.consumer:
                sample = pd.read_csv(StringIO(message.value), header=None)
                if any(sample.dtypes == 'object'):
                    print(
                        f'''Streamed sample contains text or malformatted data.
                        Dropping sample: {self.sample_idx - i}''')

                i += 1
                if (i >= batch_size - 1):
                    break

            _, cols = sample.shape
            labels = sample.columns.values.tolist()

            if (self.target_idx + self.n_targets) == cols or (self.target_idx + self.n_targets) == 0:
                # Take everything to the right of target_idx
                self.current_sample_y = sample.iloc[:, self.target_idx:].values
                self.target_names = sample.iloc[:,
                                                self.target_idx:].columns.values.tolist()
            else:
                # Take only n_targets columns to the right of target_idx, use the rest as features
                self.current_sample_y = sample.iloc[:,
                                                    self.target_idx:self.target_idx + self.n_targets].values
                self.target_names = labels[self.target_idx:
                                           self.target_idx + self.n_targets]

            self.current_sample_x = sample.drop(
                self.target_names, axis=1).values
            self.feature_names = sample.drop(
                self.target_names, axis=1).columns.values.tolist()

            _, self.n_features = self.current_sample_x.shape

            if self.cat_features_idx:
                if max(self.cat_features_idx) < self.n_features:
                    self.n_cat_features = len(self.cat_features_idx)
                else:
                    raise IndexError('Categorical feature index in {} '
                                     'exceeds n_features {}'.format(self.cat_features_idx, self.n_features))
            self.n_num_features = self.n_features - self.n_cat_features

            if np.issubdtype(self.current_sample_y.dtype, np.integer):
                self.task_type = self._CLASSIFICATION
                sample_classes = np.unique(self.current_sample_y)
                self.classes = np.unique(np.concatenate(
                    (self.classes, sample_classes)))
                self.n_classes = len(np.unique(self.classes))
            else:
                self.task_type = self._REGRESSION

            if self.n_targets < 2:
                self.current_sample_y = self.current_sample_y.flatten()

        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None
        return self.current_sample_x, self.current_sample_y

    def _get_target_values(self):
        if self.task_type == self._CLASSIFICATION:
            if self.n_targets == 1:
                return np.unique(self.classes).tolist()
            else:
                return [self.classes.tolist() for i in range(self.n_targets)]
        elif self.task_type == self._REGRESSION:
            return [float] * self.n_targets

    def has_more_samples(self):
        """ Checks if stream has more samples.

        Returns
        -------
        Boolean
            True if stream has more samples.

        """
        return (self.n_samples - self.sample_idx) > 0

    def n_remaining_samples(self):
        """ Returns the estimated number of remaining samples.

        Returns
        -------
        int
            Remaining number of samples.

        """
        return self.n_samples - self.sample_idx

    def get_all_samples(self):
        """
        returns all the samples in the stream.

        Returns
        -------
        """
        raise NotImplementedError

    def get_data_info(self):
        if self.task_type == self._CLASSIFICATION:
            return "Kafka stream: {} - {} target(s)".format(self.topic, self.n_targets)
        elif self.task_type == self._REGRESSION:
            return "Kafka stream: {} - {} target(s)".format(self.topic, self.n_targets)

    def get_info(self):
        return 'KafkaStream(topic={}, target_idx={}, n_targets={}, cat_features={})'\
            .format("'" + self.topic + "'", self.target_idx, self. n_targets, self.cat_features)

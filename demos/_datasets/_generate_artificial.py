import pandas as pd
from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import SEAGenerator, STAGGERGenerator
from skmultiflow.data import LEDGeneratorDrift, HyperplaneGenerator, RandomRBFGeneratorDrift

DATASE_SIZE = 25000


def run(generator, filename='generated', n=10000):
    print(f'Generating dataset from generator ', filename)
    generator.prepare_for_use()
    X, y = generator.next_sample(n)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y, dtype='int32')
    df = pd.concat([X, y], axis=1)
    df.to_csv(f'_datasets/{filename}.csv', header=None, index=None)


if __name__ == "__main__":

    agrawal = ConceptDriftStream(alpha=45.0, position=DATASE_SIZE/2)
    run(agrawal, 'agrawal_gen', DATASE_SIZE)

    sea = ConceptDriftStream(
        SEAGenerator(classification_function=1, noise_percentage=0.13),
        SEAGenerator(classification_function=2, noise_percentage=0.13),
        position=DATASE_SIZE/2)
    run(sea, 'sea_gen', DATASE_SIZE)

    led = LEDGeneratorDrift(has_noise=True, noise_percentage=0.28, n_drift_features=4)
    run(led, 'led_gen', DATASE_SIZE)

    stagger = ConceptDriftStream(
        STAGGERGenerator(classification_function=1, balance_classes=False),
        STAGGERGenerator(classification_function=2, balance_classes=False),
        position=DATASE_SIZE/2)
    run(stagger, 'stagger_gen', DATASE_SIZE)

    hyperplane = HyperplaneGenerator(
        noise_percentage=0.28,
        n_features=10,
        mag_change=0.25,
        sigma_percentage=0.3,
        n_drift_features=5)
    run(hyperplane, 'hyperplane_gen', DATASE_SIZE)

    rbf = RandomRBFGeneratorDrift(change_speed=0.4)
    run(rbf, 'rbf_gen', DATASE_SIZE)

import pandas as pd
from skmultiflow.data import SEAGenerator, LEDGeneratorDrift
from skmultiflow.data import HyperplaneGenerator, RandomRBFGeneratorDrift

DATASE_SIZE = 25000


def run(generator, filename='generated', n=10000):
    print(f'Generating dataset from generator ', filename)
    generator.prepare_for_use()
    X, y = generator.next_sample(n)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    df = pd.concat([X, y], axis=1)
    df.to_csv(f'_datasets/{filename}.csv', header=None, index=None)


if __name__ == "__main__":
    sea = SEAGenerator(classification_function=2, noise_percentage=0.13)
    run(sea, 'sea_gen', DATASE_SIZE)

    led = LEDGeneratorDrift(has_noise=True, noise_percentage=0.13, n_drift_features=3)
    run(led, 'led_gen', DATASE_SIZE)

    # stagger = STAGGERGenerator(classification_function=2, balance_classes=False)
    # run(stagger, 'stagger_gen', DATASE_SIZE)

    hyperplane = HyperplaneGenerator(noise_percentage=0.28, n_features=10, n_drift_features=3)
    run(hyperplane, 'hyperplane_gen', DATASE_SIZE)

    rbf = RandomRBFGeneratorDrift()
    run(rbf, 'rbf_gen', DATASE_SIZE)

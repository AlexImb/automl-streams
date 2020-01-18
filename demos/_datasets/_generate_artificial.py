import pandas as pd
from skmultiflow.data import SEAGenerator, LEDGeneratorDrift
from skmultiflow.data import AGRAWALGenerator, HyperplaneGenerator, RandomRBFGeneratorDrift


def run(generator, filename='generated', n=10000):
    print(f'Generating dataset from generator')
    generator.prepare_for_use()
    X, y = generator.next_sample(n)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    df = pd.concat([X, y], axis=1)
    df.to_csv(f'_datasets/{filename}.csv', header=None, index=None)


if __name__ == "__main__":
    sea = SEAGenerator()
    run(sea, 'sea_gen', 10**5)
    led = LEDGeneratorDrift()
    run(led, 'led_gen', 10**5)
    agrawal = AGRAWALGenerator()
    run(agrawal, 'agrawal_gen', 10**5)
    hyperplane = HyperplaneGenerator()
    run(hyperplane, 'hyperplane_gen', 10**5)
    rbf = RandomRBFGeneratorDrift()
    run(rbf, 'rbf_gen', 10**5)

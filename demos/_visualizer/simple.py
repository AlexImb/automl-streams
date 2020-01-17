import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.style as style
style.use('seaborn-poster')


def plot(path):
    df = pd.read_csv(path, comment='#')
    print(df)
    print(df.columns)
    x = df[['id']]
    y = df.drop(columns=['id'])
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    demo = 'auto-sklearn'
    demo_type = 'batch'
    topic = 'sea_big'
    path = f'{demo}/results/{demo_type}_{topic}.csv'
    plot(path)

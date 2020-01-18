import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.style as style
style.use('seaborn-poster')

def plot_topics(demo, demo_type, topics):
    Y = pd.DataFrame()

    for topic in topics:
        path = get_path(demo, demo_type, topic)
        df = pd.read_csv(path, comment='#')
        x = df[['id']]
        y = df[['mean_acc_[M0]']]
        Y = pd.concat([Y, y], axis=1)

    plt.plot(x, Y)
    plt.legend(topics)
    plt.show()


def get_path(demo, demo_type, topic):
    return f'{demo}/results/{demo_type}_{topic}.csv'


if __name__ == "__main__":
    topics = ['elec', 'covtype', 'weather', 'sea_big', 'moving_squares']
    plot_topics('auto-sklearn', 'batch', topics)


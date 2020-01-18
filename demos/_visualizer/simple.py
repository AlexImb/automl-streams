import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.style as style
style.use('seaborn-poster')

SHOW_PLOT = False
SAVE_FIG = True


def plot_topics(demo, demo_type, topics):
    plt.close()
    Y = pd.DataFrame()

    for topic in topics:
        path = get_path(demo, demo_type, topic)
        df = pd.read_csv(path, comment='#')
        x = df[['id']]
        y = df[['mean_acc_[M0]']]
        Y = pd.concat([Y, y], axis=1)

    plt.plot(x, Y)
    plt.legend(topics)

    if SAVE_FIG:
        plt.savefig(f'{demo}/results/{demo_type}_all_topics.png')
    if SHOW_PLOT:
        plt.show()


def get_path(demo, demo_type, topic):
    return f'{demo}/results/{demo_type}_{topic}.csv'


if __name__ == "__main__":
    demos = ['auto-sklearn', 'automl-streams']
    demo_types = ['batch']
    topics = ['elec', 'covtype', 'weather', 'sea_big', 'moving_squares']

    for demo in demos:
        for demo_type in demo_types:
            plot_topics(demo, demo_type, topics)

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.style as style
style.use('seaborn-poster')

SHOW_PLOT = False
SAVE_FIG = True


def plot_topics(demo, demo_type, model, topics):
    plt.close()
    mean_acc = pd.DataFrame()
    mean_kappa = pd.DataFrame()
    current_acc = pd.DataFrame()
    current_kappa = pd.DataFrame()


    for topic in topics:
        print(topic)
        path = get_path(demo, demo_type, model, topic)
        df = pd.read_csv(path, comment='#')
        x = df[['id']]
        mean_acc = pd.concat([mean_acc, df[['mean_acc_[M0]']]], axis=1)
        # mean_kappa = pd.concat([mean_kappa, df[['mean_kappa_[M0]']]], axis=1)
        # current_acc = pd.concat([current_acc, df[['current_acc_[M0]']]], axis=1)
        # current_kappa = pd.concat([current_kappa, df[['current_kappa_[M0]']]], axis=1)

    f, ax = plt.subplots(1)
    ax.plot(x, mean_acc)
    # ax.plot(x, mean_kappa)
    # ax.plot(x, current_acc)
    # ax.plot(x, current_kappa)
    ax.set_ylim(ymin=0, ymax=1)
    plt.legend(topics)

    if SAVE_FIG:
        plt.savefig(f'{demo}/results/figures/{demo_type}_{model}_all_topics.png')
    if SHOW_PLOT:
        plt.show()


def get_path(demo, demo_type, model, topic):
    return f'{demo}/results/{demo_type}.{model}.{topic}.csv'


if __name__ == "__main__":
    demos = ['auto-sklearn', 'automl-streams', 'tpot']
    demos = ['tpot']

    demo_types = ['batch', 'online', 'online_drift', 'meta']
    demo_types = ['batch']

    all_topics = [
        'hyperplane_gen', 'led_gen', 'rbf_gen', 'sea_gen',
        'covtype', 'elec', 'pokerhand', 'weather'
    ]

    all_topics = [
        'covtype', 'elec'
    ]

    for demo in demos:
        for demo_type in demo_types:
            if demo == 'auto-sklearn':
                if demo_type == 'batch':
                    topics = all_topics
                    models = ['AutoSklearnClassifier']
            elif demo == 'automl-streams':
                if demo_type == 'batch':
                    topics = all_topics
                    models = ['GradientBoostingClassifier', 'KNeighborsClassifier', 'LogisticRegression', 'SGDClassifier']
                elif demo_type == 'online':
                    topics = ['elec']
                    models = ['HoeffdingTree', 'OzaBagging', 'LeverageBagging', 'SGDClassifier', 'NaiveBayes']
                elif demo_type == 'online_drift':
                    topics = ['elec']
                    models = ['HAT', 'OzaBaggingAdwin']
                elif demo_type == 'meta':
                    topics = ['elec', 'sea_gen']
                    models = ['MetaClassifier', 'LastBestClassifier']
            elif demo_type == 'tpot':
                if demo_type == 'batch':
                    topics = all_topics
                    models = ['TPOTClassifier']
            
            for model in models:
                plot_topics(demo, demo_type, model, topics)

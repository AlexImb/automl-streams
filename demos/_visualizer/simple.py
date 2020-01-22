import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.style as style
style.use('seaborn-poster')

SHOW_PLOT = False
SAVE_FIG = True


def plot_topics(demo, demo_type, model, topics):
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
        mean_kappa = pd.concat([mean_kappa, df[['mean_kappa_[M0]']]], axis=1)
        current_acc = pd.concat([current_acc, df[['current_acc_[M0]']]], axis=1)
        current_kappa = pd.concat([current_kappa, df[['current_kappa_[M0]']]], axis=1)

    plt.close()
    plt.title(f'{model} Model Predictive Accuracy', pad=26)
    plt.ylabel('Mean predictive accuracy')
    plt.xlabel('Number of samples')
    plt.plot(x, mean_acc)
    # ax.plot(x, mean_kappa)
    # ax.plot(x, current_acc)
    # ax.plot(x, current_kappa)
    plt.ylim(ymin=0.45, ymax=1.05)
    plt.legend(topics)
    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(f'{demo}/results/figures/{demo_type}_{model}_{len(topics)}_topics.png')
    if SHOW_PLOT:
        plt.show()


def plot_topics_grouped(demo, demo_type, models, topics):
    plt.close()
    f, subplots = plt.subplots(len(models))
    f.set_size_inches(10, 20)
    f.subplots_adjust(top=0.94, hspace=0.4)
    f.suptitle('Models Predictive Accuracy', fontsize=26)

    for index, subplot in enumerate(subplots):
        mean_acc = pd.DataFrame()
        mean_kappa = pd.DataFrame()
        current_acc = pd.DataFrame()
        current_kappa = pd.DataFrame()
        for topic in topics:
            print(topic)
            path = get_path(demo, demo_type, models[index], topic)
            df = pd.read_csv(path, comment='#')
            x = df[['id']]
            mean_acc = pd.concat([mean_acc, df[['mean_acc_[M0]']]], axis=1)
            mean_kappa = pd.concat([mean_kappa, df[['mean_kappa_[M0]']]], axis=1)
            current_acc = pd.concat([current_acc, df[['current_acc_[M0]']]], axis=1)
            current_kappa = pd.concat([current_kappa, df[['current_kappa_[M0]']]], axis=1)

        subplot.set_title(models[index])
        subplot.set_ylabel('Mean accuracy')
        if index == len(subplots) - 1:
            subplot.set_xlabel('Number of samples')
        subplot.plot(x, mean_acc)
        # subplot.plot(x, mean_kappa)
        # subplot.plot(x, current_acc)
        # subplot.plot(x, current_kappa)
        # subplot.set_ylim(ymax=1.05)

    plt.legend(topics, loc='center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=3)
    
    if SAVE_FIG:
        plt.savefig(f'{demo}/results/figures/{demo_type}_all_models_all_topics.png')
    if SHOW_PLOT:
        plt.show()

def get_path(demo, demo_type, model, topic):
    return f'{demo}/results/{demo_type}.{model}.{topic}.csv'


if __name__ == "__main__":
    demos = ['auto-sklearn', 'tpot', 'automl-streams']
    demo_types = ['batch']
    topics = [
        'agrawal_gen', 'stagger_gen', 'hyperplane_gen', 'led_gen', 'rbf_gen', 'sea_gen',
        'covtype', 'elec', 'pokerhand'
    ]

    topics = [
        'agrawal_gen', 'stagger_gen', 'hyperplane_gen', 'sea_gen',
        'covtype', 'elec', 'pokerhand'
    ]

    plot_grouped = False

    for demo in demos:
        if demo == 'automl-streams':
            demo_types.extend(['online', 'meta'])

        for demo_type in demo_types:
            if demo == 'auto-sklearn':
                if demo_type == 'batch':
                    models = ['AutoSklearnClassifier']
            elif demo == 'automl-streams':
                if demo_type == 'batch':
                    models = ['RandomForestClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'LinearSVC']
                elif demo_type == 'online':
                    models = ['HoeffdingTree', 'KNN', 'PerceptronMask', 'SGDClassifier', 'HAT', 'LeverageBagging', 'OzaBaggingAdwin']
                elif demo_type == 'meta':
                    models = ['MetaClassifier', 'LastBestClassifier']
            elif demo == 'tpot':
                if demo_type == 'batch':
                    models = ['TPOTClassifier']
                    # models = ['TPOTClassifier', 'AutoSklearnClassifier']

            if plot_grouped:
                print('Plotting grouped', demo, demo_type, models, topics)
                plot_topics_grouped(demo, demo_type, models, topics)
            else:
                for model in models:
                    print('Plotting topics', demo, demo_type, model, topics)
                    plot_topics(demo, demo_type, model, topics)

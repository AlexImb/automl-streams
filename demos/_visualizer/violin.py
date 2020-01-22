import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.style as style
style.use('seaborn-poster')

SHOW_PLOT = False
SAVE_FIG = True


def plot_violins(demo, demo_type, models, topics):
    plt.close()
    metric = 'mean_acc_[M0]'
    model_metrics = []
    for index, model in enumerate(models):
        topic_metrics = pd.Series()
        for topic in topics:
            path = get_path(demo, demo_type, models[index], topic)
            df = pd.read_csv(path, comment='#')
            current_metric_topic = df[[metric]]
            mean_topic = current_metric_topic.mean()
            topic_metrics = pd.concat([topic_metrics, mean_topic])
        model_metrics.append(topic_metrics.values)
    
    plt.violinplot(model_metrics, showmeans=True, showmedians=True)
    plt.ylim(0, 1)
    plt.gca().grid(which='both', axis='y', linestyle='dotted')
    plt.xticks(np.arange(1, len(models) + 1), models, rotation=10)
    # plt.title('AutoML Models Accuracy (Dataset-averaged)', pad=26)
    plt.title(f'{demo_type.title()} Models Accuracy (Dataset-averaged)', pad=26)
    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(f'{demo}/results/figures/{demo_type}_all_models_all_topics_violin.png')
    if SHOW_PLOT:
        plt.show()


def get_path(demo, demo_type, model, topic):
    return f'{demo}/results/{demo_type}.{model}.{topic}.csv'


if __name__ == "__main__":
    demos = ['auto-sklearn', 'automl-streams', 'tpot']
    demos = ['automl-streams']

    demo_types = ['batch', 'online', 'meta']
    demo_types = ['meta']

    topics = [
        'agrawal_gen', 'stagger_gen', 'hyperplane_gen', 'led_gen', 'rbf_gen', 'sea_gen',
        'covtype', 'elec', 'pokerhand'
    ]

    for demo in demos:
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

            print('Plotting violins', demo, demo_type, models, topics)
            plot_violins(demo, demo_type, models, topics)

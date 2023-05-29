import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(feature_importance, title):
    feature_importance_as_list = list(zip(feature_importance.index, feature_importance))
    features = list(zip(*feature_importance_as_list))[0]
    importance = list(zip(*feature_importance_as_list))[1]

    plt.rcParams["figure.figsize"] = (24, 17)
    plt.bar(features, importance, align='edge')
    plt.xticks(rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Feature Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


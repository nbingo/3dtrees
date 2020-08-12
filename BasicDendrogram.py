import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rcParams

from scipy.cluster.hierarchy import dendrogram
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering

from data.data_loader import read_data

rcParams.update({'figure.autolayout': True})


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


if __name__ == '__main__':
    data = read_data(['mouse'])


    def spearmanr_connectivity(x):
        # data_ct is assumed to be (n_variables, n_examples)
        rho, _ = spearmanr(x, axis=1)
        return 1 - rho


    agglomerate = AgglomerativeClustering(
        affinity=spearmanr_connectivity,
        linkage='complete',
        n_clusters=None,
        distance_threshold=0
    )
    agglomerate.fit(data.to_numpy())

    plt.title('Hierarchical clustering')
    plot_dendrogram(agglomerate, truncate_mode='level', labels=data.index)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('dendrogram.pdf')
    plt.show()

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


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


def read_preprocess_data():
    deg_chicken = pd.read_csv('TranscriptomeData/Nomi_chickenDEGs.csv', header=0)
    deg_chicken = deg_chicken.rename(columns={'Unnamed: 0': 'Name'})
    data_chicken = pd.read_csv('TranscriptomeData/Nomi_chickenallaverage.csv', header=0, index_col=0)

    deg_mouse = pd.read_csv('TranscriptomeData/Nomi_mouseDEGs.csv', header=0)
    deg_mouse = deg_mouse.rename(columns={'Unnamed: 0': 'Name'})
    data_mouse = pd.read_csv('TranscriptomeData/Nomi_mouseallaverage.csv', header=0, index_col=0)

    deg_human = pd.read_csv('TranscriptomeData/Nomi_humanDEGs.csv', header=0)
    deg_human = deg_human.rename(columns={'Unnamed: 0': 'Name'})
    data_human = pd.read_csv('TranscriptomeData/Nomi_humanallaverage.csv', header=0, index_col=0)

    # Find the common DEGs
    common_genes = np.intersect1d(deg_human['gene'], np.intersect1d(deg_mouse['gene'], deg_chicken['gene']))

    # Filter for common DEGs
    data_chicken = data_chicken.loc[data_chicken.index.isin(common_genes)]
    data_mouse = data_mouse.loc[data_mouse.index.isin(common_genes)]
    data_human = data_human.loc[data_human.index.isin(common_genes)]

    # Divide each row by mean, as in Tosches et al, rename columns,
    # and transpose so that column labels are genes and rows are cell types
    data_chicken = data_chicken.div(data_chicken.mean(axis=1).values, axis=0).add_prefix('C_').transpose()
    data_mouse = data_mouse.div(data_mouse.mean(axis=1).values, axis=0).add_prefix('M_').transpose()
    data_human = data_human.div(data_human.mean(axis=1).values, axis=0).add_prefix('H_').transpose()

    # Concatenate all the data
    # data_all = pd.concat([data_mouse, data_chicken, data_human])
    data_all = pd.concat([data_mouse, data_chicken])

    return data_all


if __name__ == '__main__':
    data = read_preprocess_data()

    model = AgglomerativeClustering(linkage='complete', n_clusters=None, distance_threshold=0)
    model.fit(data.to_numpy())

    plt.title('Hierarchical clustering')
    plot_dendrogram(model, truncate_mode='level')
    plt.show()
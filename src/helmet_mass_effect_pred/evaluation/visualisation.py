import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def make_tsne_and_pca(
    train_targets, test_targets, train_OLTV, target_col_names, wandb_logger
):
    # check there are at least two categories for each target
    assert np.unique(train_targets).shape[0] > 1
    assert np.unique(test_targets).shape[0] > 1
    assert np.unique(train_targets).shape[0] == np.unique(test_targets).shape[0]

    colors = "r", "g", "b", "c", "m", "y", "k", "orange", "purple", "pink"
    transparency = 0.3

    tsne = TSNE(n_components=2, random_state=0, perplexity=50)
    X_2d = tsne.fit_transform(train_OLTV)
    target_ids = range(np.unique(train_targets).shape[0])
    plt.figure(figsize=(6, 5))

    for i, c, label in zip(target_ids, colors, target_col_names):
        plt.scatter(
            X_2d[train_targets == i, 0],
            X_2d[train_targets == i, 1],
            c=c,
            label=label,
            alpha=transparency,
        )
    plt.legend()
    wandb_logger.log({"tsne_plot": plt})

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(train_OLTV)
    target_ids = range(np.unique(train_targets).shape[0])
    plt.figure(figsize=(6, 5))
    for i, c, label in zip(target_ids, colors, target_col_names):
        plt.scatter(
            X_2d[train_targets == i, 0],
            X_2d[train_targets == i, 1],
            c=c,
            label=label,
            alpha=transparency,
        )
    plt.legend()
    wandb_logger.log({"pca_plot": plt})

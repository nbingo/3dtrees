from scipy.stats import spearmanr


def spearmanr_connectivity(x, y):
    # data is assumed to be (n_variables, n_examples)
    rho, _ = spearmanr(x, y, axis=1)
    return 1 - rho

# 3DPhyloTrees
Welcome!

## Overview
The purpose of this Python package is to create 3D phylogenetic trees with two axes of variation given suitable 
data in the common [AnnData](https://anndata.readthedocs.io/en/latest/index.html) format. The differentiating factor
between this package and functions like ```scipy.cluster.hierarchy.dendrogram()``` and 
```sklearn.cluster.AgglomerativeClustering``` is that the dendrogram produced tracks the splitting/merging patterns
of groups of taxa and individual taxa. Specifically, as used in the BioRxiv paper [Cerebellar nuclei evolved by 
repeatedly duplicating a conserved cell type set](https://www.biorxiv.org/content/10.1101/2020.06.25.170118v1), the
phylogenetic tree created by this package tracks the merging of different subnuclei of the cerebellar nuclei while
also tracking the merging of individual cell types within those nuclei.

An example of such a dendrogram is
![Image](https://drive.google.com/uc?id=1hOiOlr3U5zTGu6pPH8c71E0portG_n1G)

A flattened version can be found in Fig. S22C and Fig. S23H of the linked paper above. 

This package is composed of three main parts:
* agglomerate
* data
* metrics

The [agglomerate](https://github.com/nbingo/3dtrees/tree/master/agglomerate) package exposes methods to perform the
agglomeration of a single phylogenetic tree given suitable data and hyperparameters, and a method to perform batch
agglomeration over a range of hyperparameters and select the best tree according to any of the following metrics:
* Balanced Minimum Evolution (preferred)
* Minimum Evolution
* Maximum Parsimony

The [data](https://github.com/nbingo/3dtrees/tree/master/data) package exposes a 
[data_loader](https://github.com/nbingo/3dtrees/blob/master/data/data_loader.py) that the user can define to 
import their data accordingly (possibly from multiple folders or online repositories) and into an AnnData object.
The [data_types](https://github.com/nbingo/3dtrees/blob/master/data/data_types.py) are used internally by the 
agglomeration algorithm.

Finally, the [metrics](https://github.com/nbingo/3dtrees/tree/master/metrics) currently only provides the 
Spearman correlation coefficient to measure the distance between two data points, however any distance metric in the 
same form as the example provided may be added and used in the agglomeration program.

## Installation
This package requires Python version 3.7 or greater, and the requirements provided in 
[Pipfile](https://github.com/nbingo/3dtrees/blob/master/Pipfile) and 
[Pipfile.lock](https://github.com/nbingo/3dtrees/blob/master/Pipfile.lock). Using pip, installation is as easy as 
running:
```
pip install 3dtrees-nbingo
```

## Questions
If you have any questions for how to use this code or for how it was used in [Cerebellar nuclei evolved by 
repeatedly duplicating a conserved cell type set](https://www.biorxiv.org/content/10.1101/2020.06.25.170118v1), then
please feel free to email me at nomir@stanford.edu. Examples usage of this package can be found in the
[cn_evolution](https://github.com/nbingo/cn_evolution) repository, which is the analysis code used to produce the
figures in the linked paper.
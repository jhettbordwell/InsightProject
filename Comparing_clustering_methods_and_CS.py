# To locate files
import glob

# To manage data
import pandas as pd
import numpy as np

# To generate random numbers and shuffle lists
from random import shuffle
from numpy.random import randint

# To perform clustering
from sklearn.decomposition import PCA # PCA + K Means
from sklearn.cluster import KMeans

from kmodes import kmodes # K Modes

import nimfa
from nimfa import Nmf  # NMF
from nimfa import Snmf # Sparse NMF

# To generate visualizations
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# GRAND SCHEME ----------------------
# *1. ..Validate CS based on user reviews that mention their medication journey (very rough)
#     ---> Do this once the below is running

# 1. Fit a model with N random trials ----> Check what stats can get out of fitters to evaluate
# 2. ...fit the <N> model to a range of n_clusters
# 3. ...fit all types of models to that range

# 4. ...Generate two samples of user preferences [1 = most commonly mentioned, 2 = mentioned infreq]
# 5. ...Rank medications for each <N> model for each n_cluster
# 6. ...Calculate CS-ranked list
# 7. ...Calculate RBO between CS list and list from each <N> model for each n_cluster
# 8. ...Plot normalized RBOs vs n_cluster

# 9. ...Find the optimal number of clusters for each method, and then run it wild on random prefs
# 10. ..Plot a distribution (hist) of RBOs for each model at that optimal number of clusters
# 11. ..ID best method based on median RBO, run time





# Data processing bits----------------------------------------------------------
# Need to introduce cutoff for number of reviews per med, or account for that somehow
# Need to have function that takes condition processed reviews and maps to cluster mentions



# Just looking at the clustering piece first, so need functions for each routine
def fit_kMeans():
    pass

def fit_kModes():
    pass

def fit_NMF():
    pass

def fit_SNMF():
    pass

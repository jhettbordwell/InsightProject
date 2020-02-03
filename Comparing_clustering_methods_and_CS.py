# To locate files
import glob
import h5py

# To manage data
import pandas as pd
import numpy as np

# To perform clustering
from sklearn.decomposition import PCA # PCA + K Means
from sklearn.cluster import KMeans

from kmodes.kmodes import KModes # K Modes

import nimfa
from nimfa import Nmf  # NMF
from nimfa import Snmf # Sparse NMF

# To evaluate results
from scipy.optimize import fsolve

# To generate visualizations
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# GRAND SCHEME ----------------------
# *1. ..Validate CS based on user reviews that mention their medication journey (very rough)
#     ---> Do this once the below is running

## # 1. Fit a model with N random trials ----> Check what stats can get out of fitters to evaluate
## # 2. ...fit the <N> model to a range of n_clusters
## # 3. ...fit all types of models to that range

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
def mapReview2SEclusters(condition, answer=None, cutoff=100):
    import time
    st = time.time()

    # Identifying relevant medications
    ConditionFile = glob.glob('UniqueMedications/*{:s}*csv'.format(condition))[0]
    condDF = pd.read_csv(ConditionFile, sep='$', usecols=[1])
    condMeds = [med.strip() for med in list(condDF['Medication'])]

    # Reading in the massive processed dataframe
    dataframe = pd.read_csv('Final_processed_reviews/{:s}_processed.csv'.format(condition),
                            sep='$', index_col=0)

    # Identifying the medication columns and mapping back into a list of medications
    med_columns = list(condMeds)
    medsDF = dataframe.drop(columns=[col for col in dataframe.columns if col not in med_columns])
    medications = []
    for ind in medsDF.index:
        for col in medsDF:
            if medsDF.loc[ind][col]:
                medications.append(str(col))
    medications = np.array(medications)
                
    # Reading in cluster file
    clustering = pd.read_csv('SideEffectMatching/cluster_file.csv', sep='$', index_col=0)

    print(time.time()-st)
    
    if answer:
        # Calculating target variable                                                               
        SEs1 = clustering.loc[answer['SE1']][clustering.loc[answer['SE1']].eq(1)].index
        fSE1 = [dataframe[col] for col in SEs1 if col in dataframe.columns]
        if fSE1:
            fSE1 = (np.vstack(fSE1).sum(axis=0)>0)
        else:
            fSE1 = np.array([0]*len(dataframe))
            
        SEs2 = clustering.loc[answer['SE2']][clustering.loc[answer['SE2']].eq(1)].index
        fSE2 = [dataframe[col] for col in SEs2 if col in dataframe.columns]
        if fSE2:
            fSE2 = (np.vstack(fSE2).sum(axis=0)>0)
        else:
            fSE2 = np.array([0]*len(dataframe))
        
        SEs3 = clustering.loc[answer['SE3']][clustering.loc[answer['SE3']].eq(1)].index
        fSE3 = [dataframe[col] for col in SEs3 if col in dataframe.columns]
        if fSE3:
            fSE3 = (np.vstack(fSE3).sum(axis=0)>0)
        else:
            fSE3 = np.array([0]*len(dataframe))
    
        w0 = answer['eff_rating']

        EffStars = (dataframe['Effectiveness']-1)/4.
        wse1 = answer['SE1_rate']
        wse2 = answer['SE2_rate']
        wse3 = answer['SE3_rate']

        # This score was designed to measure how "compatible" a drug would be with a user based on a review
        w1 = 1# - w0
        CS = w0*EffStars - w1*( wse1*fSE1  +  wse2*fSE2  +  wse3*fSE3 ) / (wse1+wse2+wse3)
        CS += 1

        
        # Getting information about how to evaluate cluster results
        inds = []
        ratings = []
        for key in ['SE1', 'SE2', 'SE3']:
            inds.append([i for i,val in enumerate(clustering.index) if val == answer[key]][0])
            ratings.append(answer['{:s}_rate'.format(key)])
        inds.append(-1)
        ratings.append(answer['eff_rating'])
        

        # Handling the fact that we didn't save medication info when clustering
        if cutoff:
            kept_medications = medications.copy()
            n_revs = np.array([(medications == med).sum() for med in med_columns])
            min_reviews = cutoff 
        
            if (n_revs < min_reviews).any():
                for n_rev, med in zip(n_revs, med_columns):
                    if n_rev < min_reviews:
                        CS = CS[kept_medications != med]
                        kept_medications = kept_medications[kept_medications != med]
                        
            medications = kept_medications.copy()

        
        return CS, inds, ratings, medications
    
    
    # Creating a dataframe with cluster_info + effectiveness x reviews
    df_info = []
    for cluster in clustering.index:
        SEs = clustering.columns[clustering.loc[cluster].eq(1)]
        fSE = [dataframe[col] for col in SEs if col in dataframe.columns]
        if not fSE:
            df_info.append(np.zeros(len(dataframe)))
        else:
            df_info.append((np.vstack(fSE).sum(axis=0)>0))

    feature_frame = pd.DataFrame(np.array(df_info).T, columns = list(clustering.index))
    feature_frame = pd.concat([feature_frame, dataframe['Effectiveness']], axis=1)

    # Dropping medications with too few reviews
    if cutoff:
        kept_medications = medications.copy()
        n_revs = np.array([(medications == med).sum() for med in med_columns])
        min_reviews = cutoff 
        
        if (n_revs < min_reviews).any():
            for n_rev, med in zip(n_revs, med_columns):
                if n_rev < min_reviews:
                    drop_inds = np.where(medications == med)[0]
                    feature_frame = feature_frame.drop(index=drop_inds)
                    kept_medications = kept_medications[kept_medications != med]

        medications = kept_medications.copy()

    return medications, feature_frame




def getCS(condition, answerList=None, cutoff=100):
    import time
    st = time.time()

    # Identifying relevant medications
    ConditionFile = glob.glob('UniqueMedications/*{:s}*csv'.format(condition))[0]
    condDF = pd.read_csv(ConditionFile, sep='$', usecols=[1])
    condMeds = [med.strip() for med in list(condDF['Medication'])]

    # Reading in the massive processed dataframe
    dataframe = pd.read_csv('Final_processed_reviews/{:s}_processed.csv'.format(condition),
                            sep='$', index_col=0)

    # Identifying the medication columns and mapping back into a list of medications
    med_columns = list(condMeds)
    medsDF = dataframe.drop(columns=[col for col in dataframe.columns if col not in med_columns])
    medications = []
    for ind in medsDF.index:
        for col in medsDF:
            if medsDF.loc[ind][col]:
                medications.append(str(col))
    medications = np.array(medications)
                
    # Reading in cluster file
    clustering = pd.read_csv('SideEffectMatching/cluster_file.csv', sep='$', index_col=0)

    print(time.time()-st)


    CSList = []
    indsList = []
    ratingsList = []
    for answer in answerList:
        # Calculating target variable                                                               
        SEs1 = clustering.loc[answer['SE1']][clustering.loc[answer['SE1']].eq(1)].index
        fSE1 = [dataframe[col] for col in SEs1 if col in dataframe.columns]
        if fSE1:
            fSE1 = (np.vstack(fSE1).sum(axis=0)>0)
        else:
            fSE1 = np.array([0]*len(dataframe))
            
        SEs2 = clustering.loc[answer['SE2']][clustering.loc[answer['SE2']].eq(1)].index
        fSE2 = [dataframe[col] for col in SEs2 if col in dataframe.columns]
        if fSE2:
            fSE2 = (np.vstack(fSE2).sum(axis=0)>0)
        else:
            fSE2 = np.array([0]*len(dataframe))
        
        SEs3 = clustering.loc[answer['SE3']][clustering.loc[answer['SE3']].eq(1)].index
        fSE3 = [dataframe[col] for col in SEs3 if col in dataframe.columns]
        if fSE3:
            fSE3 = (np.vstack(fSE3).sum(axis=0)>0)
        else:
            fSE3 = np.array([0]*len(dataframe))
    
        w0 = answer['eff_rating']

        EffStars = (dataframe['Effectiveness']-1)/4.
        wse1 = answer['SE1_rate']
        wse2 = answer['SE2_rate']
        wse3 = answer['SE3_rate']

        # This score was designed to measure how "compatible" a drug would be with a user based on a review
        w1 = 1# - w0
        CS = w0*EffStars - w1*( wse1*fSE1  +  wse2*fSE2  +  wse3*fSE3 ) / (wse1+wse2+wse3)
        CS += 1

        
        # Getting information about how to evaluate cluster results
        inds = []
        ratings = []
        for key in ['SE1', 'SE2', 'SE3']:
            inds.append([i for i,val in enumerate(clustering.index) if val == answer[key]][0])
            ratings.append(answer['{:s}_rate'.format(key)])
        inds.append(-1)
        ratings.append(answer['eff_rating'])
        

        # Handling the fact that we didn't save medication info when clustering
        kept_medications = medications.copy()
        if cutoff:
            n_revs = np.array([(medications == med).sum() for med in med_columns])
            min_reviews = cutoff 
        
            if (n_revs < min_reviews).any():
                for n_rev, med in zip(n_revs, med_columns):
                    if n_rev < min_reviews:
                        CS = CS[kept_medications != med]
                        kept_medications = kept_medications[kept_medications != med]
        CSList.append(CS); indsList.append(inds) ; ratingsList.append(ratings)
        
    medications = kept_medications
    return CSList, indsList, ratingsList, medications



def convertData4kMeans(data, n_PCA):
    # Converting the categorical features to allow K Means
    pca = PCA(n_PCA)
    vectors = pca.fit_transform(data)
    feature_weights = pca.components_ # n_PCA * n_features
    return vectors, feature_weights
                    

# Just looking at the clustering piece first, so need functions for each routine


def fit_kMeans(data, n_cluster=2, n_PCA=4, N_trials=10):
    data_PCA, PCA_feature_weights = convertData4kMeans(data, n_PCA)

    kme = KMeans(n_clusters=n_cluster, random_state=616, n_init=N_trials)

    # Fit the data to find cluster centers and membership
    kme.fit(data_PCA)

    # Transforming from n_clust*n_PCA to n_clust*n_feat
    cluster_weights = kme.cluster_centers_
    cluster_feature_weights = cluster_weights @ PCA_feature_weights
    
    clusters = kme.labels_
    return clusters, cluster_feature_weights


def fit_kModes(data, n_cluster=2, N_trials=10):
    kmo = KModes(n_clusters=n_cluster, n_init=N_trials, init='Huang', random_state=616)
    clusters = kmo.fit_predict(data)
    cluster_feature_weights = kmo.cluster_centroids_
    return clusters, cluster_feature_weights


def fit_NMF(data, n_cluster=2, N_trials=10):
    nmf = Nmf(data, n_run=N_trials, rank=n_cluster, objective='fro', update='euclidean')
    fit = nmf()
    W = fit.basis()
    H = fit.coef()
              
    clusters = np.argmax(H,axis=0)
    cluster_feature_weights = W
    
    return clusters, cluster_feature_weights

def fit_SNMF(data, n_cluster=2, N_trials=10):
    snmf = Snmf(data, n_run=N_trials, rank=n_cluster, version='r', eta=1, beta=1e-4)
    fit = snmf()
    W = fit.basis()
    H = fit.coef()
              
    clusters = np.argmax(H,axis=0)
    cluster_feature_weights = W
    
    return clusters, cluster_feature_weights





# Running the mess as in the grand scheme
def write_clusterH5(clusters, cluster_feature_weights, model='KMeans',
                    condition='Depression',
                    cluster_number=2):
    with h5py.File('ClusterData/{:s}_{:s}_keq{:g}.h5'.format(condition, model, cluster_number), 'w') as F:
        F['labels'] = clusters
        F['weights'] = cluster_feature_weights



        
def fake_answer():
    answer = {}
    answer['SE1'] = 'Weight changes'
    answer['SE1_rate'] = 1
    answer['SE2'] = 'Sleep issues and drowsiness'
    answer['SE2_rate'] = 0.4
    answer['SE3'] = 'Changes in libido and sexual performance'
    answer['SE3_rate'] = 0.8
    answer['eff_rating'] = 0.5

    return answer


def create_random_answer(seed=0,
                         condition='Bipolar-Disorder', 
                         cluster_file='SideEffectMatching/cluster_file.csv'):
    
    clusterDF = pd.read_csv(cluster_file, sep='$', index_col=0)
    clusters = list(clusterDF.index)
    
    response = {}
    response['Diagnosis'] = condition
    
    # Randomly draw three side effect clusters
    np.random.seed(seed)
    np.random.shuffle(clusters)
    concerns = [cluster for cluster in clusters[:3]]
    response['SE1'] = concerns[0]
    response['SE2'] = concerns[1]
    response['SE3'] = concerns[2]
    
    # Randomly set three weights for tolerance and one for effectiveness
    tolerances = (np.random.randint(1,11,size=4)-1)/10. + 0.1
    response['SE1_rate'] = tolerances[0]
    response['SE2_rate'] = tolerances[1]
    response['SE3_rate'] = tolerances[2]
    
    # Consider effectiveness
    response['eff_rating'] = tolerances[3]
            
    return response


    
def get_clusters(condition='Depression', justKMeans=False):
    meds, feature_frame = mapReview2SEclusters(condition)

    for k in range(2,15):
        print('For {:g} clusters, I have run...'.format(k))

        # K Means
        print('K Means')
        clust, cfw = fit_kMeans(feature_frame, n_cluster=k)
        write_clusterH5(clust, cfw, model='KMeans', cluster_number=k, condition=condition)

        if not justKMeans:
            # K Modes
            print('K Modes')
            clust, cfw = fit_kModes(feature_frame, n_cluster=k)
            write_clusterH5(clust, cfw, model='KModes', cluster_number=k, condition=condition)
            
            # NMF
            print('NMF')
            clust, cfw = fit_NMF(np.array(feature_frame).T, n_cluster=k)
            write_clusterH5(clust, cfw, model='NMF', cluster_number=k, condition=condition)

            # SNMF
            print('SNMF')
            clust, cfw = fit_SNMF(np.array(feature_frame).T, n_cluster=k)
            write_clusterH5(clust, cfw, model='SNMF', cluster_number=k, condition=condition)


        print('\n\n')

        
def CS_combo(clustarr, weights):
    # Take the weights from the user answer, and the weights from the clustering, and find the
    # compatibility score of each cluster
    scaled_clustarr = clustarr*weights
    CS = scaled_clustarr[-1]-scaled_clustarr[:-1].sum(axis=0)/sum(weights[:-1])
    return CS
    
def rank_meds(CS_cluster, meds, clusters):
    # Using the compatibility score of each cluster, rank the medications based on which are best
    # represented in the most compatible cluster
    umeds = np.unique(meds)
    sorted_meds = []
    for med in umeds:
        inds = np.where(np.array(meds)==med)
        clusters = clusters.flatten()
        med_clust = clusters[inds]
        fracs = [(med_clust==i).sum()/len(med_clust) for i in range(len(CS_cluster))]
        sorted_meds.append(fracs)
    
    clust_ranked_inds = CS_cluster.argsort()[::-1]
    meds_best_clust = np.array(sorted_meds).T[clust_ranked_inds][0]
    ranked_meds = umeds[meds_best_clust.argsort()[::-1]]
    
    return ranked_meds

def external_rank_meds(CS, meds):
    umeds = np.unique(meds)
    median_score = []
    for med in umeds:
        inds = np.where(np.array(meds)==med)
        median_score.append(np.median(np.array(CS)[inds]))
        
    ext_ranked_meds = umeds[np.array(median_score).argsort()[::-1]]

    return ext_ranked_meds

def obtain_med_lists():
    # Find the list of medications from each clustering technique for a value of k
    pass

def find_p(d, guess=0.9):
    # Finding the root (i.e., where the tail of the sum contributes nothing)
    # Guessing high (as our d's are small), because guessing low throws off the solver
    # Finding the p value that puts 90% of the weight in the length of the list
    WRBO_Tail = lambda p: 0.1 - p**(d-1) + (1-p)/p * d * (np.log(1/(1-p))-(p**np.arange(1,d)/np.arange(1,d)).sum())

    p = fsolve(WRBO_Tail, guess)[0]
    
    return p

def nRBO(listA, listB):
    p = find_p(len(listA))

    k = len(listA)
    X_k = len(set(listA) & set(listB))
    f = 2*k - X_k

    X_d = lambda d: len(set(listA[:d]) & set(listB[:d]))

    # RBO = RBO_min in eqn
    RBO = (1-p)/p * (sum([(X_d(i)-X_k) * p**i/i for i in range(1,k+1)]) - X_k*np.log(1-p))

    # Highest base score achievable for list of len k and persistence p, eqn 22
    RBO_max = 1 - p**k - k*(1-p)/p * (sum([p**i/i for i in range(1,k+1)])+np.log(1-p))

    # The normalized RBO is therefore
    nRBO = RBO / RBO_max

    return nRBO


def RBO_score_methods(krange, featInds, user_weights, CS, medications, condition):
    # Finding the ranked list from the CS
    extList = external_rank_meds(CS, medications)

    # Finding the RBO between that list and that from each cluster from each technique
    methods = ['KMeans', 'KModes', 'NMF', 'SNMF']
    scores = []
    for k in krange:
        kscores = []
        
        for mtd in methods:
            fname = 'ClusterData/{:s}_{:s}_keq{:g}.h5'.format(condition, mtd, k)
            with h5py.File(fname,'r') as F:
                clusters = F['labels'][()]
                cfw = F['weights'][()].T
                if cfw.shape[0] == k:
                    cfw = cfw.T
                
            mtdList = rank_meds(CS_combo(cfw[featInds].T, user_weights),
                                medications, clusters)

            kscores.append(nRBO(mtdList, extList))
        scores.append(kscores)

    scores = np.array(scores).T
    return scores

def make_RBO_vs_k_plot(scores, condition):
    methods = ['KMeans', 'KModes', 'NMF', 'SNMF']

    # Plotting everything
    plt.figure(figsize=(8,6))
    for i, mtd in enumerate(methods):
        plt.plot(krange, scores[i], label=mtd)

    plt.xlabel('Number of clusters')
    plt.ylabel('Rank biased overlap (RBO)')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.title(condition)
    plt.savefig('RBO_chart_{:s}.png'.format(condition))

    print(scores)
    

            

    

if __name__=="__main__":
    # get_clusters('ADHD')
    # get_clusters('Anxiety')
    # get_clusters('Bipolar-Disorder')
    # get_clusters('Depression')
    # get_clusters('Schizophrenia')

    # Test on depression
    answer = fake_answer()
    CS, inds, ratings, meds = mapReview2SEclusters('Depression', answer=answer)
    krange = np.arange(2,15)

    scores = RBO_score_methods(krange, inds, ratings, CS, meds, 'Depression')
    make_RBO_vs_k_plot(scores, 'Depression')
    
    

    # conditions=['ADHD', 'Anxiety', 'Bipolar-Disorder', 'Depression', 'Schizophrenia']
    # for condition in conditions:
    #     kMe_dist = []
    #     kMo_dist = []
    #     NMF_dist = []
    #     SNMF_dist = []

    #     answerList = [create_random_answer(seed=i, condition=condition) for i in range(1000)]
    #     CSList, indsList, ratingsList, meds = getCS(condition, answerList=answerList)
    #     for i in range(1000):
    #         print("I'm on iteration {:g} for {:s}".format(i,condition))
    #         scores = RBO_score_methods(krange, indsList[i], ratingsList[i], CSList[i],
    #                                    meds, condition)
    #         kMe_dist.append(scores[0])
    #         kMo_dist.append(scores[1])
    #         NMF_dist.append(scores[2])
    #         SNMF_dist.append(scores[3])

    #     kMe_dist = np.vstack(kMe_dist)
    #     kMo_dist = np.vstack(kMo_dist)
    #     NMF_dist = np.vstack(NMF_dist)
    #     SNMF_dist = np.vstack(SNMF_dist)

    #     for k in krange:
    #         plt.hist(kMe_dist[:,k], alpha=0.5, bins=50, label='KMe')
    #         plt.hist(kMo_dist[:,k], alpha=0.5, bins=50, label='KMo')
    #         plt.hist(NMF_dist[:,k], alpha=0.5, bins=50, label='NMF')
    #         plt.hist(SNMF_dist[:,k], alpha=0.5, bins=50, label='SNMF')
    #         plt.legend(loc='best')
    #         plt.xlabel('nRBO')
    #         plt.ylabel('Counts')
    #         plt.title('{:g} Clusters'.format(k))
    #         plt.savefig('ClusterData/{:s}_keq{:g}_RBOHist.png'.format(condition, k))
    #         plt.clf()


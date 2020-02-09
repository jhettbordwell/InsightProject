import pandas as pd
import numpy as np
import glob as glob
import h5py

def Label500(conditions):
    listOdf = []
    for i,condition in enumerate(conditions):
        # Concatenate all the medications for a given condition
        data_files = glob.glob('ProcessedReviews/{:s}/*.csv'.format(condition))
        for j,dfile in enumerate(data_files):
            if j == 0:
                df = pd.read_csv(dfile, sep='$', index_col=0)
                df['Medication'] = dfile[dfile.rfind('/clean_')+7:dfile.rfind('_reviews.csv')]
            else:
                ndf = pd.read_csv(dfile, sep='$', index_col=0)
                ndf['Medication'] = dfile[dfile.rfind('/clean_')+7:dfile.rfind('_reviews.csv')]
                df = df.append(ndf, ignore_index=True)

        # Shuffle the dataframe and grab the first 100 shuffled reviews
        np.random.seed = i
        inds = np.arange(0,len(df))
        np.random.shuffle(inds)
        inds = inds[:100]
        df = df.loc[inds]
        df = df.reset_index()
        
        medCol = df['Medication']
        reviews = df['Full Review']
        condCol = [condition]*len(medCol)
        listOdf.append(pd.DataFrame(np.array([condCol, medCol, reviews]).T, 
                       columns=['Condition', 'Medication', 'Full Review']))
        
    fulldf = pd.concat(listOdf, axis=0).reset_index().drop(columns='index')
    return fulldf
    
conditions = ['ADHD', 'Anxiety', 'Bipolar-Disorder', 'Depression', 'Schizophrenia']
if not glob.glob('LabeledReviews.csv'):
    combo = Label500(conditions)
    combo.to_csv('LabeledReviews.csv', sep='$')
else:
    combo = pd.read_csv('LabeledReviews.csv', sep='$', index_col=0)

# The condition for a "side effect detection" are that
# 1. The side effects aren't mentioned for another drug, but for the drug in question
# 2. The side effects aren't mentioned as an absence ("I haven't experienced any weight gain")
# 3. There are side effects mentioned, of any kind (even if they're not in our clustered set,
#                                                   as we have to remake that)

# 1 = conditions 1-3 met
# 0 = not met


# Ensuring it saves when I'm tired of labeling
if not glob.glob('temporaryLabeledReviewInfo.h5'):
    results = np.zeros((2,len(combo)))
else:
    with h5py.File('temporaryLabeledReviewInfo.h5','r') as F:
        results = F['results'][()]

# Labeling the reviews    
for ind in combo.index:
    if ind not in results[0]:
        results[0][ind] = ind
        results[1][ind] = int(input(combo.loc[ind]['Full Review']))

        with h5py.File('temporaryLabeledReviewInfo.h5','w') as F:
            F['results'] = results

        print('\n\n')


# Making a big dataframe with all the relevant info
combo['Presence of side effect'] = results[1]
combo.to_csv('LabeledReviews/randomlySelectedReviews.csv', sep='$')

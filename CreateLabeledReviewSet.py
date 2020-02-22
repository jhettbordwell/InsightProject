"""
In this short bit of code, I wrote out a routine to allow me to hand label 500 reviews,
so that I could verify the use of my algorithm for detecting side effects against
classifiers using features of the reviews.
"""

import pandas as pd
import numpy as np
import glob as glob
import h5py

def Label500(conditions):
    """
    Input: The 5 psychiatric conditions under consideration for PsychedUp.

    Output: a dataframe with 100 randomly sampled reviews from each dataframe.
    """
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

        # Randomly sample 100 reviews from the dataframe
        df = df.sample(n=100, random_state=i).reset_index()

        # Create a dataframe with the shuffled reviews
        medCol = df['Medication']
        reviews = df['Full Review']
        condCol = [condition]*len(medCol)
        listOdf.append(pd.DataFrame(np.array([condCol, medCol, reviews]).T, 
                       columns=['Condition', 'Medication', 'Full Review']))

    # Concatenate that dataframe
    fulldf = pd.concat(listOdf, axis=0).reset_index().drop(columns='index')
    return fulldf


if __name__ == "__main__":
    # Selecting 500 reviews to hand label
    conditions = ['ADHD', 'Anxiety', 'Bipolar-Disorder', 'Depression', 'Schizophrenia']
    if not glob.glob('LabeledReviews.csv'):
        combo = Label500(conditions)
        combo.to_csv('LabeledReviews.csv', sep='$')
    else:
        combo = pd.read_csv('LabeledReviews.csv', sep='$', index_col=0)

    # Ensuring it saves when I'm tired of labeling
    if not glob.glob('temporaryLabeledReviewInfo.h5'):
        results = np.zeros((2,len(combo)))
    else:
        with h5py.File('temporaryLabeledReviewInfo.h5','r') as F:
            results = F['results'][()]

    # Labeling the reviews
    print("""
    The condition for a "side effect detection" are that,
    1. The side effects aren't mentioned for another drug, but for the drug in question
    2. The side effects aren't mentioned as an absence ("I haven't experienced any weight gain")
    3. There are side effects mentioned, of any kind (even if they're not in our clustered set,
                                                      as we have to remake that)

    In answer to the reviews, input:
    1 = conditions 1-3 met
    0 = not met
    """          )

    # Iterating over each index in the 500 reviews and asking for input on whether or not side effects are mentioned
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

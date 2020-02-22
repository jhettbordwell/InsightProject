"""
This code enables the very basic cross validation of my compatibility score by allowing me to 
quickly go through reviews and identify those that had a reviewer who switched medications by 
finding reviews that, 
    1. Mention 2 medications (more often indicates polypharmacy plus or rather than a medication switch)
    2. Mention side effects
    3. Show a marked preference for either the previous or current medication.

Having found those medications, I assigned user profiles with the following criteria:
    1. The medication that was preferred was treated as the second medication, EVEN if it had been tried
       first initially, with the motivation being that the idea was to test if a user's preferences 
       could be detected by PsychedUp, not to measure how well psychiatrists and doctors do at a first
       shot (which has many confounding factors).
    2. The three side effects were the three mentioned which drew the strongest emotional response. If 
       fewer were mentioned, fewer side effects were entered by the user (they were set to a randomly 
       chosen side effect, and weighted 0). If side effects for both medications were mentioned, side 
       effects that led to a switch were given priority, and the side effects for the preferred drug
       were used to place a bound on how much the user could tolerate side effects with respect to 
       effectiveness.
    3. Side effects that were blamed for a switch received a concern score of 10/10
    4. Side effects that were mentioned at all received a concern score of 5/10
    5. If a reviewer mentioned their new medication working better (or the old one not working), 
       tolerance concern was rated a 5/10
    6. If the above holds true, but side effects for the current drug were also mentioned, 
       the tolerance concern was rated 10/10
    7. If a reviewer mentioned the old medication working better than the preferred, the 
       tolerance concern was rated 1/10.
    8. Condition was taken from the condition associated with the review.
"""

import pandas as pd
import numpy as np
import glob

np.random.seed = 616

class crossValidator:
    def __init__(self):
        pass

    def gatherMentions(self, condition):
        """
        Gathering the reviews that mention 2 medications for a given condition, and flagging them
        as useful until 30 have been flagged (and then calling it good).
        """
        # Reading in the matched review files
        if condition == 'Depression':
            df = pd.read_csv('ReviewsMatched2SideEffects/{:s}_matched_22_37.csv'.format(condition),
                             sep='$', index_col=0)
        else:
            df = pd.read_csv('ReviewsMatched2SideEffects/{:s}_matched.csv'.format(condition),
                             sep='$', index_col=0)

        mentions = df['Medication mentions']

        df_info = []

        # Looping through all the reviews with 2 medication mentions UNLESS I find 30
        i = 0
        section = df[df['Medication mentions'].eq(2)]
        reviews = list(section['Full Review'])
        np.random.shuffle(reviews)
        while len(df_info) < 30:
            rev = reviews[i]
            print(rev)
            keep = input("\nWhat do you think?\n\n")
            if int(keep):
                df_info.append({'Review': rev})
            i += 1

        # Saving the flagged reviews
        df_info = pd.DataFrame(df_info)
        df_info.to_csv('CrossValidation/{:s}_reviews.csv'.format(condition), sep='$')
    
    def userStory(self, condition):
        """
        Creating the user profiles according to the rules at the top of this file.
        """
        
        # Gathering clusters for me to reference in a text doc
        df = pd.read_csv('ClusteredSideEffects.csv', sep='$', index_col=0)
        clusters = np.unique(df['Cluster'])
        for i, c in enumerate(clusters):
            print(i,'\t\t', c)

        # Creating the user story for the reviews
        revDF = pd.read_csv('CrossValidation/{:s}_reviews.csv'.format(condition),
                            sep='$', index_col=0)
        df_info = []
        for rev in revDF['Review']:
            item = {}
            item['Review'] = rev

            print(rev, "\n\n")
            item['Medication 1'] = input("What is the first medication?\t")
            item['Medication 2'] = input("What is the second medication?\t")
            
            clust = int(input("Which cluster for SE1?\t"))
            item['SE1'] = clusters[clust]
            rate = int(input("What rating for SE1?\t"))
            item['SE1_rate'] = (rate-1)/9. # automatically converting the input to a weight
            
            clust = int(input("Which cluster for SE2?\t"))
            item['SE2'] = clusters[clust]
            rate = int(input("What rating for SE2?\t"))
            item['SE2_rate'] = (rate-1)/9.
            
            clust = int(input("Which cluster for SE3?\t"))
            item['SE3'] = clusters[clust]
            rate = int(input("What rating for SE3?\t"))
            item['SE3_rate'] = (rate-1)/9.

            eff = int(input("What about effectiveness?\t"))
            item['eff_rating'] = (eff-1)/9.

            df_info.append(item)

            # Saving the ratings results to a csv file
            finalDF = pd.DataFrame(df_info)
            finalDF.to_csv('CrossValidation/{:s}_reviewsAndstories.csv'.format(condition), sep='$')

    def classify(self, condition):
        """
        Ranking the medications based on the flagged reviews and user profiles for each condition.
        """

        # Reading in the results of both of my previous flagging/ratings
        finalDF = pd.read_csv('CrossValidation/{:s}_reviewsAndstories.csv'.format(condition),
                              sep='$', index_col=0)

        dataframe = pd.read_csv('FinalProcessedReviews/{:s}_processed.csv'.format(condition),
                                sep='$', index_col=0)

        rank_diffs = []
        for ind in finalDF.index:
            answers = finalDF.loc[ind]
            
            # Identifying the feature columns and gathering that data
            medications = dataframe['Medication']
        
            fSE1 = dataframe[answers['SE1']]
            fSE2 = dataframe[answers['SE2']]
            fSE3 = dataframe[answers['SE3']]
        
        
            # Calculating target variable
            w0 = answers['eff_rating']

            EffStars = (dataframe['Effectiveness']-1)/4
            
            wse1 = answers['SE1_rate']
            wse2 = answers['SE2_rate']
            wse3 = answers['SE3_rate']
        
            CS = w0*EffStars - ( wse1*fSE1  +  wse2*fSE2  +  wse3*fSE3 )
            CS_min = -(wse1+wse2+wse3)
            CS_max = w0
        
            normed_CS = (CS-CS_min) / (CS_max-CS_min)
            
            # Identifying medications
            med_nCS = []
            medsUniq = np.unique(medications)
            for i,med in enumerate(medsUniq):
                inds = dataframe[dataframe['Medication'].eq(med)].index
                score = normed_CS[inds]
                med_nCS.append(np.median(score))
            
            meds = medsUniq[np.argsort(med_nCS)[::-1]]
            med_nCS = np.array(med_nCS)[np.argsort(med_nCS)[::-1]]

            print("Before:", answers['Medication 1'],"\n",
                  "After:", answers['Medication 2'], "\n")
            
            for med, nCS in zip(meds, med_nCS):
                print(med, nCS)
            print("\n")

            # Finding the difference in the ranking between the two mentioned medications by hand given
            # the small sample size
            rank_diffs.append(int(input("What's the diff in the ranking?\t")))

        # Saving the results
        finalDF['Rank differences'] = rank_diffs
        finalDF.to_csv('CrossValidation/{:s}_reviewsStoriesAndranks.csv'.format(condition),
                       sep='$')

        # Reporting back so I can quickly get a feel for how well PsychedUp does for a given condition
        print("The mean difference in ranks was: ", np.mean(rank_diffs))

if __name__=="__main__":
    CV = crossValidator()

    # Only ADHD and Bipolar had enough reviews of the type I was looking for in the end to perform
    # this analysis
    for condition in ['ADHD', 'Bipolar-Disorder']:
        #for condition in ['ADHD', 'Anxiety', 'Bipolar-Disorder', 'Depression', 'Schizophrenia']: 
        if not glob.glob('CrossValidation/{:s}_reviews.csv'.format(condition)):
            CV.gatherMentions(condition)
        if not glob.glob('CrossValidation/{:s}_reviewsAndstories.csv'.format(condition)):
            CV.userStory(condition)
            
        CV.classify(condition)

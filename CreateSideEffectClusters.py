import numpy as np
import glob
import pandas as pd

class CreateClusters:
    """
    This class allows me to generally match the "side effects" that were identified by my algorithm
    to more general clusters which are more easily parseable by users.
    """
    def __init__(self):
        """
        When the class is initialized, I start off by predefining the clusters I'll be sorting
        the side effects into, and collecting the side effects that I'll be sorting.
        """
        self._predefined_cluster_topics()
        self._gatherSEs()
        
    def _predefined_cluster_topics(self):
        """
        Cluster definitions were chosen based on the top classes of side effects that were
        repeatedly mentioned when exploring the side effect identifications in the top 99.5%
        of detections. These have been typed in so that there are names that are human-readable
        for the website.

        As a note to remind myself, I did look into doing topic modeling via LSA/LDA/NMF, but I 
        found that reviews typically mentioned "side effects" as a single topic (essentially), 
        rather than commonly sharing a topic like "weight changes". The distinction is somewhat 
        subtle, but given that weight gain has relatively few words used to describe it (like 
        weight, gain/loss, decrease/increase, pounds, fat/skinny) and 
        most folks don't wax poetic on the subject, it doesn't stand out as a distinct topic
        easily from the data. And it was the most commonly mentioned side effect, so if topic 
        modeling didn't pick it up distinctly, it's easy to understand why it would fail to pick 
        up on something else.
        """

        self.clusters = ['Doesnt fit',
                         'Weight changes', 
                         'Mood and behavioral changes', 
                         'Vision changes',
                         'Headaches',
                         'Body aches and pain',
                         'Memory and concentration issues',
                         'Menstrual changes',
                         'Sleep issues and drowsiness',
                         'Balance, coordination, and muscle control',
                         'Dizziness and fainting',
                         'Stomach issues',
                         'Intestinal issues',
                         'Skin issues',
                         'Dry mouth and changes in taste',
                         'Blood sugar changes',
                         'Hair loss and abnormal growth',
                         'Changes in libido and sexual performance',
                         'Changes in energy',
                         'Sweating and temperature control issues',
                         'Eye itching or sensitivity changes',
                         'Blood pressure and heart rate changes',
                         'Changes in appetite',
                         'Urinary changes',
                         'Kidney issues',
                         'Hearing issues',
                         'Respiratory issues and coughing',
                         'Salivary issues',
                         'Breast growth and swelling (all genders)',
                         'Dental issues']

    def _gatherSEs(self):
        """
        Using the side effect information that was matched for each review using IdentifySideEffects.py,
        in this function I go through the results and drop any side effect that showed up in 
        0.05% or fewer of the reviews, and collect the names of the others to be hand clustered into different
        side effect bunches.
        """
        # Grab all files for every condition to sort
        files = glob.glob('ReviewsMatched2SideEffects/*csv')
        files = np.sort(files)

        # For each dataframe, grab top 99.5% of reviews, and just consider side effect info
        notSEcols = ['Full Review', 'Positive polarity', 'Negative polarity',
                     'Medication mentions', 'Effectiveness']

        side_effects = []
        for f in files:
            df = pd.read_csv(f, sep='$', index_col=0).drop(columns=notSEcols)
            for col in df.columns:
                if df[col].sum() >= 0.005*len(df):
                    side_effects.append(col)

        self.side_effects = side_effects

    def sortSEs(self):
        """
        For each unique side effect, I enter in the number of the cluster that best describes it 
        generally, and then write that information to a csv file for use farther down the pipeline.

        It should be noted that I do have a "doesn't fit" category, because certain side effects 
        were identified based on words that are not particularly descriptive, and so the information
        that is being matched isn't actually relevant.
        """
        # Starting off by printing to screen all of the information I need to know to do the sorting
        for i,cluster in enumerate(self.clusters):
            print("Cluster ID: {:g}\t\t Cluster: {:s}".format(i, cluster))
        
        df_info = []
        for SE in np.unique(self.side_effects):
            # Intelligently handling the possibility that I may fail to do the sorting continuously
            # by saving the results to a csv (which I can edit if I make a misstep) each entry.
            if glob.glob('ClusteredSideEffects.csv'):
                df = pd.read_csv('ClusteredSideEffects.csv', sep='$', index_col=0)
                if SE not in df['Side effect']:
                    item = {}
                    item['Side effect'] = SE
                    ind = input('\n'+SE)
                    cluster = self.clusters[int(ind)]
                    item['Cluster'] = cluster
                    df_info.append(item)
                else:
                    df_info.append({'Side effect': SE,
                                    'Cluster': df[df['Side effect'].eq(SE)]['Cluster']})
            else:
                item = {}
                item['Side effect'] = SE
                ind = input('\n'+SE)
                cluster = self.clusters[int(ind)]
                item['Cluster'] = cluster
                df_info.append(item)
                
            newdf = pd.DataFrame(df_info)
            newdf.to_csv('ClusteredSideEffects.csv', sep='$')


class Reviews2Clusters:
    """
    This class takes the results of CreateClusters and does a final processing of the review
    dataframes into a reduced frame that just includes yes/no information on whether the side
    effect was mentioned, medication info, sentiment information, the number of times medications 
    were mentioned, and effectiveness information, which is all the additional info I need to 
    perform the calculations on the website.
    """
    # A final processing of the reviews to map SE detections to smaller collection of clusters
    def __init__(self):
        self.conditions = ['ADHD', 'Anxiety', 'Bipolar-Disorder', 'Depression', 'Schizophrenia']

    def mapRev2Cluster(self):
        """
        This function takes the side effect clustering info, and translates the columns of the 
        reviews where side effects were identified into this smaller collection of categories,
        by just doing a boolean check of whether any of the side effects in the cluster were 
        mentioned in the review.
        """

        # For each condition, operating on the side effect matching file to reduce down into
        # the more general categories
        clusterMapping = pd.read_csv('ClusteredSideEffects.csv', sep='$', index_col=0)
        for condition in self.conditions:
            print("I'm working on {:s}".format(condition))
            files = glob.glob('ReviewsMatched2SideEffects/{:s}*csv'.format(condition))
            files = np.sort(files)

            for i,f in enumerate(files):
                df = pd.read_csv(f, sep='$', index_col=0)

                for cluster in np.unique(clusterMapping['Cluster']):
                    # Finding the relevant SEs for the cluster
                    SEs = clusterMapping[clusterMapping['Cluster'].eq(cluster)]['Side effect']

                    # Summing across all those SEs in the dataframe and creating a new column
                    match = [SE for SE in SEs if SE in df.columns]
                    df[cluster] = (df[match].sum(axis=1) > 0)
                    
                    if not match:
                        df[cluster] = [0]*len(df)
                    
                # Stacking to allow for the depression split
                if i == 0:
                    master_df = df.copy()
                else:
                    master_df = master_df.append(df, ignore_index=0, sort=False)


            # Dropping all columns not in clusters
            clusters = list(np.unique(clusterMapping['Cluster']))
            keepers = ['Medication','Positive polarity','Negative polarity','Medication mentions','Effectiveness']
            keepers += clusters
            master_df = master_df[keepers]
                    
            # Writing the stack to a file to load on to AWS
            master_df.to_csv('FinalProcessedReviews/{:s}_processed.csv'.format(condition), sep='$')
            print("I've saved the clustered file\n")
            
            
if __name__=="__main__":
    # Doing the hand sorting of the clusters
    recorder = CreateClusters()
    recorder.sortSEs()

    # Doing the automatic sorting of the reviews
    stacker = Reviews2Clusters()
    stacker.mapRev2Cluster()

import pandas as pd
import numpy as np
import glob

class ProcessReviews:
    """
    This class takes raw reviews scraped from WebMD and does all the parsing to match 
    information for each medication to each condition, and break apart the reviewer info
    into demographics (which I never used, but which has potential).
    
    The first four initialization functions (beginning with '_') serve to set a bunch of 
    attributes including information on the medications, condition, and raw reviews, and
    then to group the reviews based on medication to the relevant condition.

    The following functions process the demographic information, screen reviews for the actual
    condition they're discussing, drop medications that don't have more than 25 reviews, and
    drops reviews that don't have a text comment together with duplicates. These functions also
    account for the fact that webMD often shared reviews between slightly different versions of
    a medication (i.e., lamictal green and lamictal orange would both have the same set of reviews).
    """
    def __init__(self):
        self._defineCondsAndMeds()
        self._getReviewFiles()
        self._groupReviewsByMedAndCondition()
        self._conditionWords()
        
    def _defineCondsAndMeds(self, medFile='MedicationsAndSideEffects.csv'):
        # Reading in the medications and associated conditions
        dataframe = pd.read_csv(medFile, sep='$', index_col=0)
        self.medications = np.array(dataframe['Medication'])
        self.conditions = np.array(dataframe['Condition'])
        
    def _getReviewFiles(self, revDir='./Reviews/'):
        # Reading in all of the raw reviews
        self.reviews = glob.glob(revDir+'*csv')
        self.review_meds = [s[s.rfind('/')+1:s.find('_reviews.csv')] for s in self.reviews]

    def _groupReviewsByMedAndCondition(self):
        # Grouping reviews by the medication and condition into a dictionary
        self.conditionBunches = {}
        for condition in np.unique(self.conditions):
            inds = np.where(self.conditions==condition)
            self.conditionBunches[condition] = {}
            relReviews = []
            relReviewMeds = []
            for med in self.medications[inds]:
                clean_med = med.lower().replace(' ','-')
                relReviews += [rev for rev in self.reviews if rev.find(clean_med) != -1]
                relReviewMeds += [revm for revm in self.review_meds if revm.find(clean_med) != -1]
            self.conditionBunches[condition]['Reviews'] = relReviews
            self.conditionBunches[condition]['Review Medications'] = relReviewMeds
                
    def _conditionWords(self):
        # Words that are relevant for the conditions we're looking at
        self.addCondWords = {}
        self.addCondWords['ADHD'] = ['ADHD', 'Attention', 'ADD']
        self.addCondWords['Anxiety'] = ['Anxiety']
        self.addCondWords['Bipolar-Disorder'] = ['Bipolar', 'Bipolar Disorder', 'Manic', 'Mania']
        self.addCondWords['Depression'] = ['Depression', 'Depressive']
        self.addCondWords['Schizophrenia'] = ['Schizophrenia']


    def parse_reviewer(self, reviewer):
        """
        Takes all the information about the reviewer and breaks it down into separate demographics,
        when present.
        """
        
        # Find name as unique identifier if present
        if reviewer.find(',') != -1:
            name = reviewer[reviewer.find(':')+2:reviewer.find(',')]
        else:
            name = np.NaN
    
        # Find age range as datapoint if present
        if reviewer.find('-') != -1:
            if reviewer.find(',') != -1:
                age = reviewer[reviewer.find(',')+2:reviewer.find(' ', reviewer.find('-'))]
            else:
                age = reviewer[reviewer.find(':')+2:reviewer.find(' ', reviewer.find('-'))]
        else:
            age = np.NaN
        
        # Find gender if present
        if reviewer.find('Male') != -1:
            gender = 'Male'
        elif reviewer.find('Female') != -1:
            gender = 'Female'
        else:
            gender = np.NaN
        
        # Find treatment time
        if reviewer.find('on Treatment') != -1:
            if reviewer.rstrip()[-1] == ')':
                treatment_time = reviewer[reviewer.find('on Treatment for ')+16:reviewer.rfind('(')].strip()
            else:
                treatment_time = reviewer[reviewer.find('on Treatment for ')+16:].rstrip().strip()
        else:
            treatment_time = np.NaN
    
        # Put info in a dictionary that can be made into a row of a dataframe
        reviewer_info = {}
        reviewer_info['Name'] = name
        reviewer_info['Age'] = age
        reviewer_info['Gender'] = gender
        reviewer_info['Length of treatment'] = treatment_time
    
        return reviewer_info

    def processReviewerColumn(self,reviewDF):
        """
        Takes the dataframe of reviews, and adds columns with all the demographics information (dropping the reviewer column)
        """
        # Parse the reviewer info
        reviewers = []
        for reviewer in reviewDF['reviewer']:
            reviewers.append(self.parse_reviewer(reviewer))
        reviewersDF = pd.DataFrame(reviewers, index=reviewDF.index)

        # Drop the reviewer column from the original dataframe
        reviewDF = reviewDF.drop(columns=['reviewer'])

        # Add the parsed reviewer info to the original dataframe
        reviewDF = pd.concat([reviewDF, reviewersDF], axis=1)
    
        return reviewDF

    def dropRepeats(self,condition):
        """
        For all of the reviews pertaining to a condition, checking for,
        1. Duplicate rows between different medications for the same condition (defined as having the 
           exact same timestamp, which should be stringent because the timestep is down to 0.01 seconds)
        2. For a duplicate row, a long medication name (longest name is dropped, because it often contains
           extraneous information, like the color of the pill)
        """
        # Due to the fact that many links point to the same collection of reviews, need to drop
        # repeating documents
        reviews = self.conditionBunches[condition]['Reviews']
        revMeds = self.conditionBunches[condition]['Review Medications']
        keeprevs = []  ;  keeprevMeds = []
        for i,rev in enumerate(reviews):
            df1 = pd.read_csv(rev, sep='$', index_col=0)
            date1 = np.array(df1['date'])

            # Allowing for alterations to list that is being compared against so that no repeats
            # are compared against, and things go a little faster
            if i == 0:
                newrevs = list(reviews.copy())
                newmeds = list(revMeds.copy())

            removeNames = []
            removeRevs = []
            for rev2,revnem in zip(newrevs, newmeds):
                df2 = pd.read_csv(rev2, sep='$', index_col=0)
                date2 = np.array(df2['date'])

                # Should always be dates so don't have to worry about NaNs
                if np.array(date1 == date2, dtype=int).sum() == date1.size:
                    removeNames.append(revnem)
                    removeRevs.append(rev2)

            # Reducing down to the shortest description of the medication
            if removeNames:
                minInd = np.argmin([len(nem) for nem in removeNames])
                for nem, rev in zip(removeNames, removeRevs):
                    if nem != removeNames[minInd]:
                        newrevs.remove(rev)
                        newmeds.remove(nem)

        # Updating the saved lists
        self.conditionBunches[condition]['Reviews'] = newrevs
        self.conditionBunches[condition]['Review Medications'] = newmeds
                        

                        
    def cleanReviews(self,condition):
        """
        Sorting and parsing demographic information for each review, as well as dropping medications with < 25 reviews.
        """
        reviews = self.conditionBunches[condition]['Reviews']
        for rev in reviews:
            # Selecting for reviews based on condition
            df = pd.read_csv(rev, sep='$', index_col=0)
            conditions = df['conditionInfo']
            inds = []
            for ind, cond in enumerate(conditions):
                if [cond for word in self.addCondWords[condition] if cond.find(word) != -1]:
                    inds.append(ind)
            if inds:
                df = df.loc[np.array(inds)]
                conds = [df.loc[ind]['conditionInfo'][len('Condition: '):] for ind in df.index]
                df['conditionInfo'] = conds
                
                # Dropping rows without a text review and dropping duplicates
                df = df.drop(index=[ind for ind in df.index if type(df.loc[ind]['Full Review']) == float])
                df = df.drop_duplicates()
                df = df.reset_index()

                if len(df) >= 25:  
                    # Cleaning up the reviewer column
                    df = self.processReviewerColumn(df)
                
                    # Writing the review to a csv
                    df.to_csv('ProcessedReviews/{:s}/clean_'.format(condition)+rev[rev.rfind('/')+1:], sep='$')

    def processAllReviews(self):
        """
        Going through the reviews for every condition, removing duplicates and cleaning up the scraped review,
        as described more specifically in each function above.
        """
        for condition in self.conditionBunches:
            self.dropRepeats(condition)
            self.cleanReviews(condition)

if __name__ == '__main__':
    # Grabbing the conditions, medications, and reviews, bunching them appropriately
    Processer = ProcessReviews()

    # Grouping reviews by condition and parsing reviewer info, dropping duplicates, and reducing to reviews with text
    Processer.processAllReviews()

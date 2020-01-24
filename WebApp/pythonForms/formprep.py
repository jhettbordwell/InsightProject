# Tracing everything by following this website:
# https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/

from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SelectField, BooleanField, SubmitField
from wtforms.validators import DataRequired
import glob
import pandas as pd
import numpy as np

class conditionData:    
    def __init__(self, metadata_file='Medications_SideEffects_brandAndlinks.csv'):
        self.medsDF = pd.read_csv(metadata_file, index_col=0)
        
    def clean_medications(self):
        # Finding all duplicate medications and ensuring there aren't duplicates
        popem = []   # Tracking rows that are technically duplicates
        keepers = {} # Finding lists of all the matched medication names
        for ind, med in zip(self.medications.index, self.medications):
            # Removing medication from a dataframe and finding relevant alternate names
            testDF = self.medications.drop(index=ind)
            altnames = self.condDF.loc[ind]['Alternate names']

            if type(altnames) == str:
                altnames = altnames.split(', ')+[med]
                # Comparing against every other medication and finding matched cases
                for ind2, med2 in zip(testDF.index, testDF):
                    
                    # If there is a match, create a set with all the unique alternate names and save it
                    if med2 in altnames:
                        altnames2 = self.condDF.loc[ind2]['Alternate names']
                        if type(altnames2) == str:
                            allAltnames = list(set([med2]+altnames2.split(', '))|set(altnames))
                        else:
                            allAltnames = list(set(altnames + [med2]))
                            
                        # Handling case where more than one row fits those criteria
                        if med in keepers:
                            x = list(set(allAltnames)|set(keepers[med]))
                            x.sort()
                            keepers[med] = x
                        else:
                            allAltnames.sort()
                            keepers[med] =  allAltnames

                # Handling case where no altnames became a row
                if med not in keepers:
                    x = altnames
                    x.sort()
                    keepers[med] = x
                    
        # Creating a dataframe with all that information so I can drop duplicate rows
        organized = [{'Medication': key, 'All names': ', '.join(keepers[key])} for key in keepers]
        organizedDF = pd.DataFrame(organized)

        # Dropping rows that have same lists of alternate names (would be nice to do this by brandname...but not right now)
        organizedDF = organizedDF.drop_duplicates(subset=['All names'])
        organizedDF.to_csv('Medications_unique_{:s}.csv'.format(self.condition), sep='$')


    def pull_condition(self, condition):
        # Pulling condition
        print(condition)
        self.condition = condition
        self.medications = self.medsDF[self.medsDF['Condition'].eq(condition)]['Medication']
        self.condDF = self.medsDF[self.medsDF['Condition'].eq(condition)]
        if not glob.glob('Medications_unique_{:s}.csv'.format(self.condition)):
            self.clean_medications()
        self.unique_meds = pd.read_csv('Medications_unique_{:s}.csv'.format(self.condition), sep='$')

    def pull_SideEffects(self, medication):
        sideEffects = []
        for key in ['More common', 'Less common', 'Incidence not known']:
            ind = self.medsDF[self.medsDF['Medication'].eq(medication)].index[0]
            value = self.medsDF.loc[ind][key]
            if type(value) == str:
                value = value.split('; ')
                sideEffects += value
        return sideEffects

    def pull_allSideEffects(self):
        # Make a list of every side effect for every med for a given condition
        self.listOfSEs = []
        for ind, med in zip(self.unique_meds.index, self.unique_meds['Medication']):
            self.listOfSEs += self.pull_SideEffects(med)

        # Find all the unique side effects (Probably will have to curate)
        self.listOfSEs = list(set(self.listOfSEs))
        self.listOfSEs.sort() # Make this intelligble for human beings

        # Make this parseable for flask forms
        self.SEresponseList = []
        for i,SE in enumerate(self.listOfSEs):
            self.SEresponseList.append(SE)

        if not glob.glob('SideEffects/{:s}_SideEffects.csv'.format(self.condition)):
            SEList = np.array(self.SEresponseList)
            np.savetxt('SideEffects/{:s}_SideEffects.csv'.format(self.condition),
                       SEList, delimiter=',', fmt='%s')


class SetUpForm:
    def __init__(self, condition):

        # Creating a class instance to read the metadata file for all the reviews
        medicationInfo = conditionData()

        # Working with that diagnosis information
        medicationInfo.pull_condition(condition)
    
        # Side effects questions
        # Grabbing a list of every side effect for relevant meds
        medicationInfo.pull_allSideEffects()  



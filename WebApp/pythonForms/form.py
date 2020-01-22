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
            sideEffects += str(self.medsDF[self.medsDF['Medication'].eq(medication)][key]).split('; ')
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


            
class DiagnosisForm(FlaskForm):    
    # Defining question and choices (to match metadata file) by hand
    diagnosis_question = u"What is your (or your patient's) preliminary diagnosis?"
    diagnosis_choices = [('Bipolar-Disorder', 'Bipolar Disorder (type I or II)'),
                         ('Depression', 'Depression'),
                         ('Anxiety', 'Anxiety'),
                         ('ADHD','ADHD'),
                         ('Schizophrenia','Schizophrenia')]
    diagnosis = SelectField(diagnosis_question, choices=diagnosis_choices,
                            validators=[DataRequired()])

    submitDiagnosis = SubmitField("Begin with this diagnosis")


class SetUpForm:
    def __init__(self, condition):

        # Creating a class instance to read the metadata file for all the reviews
        medicationInfo = conditionData()

        # Working with that diagnosis information
        medicationInfo.pull_condition(condition)
    
        # Side effects questions
        # Grabbing a list of every side effect for relevant meds
        medicationInfo.pull_allSideEffects()  

# class ADHDForm(FlaskForm):

#     SEresponseList = np.genfromtxt('SideEffects/ADHD_SideEffects.csv',delimiter=',',
#                                    dtype=str)

#     concern1_question = u"What is the 1st side effect concern you have about your medication?"
#     concern1 = SelectField(concern_question, choices=SEresponseList)
        
#     concern1_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern1_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     concern2_question = u"What is the 2nd side effect concern you have about your medication?"
#     concern2 = SelectField(concern_question, choices=SEresponseList)
        
#     concern2_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern2_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     concern3_question = u"What is the 3rd side effect concern you have about your medication?"
#     concern3 = SelectField(concern_question, choices=SEresponseList)
        
#     concern3_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern3_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     # Lifestyle concerns
#     lifestyle_question = u"(STRETCH GOAL) I am worried about interactions with: "
#     lifestyle_choices = [(1, 'Alcohol'), (2, 'Marijuana'), (3, 'Both')]
#     lifestyle = SelectField(lifestyle_question, lifestyle_choices)

#     # Addiction concerns
#     addiction_question = u"(STRETCH GOAL) Are you worried about addiction to your medication? "
#     addiction_choices = [(1, 'Yes'), (0, 'No')]
#     addiction = SelectField(addiction_question, addiction_choices)

#     # Comparison of side effects to effectiveness
#     eff_question = u"On a scale of 1 to 10, how tolerable would these side effects be if the medication were effective at treating your condition?"
#     eff_rating = SelectField(eff_question,
#                              choices=[(0.1*(i-1),i) for i in range(1,11)],
#                              validators=[DataRequired()])
#     submit = SubmitField("Let's do the research")

# class AnxietyForm(FlaskForm):

#     SEresponseList = np.genfromtxt('SideEffects/Anxiety_SideEffects.csv',delimiter=',',
#                                    dtype=str)

#     concern1_question = u"What is the 1st side effect concern you have about your medication?"
#     concern1 = SelectField(concern_question, choices=SEresponseList)
        
#     concern1_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern1_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     concern2_question = u"What is the 2nd side effect concern you have about your medication?"
#     concern2 = SelectField(concern_question, choices=SEresponseList)
        
#     concern2_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern2_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     concern3_question = u"What is the 3rd side effect concern you have about your medication?"
#     concern3 = SelectField(concern_question, choices=SEresponseList)
        
#     concern3_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern3_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     # Lifestyle concerns
#     lifestyle_question = u"(STRETCH GOAL) I am worried about interactions with: "
#     lifestyle_choices = [(1, 'Alcohol'), (2, 'Marijuana'), (3, 'Both')]
#     lifestyle = SelectField(lifestyle_question, lifestyle_choices)

#     # Addiction concerns
#     addiction_question = u"(STRETCH GOAL) Are you worried about addiction to your medication? "
#     addiction_choices = [(1, 'Yes'), (0, 'No')]
#     addiction = SelectField(addiction_question, addiction_choices)

#     # Comparison of side effects to effectiveness
#     eff_question = u"On a scale of 1 to 10, how tolerable would these side effects be if the medication were effective at treating your condition?"
#     eff_rating = SelectField(eff_question,
#                              choices=[(0.1*(i-1),i) for i in range(1,11)],
#                              validators=[DataRequired()])
#     submit = SubmitField("Let's do the research")
    
# class BipolarForm(FlaskForm):

#     SEresponseList = np.genfromtxt('SideEffects/Bipolar-Disorder_SideEffects.csv',delimiter=',',
#                                    dtype=str)

#     concern1_question = u"What is the 1st side effect concern you have about your medication?"
#     concern1 = SelectField(concern_question, choices=SEresponseList)
        
#     concern1_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern1_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     concern2_question = u"What is the 2nd side effect concern you have about your medication?"
#     concern2 = SelectField(concern_question, choices=SEresponseList)
        
#     concern2_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern2_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     concern3_question = u"What is the 3rd side effect concern you have about your medication?"
#     concern3 = SelectField(concern_question, choices=SEresponseList)
        
#     concern3_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern3_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     # Lifestyle concerns
#     lifestyle_question = u"(STRETCH GOAL) I am worried about interactions with: "
#     lifestyle_choices = [(1, 'Alcohol'), (2, 'Marijuana'), (3, 'Both')]
#     lifestyle = SelectField(lifestyle_question, lifestyle_choices)

#     # Addiction concerns
#     addiction_question = u"(STRETCH GOAL) Are you worried about addiction to your medication? "
#     addiction_choices = [(1, 'Yes'), (0, 'No')]
#     addiction = SelectField(addiction_question, addiction_choices)

#     # Comparison of side effects to effectiveness
#     eff_question = u"On a scale of 1 to 10, how tolerable would these side effects be if the medication were effective at treating your condition?"
#     eff_rating = SelectField(eff_question,
#                              choices=[(0.1*(i-1),i) for i in range(1,11)],
#                              validators=[DataRequired()])
#     submit = SubmitField("Let's do the research")

# class DepressionForm(FlaskForm):

#     SEresponseList = np.genfromtxt('SideEffects/Depression_SideEffects.csv',delimiter=',',
#                                    dtype=str)

#     concern1_question = u"What is the 1st side effect concern you have about your medication?"
#     concern1 = SelectField(concern_question, choices=SEresponseList)
        
#     concern1_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern1_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     concern2_question = u"What is the 2nd side effect concern you have about your medication?"
#     concern2 = SelectField(concern_question, choices=SEresponseList)
        
#     concern2_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern2_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     concern3_question = u"What is the 3rd side effect concern you have about your medication?"
#     concern3 = SelectField(concern_question, choices=SEresponseList)
        
#     concern3_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern3_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     # Lifestyle concerns
#     lifestyle_question = u"(STRETCH GOAL) I am worried about interactions with: "
#     lifestyle_choices = [(1, 'Alcohol'), (2, 'Marijuana'), (3, 'Both')]
#     lifestyle = SelectField(lifestyle_question, lifestyle_choices)

#     # Addiction concerns
#     addiction_question = u"(STRETCH GOAL) Are you worried about addiction to your medication? "
#     addiction_choices = [(1, 'Yes'), (0, 'No')]
#     addiction = SelectField(addiction_question, addiction_choices)

#     # Comparison of side effects to effectiveness
#     eff_question = u"On a scale of 1 to 10, how tolerable would these side effects be if the medication were effective at treating your condition?"
#     eff_rating = SelectField(eff_question,
#                              choices=[(0.1*(i-1),i) for i in range(1,11)],
#                              validators=[DataRequired()])
#     submit = SubmitField("Let's do the research")

# class SchizophreniaForm(FlaskForm):

#     SEresponseList = np.genfromtxt('SideEffects/Schizophrenia_SideEffects.csv',delimiter=',',
#                                    dtype=str)

#     concern1_question = u"What is the 1st side effect concern you have about your medication?"
#     concern1 = SelectField(concern_question, choices=SEresponseList)
        
#     concern1_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern1_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     concern2_question = u"What is the 2nd side effect concern you have about your medication?"
#     concern2 = SelectField(concern_question, choices=SEresponseList)
        
#     concern2_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern2_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     concern3_question = u"What is the 3rd side effect concern you have about your medication?"
#     concern3 = SelectField(concern_question, choices=SEresponseList)
        
#     concern3_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
#     concern3_rating = SelectField(concern_rating_question,
#                                      choices=[(0.1*(i-1),i) for i in range(1,11)])

#     # Lifestyle concerns
#     lifestyle_question = u"(STRETCH GOAL) I am worried about interactions with: "
#     lifestyle_choices = [(1, 'Alcohol'), (2, 'Marijuana'), (3, 'Both')]
#     lifestyle = SelectField(lifestyle_question, lifestyle_choices)

#     # Addiction concerns
#     addiction_question = u"(STRETCH GOAL) Are you worried about addiction to your medication? "
#     addiction_choices = [(1, 'Yes'), (0, 'No')]
#     addiction = SelectField(addiction_question, addiction_choices)

#     # Comparison of side effects to effectiveness
#     eff_question = u"On a scale of 1 to 10, how tolerable would these side effects be if the medication were effective at treating your condition?"
#     eff_rating = SelectField(eff_question,
#                              choices=[(0.1*(i-1),i) for i in range(1,11)],
#                              validators=[DataRequired()])
#     submit = SubmitField("Let's do the research")
    
class LetsGo(FlaskForm):
    submit = SubmitField("Let's get started!")




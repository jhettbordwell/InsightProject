# Tracing everything by following this website:
# https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/

from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SelectField, BooleanField, SubmitField
from wtforms.validators import DataRequired
import glob
import pandas as pd
import numpy as np


            
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

class ADHDForm(FlaskForm):

    SEresponseList = np.genfromtxt('SideEffects/ADHD_SideEffects.csv',delimiter='$',
                                   dtype=str)
    SEresponseList = [(SE,SE) for SE in SEresponseList if SE.strip()]
    
    concern1_question = u"What is the 1st side effect concern you have about your medication?"
    concern1 = SelectField(concern1_question, choices=SEresponseList, validators=[DataRequired()])
        
    concern1_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern1_rating = SelectField(concern1_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)], validators=[DataRequired()])

    concern2_question = u"What is the 2nd side effect concern you have about your medication?"
    concern2 = SelectField(concern2_question, choices=SEresponseList)
        
    concern2_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern2_rating = SelectField(concern2_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)])

    concern3_question = u"What is the 3rd side effect concern you have about your medication?"
    concern3 = SelectField(concern3_question, choices=SEresponseList)
        
    concern3_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern3_rating = SelectField(concern3_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)])

    # Lifestyle concerns
    lifestyle_question = u"(STRETCH GOAL) I am worried about interactions with: "
    lifestyle_choices = [(1, 'Alcohol'), (2, 'Marijuana'), (3, 'Both')]
    lifestyle = SelectField(lifestyle_question, choices=lifestyle_choices)

    # Addiction concerns
    addiction_question = u"(STRETCH GOAL) Are you worried about addiction to your medication? "
    addiction_choices = [(1, 'Yes'), (0, 'No')]
    addiction = SelectField(addiction_question, choices=addiction_choices)

    # Comparison of side effects to effectiveness
    eff_question = u"On a scale of 1 to 10, how tolerable would these side effects be if the medication were effective at treating your condition?"
    eff_rating = SelectField(eff_question,
                             choices=[(0.1*(i-1),str(i)) for i in range(1,11)],
                             validators=[DataRequired()])
    submit = SubmitField("Let's do the research")

class AnxietyForm(FlaskForm):

    SEresponseList = np.genfromtxt('SideEffects/Anxiety_SideEffects.csv',delimiter='$',
                                   dtype=str)
    SEresponseList = [(SE,SE) for SE in SEresponseList if SE.strip()]
    
    concern1_question = u"What is the 1st side effect concern you have about your medication?"
    concern1 = SelectField(concern1_question, choices=SEresponseList, validators=[DataRequired()])
        
    concern1_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern1_rating = SelectField(concern1_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)], validators=[DataRequired()])

    concern2_question = u"What is the 2nd side effect concern you have about your medication?"
    concern2 = SelectField(concern2_question, choices=SEresponseList)
        
    concern2_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern2_rating = SelectField(concern2_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)])

    concern3_question = u"What is the 3rd side effect concern you have about your medication?"
    concern3 = SelectField(concern3_question, choices=SEresponseList)
        
    concern3_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern3_rating = SelectField(concern3_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)])

    # Lifestyle concerns
    lifestyle_question = u"(STRETCH GOAL) I am worried about interactions with: "
    lifestyle_choices = [(1, 'Alcohol'), (2, 'Marijuana'), (3, 'Both')]
    lifestyle = SelectField(lifestyle_question, choices=lifestyle_choices)

    # Addiction concerns
    addiction_question = u"(STRETCH GOAL) Are you worried about addiction to your medication? "
    addiction_choices = [(1, 'Yes'), (0, 'No')]
    addiction = SelectField(addiction_question, choices=addiction_choices)

    # Comparison of side effects to effectiveness
    eff_question = u"On a scale of 1 to 10, how tolerable would these side effects be if the medication were effective at treating your condition?"
    eff_rating = SelectField(eff_question,
                             choices=[(0.1*(i-1),str(i)) for i in range(1,11)],
                             validators=[DataRequired()])
    submit = SubmitField("Let's do the research")
    
class BipolarForm(FlaskForm):

    SEresponseList = np.genfromtxt('SideEffects/Bipolar-Disorder_SideEffects.csv',delimiter='$',
                                   dtype=str)
    SEresponseList = [(SE,SE) for SE in SEresponseList if SE.strip()]
    
    concern1_question = u"What is the 1st side effect concern you have about your medication?"
    concern1 = SelectField(concern1_question, choices=SEresponseList, validators=[DataRequired()])
        
    concern1_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern1_rating = SelectField(concern1_rating_question,
                                  choices=[(0.1*(i-1),str(i)) for i in range(1,11)],
                                  validators=[DataRequired()])

    concern2_question = u"What is the 2nd side effect concern you have about your medication?"
    concern2 = SelectField(concern2_question, choices=SEresponseList)
        
    concern2_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern2_rating = SelectField(concern2_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)])

    concern3_question = u"What is the 3rd side effect concern you have about your medication?"
    concern3 = SelectField(concern3_question, choices=SEresponseList)
        
    concern3_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern3_rating = SelectField(concern3_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)])

    # # Lifestyle concerns
    lifestyle_question = u"(STRETCH GOAL) I am worried about interactions with: "
    lifestyle_choices = [(1, 'Alcohol'), (2, 'Marijuana'), (3, 'Both')]
    lifestyle = SelectField(lifestyle_question, choices=lifestyle_choices)

    # Addiction concerns
    addiction_question = u"(STRETCH GOAL) Are you worried about addiction to your medication? "
    addiction_choices = [(1, 'Yes'), (0, 'No')]
    addiction = SelectField(addiction_question, choices=addiction_choices)

    # Comparison of side effects to effectiveness
    eff_question = u"On a scale of 1 to 10, how tolerable would these side effects be if the medication were effective at treating your condition? "
    eff_rating = SelectField(eff_question,
                             choices=[(0.1*(i-1),str(i)) for i in range(1,11)],
                             validators=[DataRequired()])
    submit = SubmitField("Let's do the research")

class DepressionForm(FlaskForm):

    SEresponseList = np.genfromtxt('SideEffects/Depression_SideEffects.csv',delimiter='$',
                                   dtype=str)
    SEresponseList = [(SE,SE) for SE in SEresponseList if SE.strip()]
    
    concern1_question = u"What is the 1st side effect concern you have about your medication?"
    concern1 = SelectField(concern1_question, choices=SEresponseList, validators=[DataRequired()])
        
    concern1_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern1_rating = SelectField(concern1_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)], validators=[DataRequired()])

    concern2_question = u"What is the 2nd side effect concern you have about your medication?"
    concern2 = SelectField(concern2_question, choices=SEresponseList)
        
    concern2_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern2_rating = SelectField(concern2_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)])

    concern3_question = u"What is the 3rd side effect concern you have about your medication?"
    concern3 = SelectField(concern3_question, choices=SEresponseList)
        
    concern3_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern3_rating = SelectField(concern3_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)])

    # Lifestyle concerns
    lifestyle_question = u"(STRETCH GOAL) I am worried about interactions with: "
    lifestyle_choices = [(1, 'Alcohol'), (2, 'Marijuana'), (3, 'Both')]
    lifestyle = SelectField(lifestyle_question, choices=lifestyle_choices)

    # Addiction concerns
    addiction_question = u"(STRETCH GOAL) Are you worried about addiction to your medication? "
    addiction_choices = [(1, 'Yes'), (0, 'No')]
    addiction = SelectField(addiction_question, choices=addiction_choices)

    # Comparison of side effects to effectiveness
    eff_question = u"On a scale of 1 to 10, how tolerable would these side effects be if the medication were effective at treating your condition?"
    eff_rating = SelectField(eff_question,
                             choices=[(0.1*(i-1),str(i)) for i in range(1,11)],
                             validators=[DataRequired()])
    submit = SubmitField("Let's do the research")

class SchizophreniaForm(FlaskForm):

    SEresponseList = np.genfromtxt('SideEffects/Schizophrenia_SideEffects.csv',delimiter='$',
                                   dtype=str)
    SEresponseList = [(SE,SE) for SE in SEresponseList if SE.strip()]
    
    concern1_question = u"What is the 1st side effect concern you have about your medication?"
    concern1 = SelectField(concern1_question, choices=SEresponseList, validators=[DataRequired()])
        
    concern1_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern1_rating = SelectField(concern1_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)], validators=[DataRequired()])

    concern2_question = u"What is the 2nd side effect concern you have about your medication?"
    concern2 = SelectField(concern2_question, choices=SEresponseList)
        
    concern2_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern2_rating = SelectField(concern2_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)])

    concern3_question = u"What is the 3rd side effect concern you have about your medication?"
    concern3 = SelectField(concern3_question, choices=SEresponseList)
        
    concern3_rating_question = u"How much of a concern is this for you on a scale of 1 to 10?"
    concern3_rating = SelectField(concern3_rating_question,
                                     choices=[(0.1*(i-1),str(i)) for i in range(1,11)])

    # Lifestyle concerns
    lifestyle_question = u"(STRETCH GOAL) I am worried about interactions with: "
    lifestyle_choices = [(1, 'Alcohol'), (2, 'Marijuana'), (3, 'Both')]
    lifestyle = SelectField(lifestyle_question, choices=lifestyle_choices)

    # Addiction concerns
    addiction_question = u"(STRETCH GOAL) Are you worried about addiction to your medication? "
    addiction_choices = [(1, 'Yes'), (0, 'No')]
    addiction = SelectField(addiction_question, choices=addiction_choices)

    # Comparison of side effects to effectiveness
    eff_question = u"On a scale of 1 to 10, how tolerable would these side effects be if the medication were effective at treating your condition?"
    eff_rating = SelectField(eff_question,
                             choices=[(0.1*(i-1),str(i)) for i in range(1,11)],
                             validators=[DataRequired()])
    submit = SubmitField("Let's do the research")
    
class LetsGo(FlaskForm):
    submit = SubmitField("Let's get started!")




import pandas as pd
import numpy as np
import glob

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor



def fit_and_predict(data, target, split=0.05, n_est=1000):
    # Splitting data appropriately to start off
    x_train, x_test, y_train, y_test = train_test_split(np.array(data), target,
                                                        test_size=split, random_state=616)

    # Doing the logistic regression bit
    logReg = LinearRegression()#LogisticRegression()
    logReg.fit(x_train, y_train)
    predLR = logReg.predict(x_test)
    SE_LR = (predLR-y_test)**2

    
    # Doing the random forest bit
    # rf = RandomForestRegressor(n_estimators=n_est, random_state=616)
    # rf.fit(x_train, y_train)
    # predRF = rf.predict(x_test)
    # SE_RF = (predRF-y_test)**2

    return (data.columns,np.abs(logReg.coef_), SE_LR)#, (rf,SE_RF)


def calculate_targetAndData(answers):
    # Pulling out info on the diagnosis
    condition = answers['Diagnosis']

    # Identifying relevant medications
    ConditionFile = glob.glob('UniqueMedications/*{:s}*csv'.format(condition))[0]
    DF = pd.read_csv(ConditionFile, sep='$', usecols=[1])
    condMeds = [med.strip() for med in list(DF['Medication'])]

    # Reading in the massive processed dataframe
    dataframe = pd.read_csv('MONSTER_DFs/{:s}_processed.csv'.format(condition),
                            sep='$', index_col=0)

    # Identifying the feature columns and gathering that data
    feature_columns = list(condMeds)+['Postive polarity fraction','Negative polarity fraction']
    DF = dataframe.drop(columns=[col for col in dataframe.columns if col not in feature_columns])
    
    # Calculating target variable
    w0 = answers['eff_rating']
    w1 = 1 - w0
    
    EffStars = (dataframe['Effectiveness']-1)/4

    fSE1 = dataframe[answers['SE1']]/(3*0.9)
    fSE2 = dataframe[answers['SE2']]/(3*0.9)
    fSE3 = dataframe[answers['SE3']]/(3*0.9)
    
    wse1 = answers['SE1_rate']
    wse2 = answers['SE2_rate']
    wse3 = answers['SE3_rate']
    
    CS = w0*EffStars + w1*( wse1*fSE1  +  wse2*fSE2  +  wse3*fSE3 )

    # Returning dataframe to fit and compatibility score
    return DF, CS

def rank_medications(answers):
    # Putting everything together
    data, target = calculate_targetAndData(answers)

    features, scores, error = fit_and_predict(data, target)



    
    # Pulling out info on the diagnosis
    condition = answers['Diagnosis']

    # Identifying relevant medications
    ConditionFile = glob.glob('UniqueMedications/*{:s}*csv'.format(condition))[0]
    DF = pd.read_csv(ConditionFile, sep='$', usecols=[2])
    condMeds = [med.strip() for med in list(DF['All names'])]

    meds = [features[i] for i in scores.argsort()[::-1]]
    match = []
    for med in meds:
        for cM in condMeds:
            if cM.find(med) != -1:
                match.append(med +' ({:s})'.format(', '.join([m for m in cM.split(', ') if m != med])))
    
    return match


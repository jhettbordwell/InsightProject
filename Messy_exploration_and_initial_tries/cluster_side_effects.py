import pandas as pd
import numpy as np
import glob as glob

def getTop995Perc(condition):
    data_file = 'Final_processed_reviews/{:s}_processed.csv'.format(condition)
    df = pd.read_csv(data_file, sep='$', index_col=0)

    meds_file = 'UniqueMedications/Medications_unique_{:s}.csv'.format(condition)
    medications = [med.strip() for med in pd.read_csv(other_file, sep='$')['Medication']]

    columns_to_process = [col for col in df.columns if col not in (medications+['Effectiveness',
                                                                                'Satisfaction',
                                                                                'Positive polarity',
                                                                                'Negative polarity',
                                                                                'Full Review',
                                                                                'Age',
                                                                                'Gender',
                                                                                'Length of treatment'])]

    # Dropping all columns that aren't side effects or where a side effect shows up in fewer than
    # 0.5% of the reviews ---> WHAT ABOUT MEDS THAT HAVE A NUMBER OF REVIEWS ON THIS ORDER?
    # ---> DROPPING MEDS WITH LESS THAN 50 reviews, don't have more than 10,000 total
    list_col = columns_to_process.copy()
    for col in list_col:
        if df[col].sum() <= 0.01*df.shape[0]:
            columns_to_process.remove(col)

    return columns_to_process


all_SEs = []
for condition in ['ADHD', 'Anxiety', 'Bipolar-Disorder', 'Depression', 'Schizophrenia']:
    all_SEs += getTop995Perc(condition)

uniq_SEs = np.unique(all_SEs)



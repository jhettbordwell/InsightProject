import numpy as np
import glob
import pandas as pd

class CreateClusters:
    def __init__(self, condition):
        self._predefine_cluster_topics()
        self.condition = condition
        
    def _predefined_cluster_topics(self):
        self.clusters = ['Doesnt fit',
                         'Weight changes', 
                         'Mood and behavioral changes', 
                         'Vision changes',
                         'Headaches and',
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

        
        for i, c in enumerate(self.clusters):
            print(i, '\t\t', c)

    def sortSEs(self):
        

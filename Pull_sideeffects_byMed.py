"""
This function takes the reported side effects for each medication that has reports in FAERS,
and pulls the definition and synonym for them from SIDER based on the UMLS code IF
the side effect is attributed to the medication by clinical studies AND it is a primary or 
secondary suspect in the report.
"""

import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import numpy as np
import glob


# Defining conditions and how to screen for them (the condition in my language, and a word that uniquely IDs it in FAERs)
condition = ['ADHD', 'Anxiety','Bipolar-Disorder','Depression', 'Schizophrenia']
condition_filter = ['Attention', 'Anxiety','Bipolar','Depression', 'Schizophrenia']

def uniqSEs_from_medfile(medfile, condfilter):
    """
    Finds all the unique side effects for a given medication based on FAERs reports.

    Input:
    ------
    medfile = The csv containing all the reports for the medication, created with 
              MakeMedsTables.py
    condfilter = The word uniquely identifying the condition in the indication in the report

    Output:
    -------
    uniq_SEs = An array of unique side effects (MEDDRA term)
    """
    
    # Grabbing the data and making sure it only includes the condition we care about
    df = pd.read_csv(medfile, sep='$', index_col=0)
    df = df.dropna(subset=['indi_pt'])
    df = df[[indi.find(condfilter) != -1 for indi in df['indi_pt']]]

    # Only grabbing data when the drug described is the primary or secondary suspect
    df = df[[role.find('C') == -1 for role in df['role_cod']]]

    # Finding all the unique side effects mentioned
    if any(df):
        SEs = []
        for pt in df['pt']:
            SEs += pt.split(', ')
        uniq_SEs = np.unique(SEs)

        return uniq_SEs
    else:
        return []


def findUMLSCodes_forUniqSEs(uniq_SEs, condition, medication, path='./'):
    """
    Takes the list of unique side effects created with uniqSEs_from_medfile, 
    and associates them with a UMLS code (which makes it easy to scrape them from 
    SIDER).

    Input:
    ------
    uniq_SEs = An array of MEDDRA terms for side effects
    condition = The condition being treated
    medication = The associated medication

    Output:
    -------
    Writes a file with the side effects that were found in the UMLS-MEDDRA term matching tsv file,
    and a file with all the "side effects" (usually notes from the Dr.) that were not
    """

    # Reading in the file that ties side effects to codes for searching SIDER
    df = pd.read_csv(path+'meddra_all_se.tsv', sep='\t')

    # Grabbing only the MedDRA terms
    df = df[df['Type'] == 'PT']

    # Cleaning up the duplicates from the side effect names (may not work with webscrape, we'll see)
    df = df.drop_duplicates(subset=['SE Name'])

    # Splitting the side effects into those that were located, and those that were not
    not_found = []
    found = []
    for SE in uniq_SEs:
        if df['SE Name'].eq(SE).sum():
            conceptID = df[df['SE Name'].eq(SE)]['Concept ID'].values
            found.append({'Side Effect':SE, 'Concept ID': conceptID[0]})
        # This covers issues with providers entering in more human descriptions
        elif SE.lower().find('increase') != -1 or SE.lower().find('decrease') != -1 or SE.lower().find('abnormal') != -1:
            found.append({'Side Effect':SE, 'Concept ID': '0'})
        else:
            not_found.append(SE)

    # Writing a csv with the side effects that were found
    foundDF = pd.DataFrame(found)
    foundDF.to_csv(path+'faers_results/{:s}/SideEffects_Found_{:s}.csv'.format(condition, medication), sep='$')
            
    # Saving those that were not found so that I can check by hand as necessary
    np.savetxt(path+'faers_results/{:s}/SideEffects_NotFound_{:s}.csv'.format(condition, medication), not_found, fmt='%s')


def conglommerate_SEs_and_meds(condition):
    """
    For every medication for which side effects were found, this function groups
    the results by condition and then pivots the table so that every concept ID
    is associated with a unique side effect and a list of medications.
    """
    # Gathering files and medications
    files = glob.glob('faers_results/{:s}/SideEffects_Found_*.csv'.format(condition))
    medications = [f[f.find('Found_')+6:f.find('.csv')] for f in files]

    # Creating one monster dataframe that will become a pivot table
    for i,f in enumerate(files):
        if i == 0:
            masterdf = pd.read_csv(f,sep='$')
            masterdf['Medication'] = [medications[i]]*len(masterdf)
        else:
            df = pd.read_csv(f,sep='$')
            df['Medication'] = [medications[i]]*len(df)
            masterdf = masterdf.append(df, ignore_index=True,sort=False)

    # Pivoting the dataframe
    # The last agg func is taking all the medications associated with that side effect,
    # finding the unique keys, and joining them into a list
    masterPV = masterdf.pivot_table(index='Concept ID', values=['Side Effect', 'Medication'],
                                    aggfunc={'Side Effect': lambda x: np.unique(x)[0],
                                             'Medication':lambda x: ', '.join(np.unique(x))})

    masterPV.to_csv('faers_results/{:s}/SideEffectsJoined.csv'.format(condition),sep='$')
    

def searchSIDER(conceptID, meds4compare, rootUrl='http://sideeffects.embl.de/se/'):
    """
    This function takes in an UMLS concept ID, scrapes the defintion and synonyms for a side
    effect associated with it, and checks to see if the medication is in the list of medications
    associated with the concept ID, if it is, all the info is returned.
    """
    # Getting the query page html
    response = requests.get(rootUrl+conceptID+'/')
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Pulling out info on the condition
    container = soup.find_all('div', attrs={'class':'container'})[2]

    founddef = False  ;  foundsyn = False
    for strong_tag in container.find_all('strong'):
        if strong_tag.text == 'Definition':
            definition = strong_tag.next_sibling
            definition = definition[definition.find(': ')+2:]
            founddef = True
        elif strong_tag.text == 'Synonyms':
            synonyms = strong_tag.next_sibling
            synonyms = synonyms[synonyms.find(': ')+2:synonyms.find('\n')]
            foundsyn = True
            
    # pulling out info on the relevant medications
    meds = []
    links = []
    
    drugsWSE = soup.find('table', attrs={'class':'seDetailTable'})
    col = drugsWSE.find_all('td')
    found = []
    if col:
        col = col[0]
        for bullet in col.find_all('li'):
            link = str(bullet.find('a', href=True))
            medication = link[link.rfind('">')+2:link.find('</a>')]
            link = link[link.find('href="')+6:link.find('/"')]
            
            meds.append(medication.lower().strip())
            links.append(link)

        # Checking to see if medications represented in that side effect list
        for med in meds4compare:
            if med.lower().strip() in meds:
                found.append(links[meds.index(med.lower().strip())])
            else:
                found.append('')

    if founddef and foundsyn:
        return definition, synonyms, found
    elif foundsyn:
        return '', synonyms, found
    elif founddef:
        return definition, '', found
            

def find_percentage_occurrence(medication, medlink, conceptIDs,
                               rootUrl='http://sideeffects.embl.de/'):
    """
    This function then goes to the medication page on SIDER, and if possible, collects
    the side effect occurence frequency information from SIDER.
    """
    
    # Finding the frequency with which those side effects show up for found medications
    percentage = []
    # Getting the query page html
    response_med = requests.get(rootUrl+medlink)
    newsoup = BeautifulSoup(response_med.content, 'html.parser')

    for row in newsoup.find_all('tr')[2:]:
        checkme=False
        for i,item in enumerate(row.find_all('td')[:2]):
            if i == 0:
                link = str(item.find('a', href=True))
                concept = link[link.find('"/se/')+5:link.find('/" title')]
                if concept in conceptIDs:
                    checkme=True
            elif i == 1 and checkme:
                percentage.append((concept, item.text.replace('\n','')))
            
    return percentage
    

def workWConglommerate(condition):
    """
    This function goes through the dataframe of all the side effects for a given condition that 
    was created with conglommerate_SEs_and_meds, and scrapes SIDER for info on each one, for a 
    given condition
    """
    
    # Read in the file of all the medications and conceptIDs
    master_df = pd.read_csv('faers_results/{:s}/SideEffectsJoined.csv'.format(condition),sep='$')

    # Iterate over it
    all_findings = []
    all_defs = []
    all_syns = []
    uniq_meds = {}
    for ID, se, meds in zip(master_df['Concept ID'], master_df['Side Effect'], master_df['Medication']):
        medsList = meds.split(', ')
        if ID != '0':
            print(ID)
            defn, syn, findings = searchSIDER(ID, medsList)
            time.sleep(0.025)  # Waiting so as not to spam the site

            # Getting all the concept ids for each medication that have been found in the side effects
            relMeds = [medsList[i].strip().lower() for i, link in enumerate(findings) if link]
            relLinks = [link for link in findings if link]

            for med,link in zip(relMeds, relLinks):
                if med not in uniq_meds:
                    uniq_meds[med] = {'Link': link, 'Concept ID': [ID]}
                else:
                    uniq_meds[med]['Concept ID'].append(ID)

            all_defs.append(defn)
            all_syns.append(syn)
            all_findings.append(findings)
        else:
            print('Check: '+se+'\t'+meds)
            all_defs.append('')
            all_syns.append('')
            all_findings.append('')
            
    # Gather everything into the dataframe
    master_df['Definition'] = all_defs
    master_df['Synonyms'] = all_syns
    master_df['Medications observed'] = ['']*len(master_df) # Placeholder
    master_df['Percentage observed'] = ['']*len(master_df) # Placeholder

    master_df.to_csv('faers_results/{:s}/SideEffectsHalfExtracted.csv'.format(condition),sep='$')
    
    # Working with the dictionary I made
    if uniq_meds:
        copy = master_df[['Concept ID', 'Medications observed', 'Percentage observed']]
        for med in uniq_meds:
            medlink = uniq_meds[med]['Link']
            IDs = uniq_meds[med]['Concept ID']
            uniq_meds[med]['Percentages'] = find_percentage_occurrence(med, medlink, IDs)
            time.sleep(0.1) # Waiting so as not to spam the site

            for ID, perc in uniq_meds[med]['Percentages']:
                ind = copy[copy['Concept ID']==ID].index[0]
                copy.loc[ind]['Medications observed'] += med+', '
                copy.loc[ind]['Percentage observed'] += perc+', '

        # Cutting off trailing commas
        master_df['Medications observed'] = [obs[:-2] for obs in copy['Medications observed']]
        master_df['Percentage observed'] = [obs[:-2] for obs in copy['Percentage observed']]
        
    master_df = master_df.drop(columns=['Medication'])

    # Saving 
    master_df.to_csv('faers_results/{:s}/SideEffectsExtracted.csv'.format(condition),sep='$')

        
# Goes through every condition and scrapes SIDER
for cond, condfilter in zip(condition, condition_filter):
    files = glob.glob('faers_results/{:s}/faers_pull_*csv'.format(cond))
    for f in files:
        medication = f[f.find('pull_')+5:f.find('.csv')]
        print(medication)

        if not glob.glob('faers_results/{:s}/SideEffects_Found_{:s}.csv'.format(cond,medication)):
            uSEs = uniqSEs_from_medfile(f, condfilter)
            if any(uSEs):
                findUMLSCodes_forUniqSEs(uSEs, cond, medication)

    if not glob.glob('faers_results/{:s}/SideEffectsExtracted.csv'.format(cond)):
        conglommerate_SEs_and_meds(cond)
        workWConglommerate(cond)



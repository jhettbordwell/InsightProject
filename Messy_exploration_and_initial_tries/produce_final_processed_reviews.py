import pandas as pd
import numpy as np
import glob

import nltk
from nltk.sentiment import vader
VADER_SIA = vader.SentimentIntensityAnalyzer()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import spacy
from spacy.tokenizer import Tokenizer
import en_core_web_lg
nlp = en_core_web_lg.load()
tokenizer = Tokenizer(nlp.vocab)

from autocorrect import Speller
spell = Speller(lang='en')





# Magic tokenizer thing
def spacyTokenizer(s: str)-> list:
    doc = tokenizer(s.lower().strip())
    tokens = []
    for token in doc:
        if not token.is_stop and token.is_alpha and token.lemma_ != '-PRON-':
            tokens.append(token.lemma_)
        
    return tokens


# Gathering top unique words in both reviews and side effects because it greatly increases fidelity of matching
def findTop(strList, keeptop=10):
    tfidf_vectr = TfidfVectorizer()
    corpus = [' '.join(SE) for SE in strList]
    tfidf_score = tfidf_vectr.fit_transform(corpus).toarray()
    features = np.array(tfidf_vectr.get_feature_names())
    
    words = []
    for row in tfidf_score:
        inds = row.argsort()[::-1][:keeptop]
        row_words = []
        for ind in inds:
            if row[ind].round(2) != 0:
                row_words.append(features[ind])

        if not row_words:
            row_words = list(features[inds][:5])
        words.append(row_words)

    return words




# Parse files
def parseRevnew(file, return_df=False):
    df = pd.read_csv(file, sep='$')
    reviews = df['Comment']
    clean_reviews = [spacyTokenizer(rev.replace('/', ' ')) for rev in reviews]
    cleaner_reviews = findTop(clean_reviews, 50)
    cleaner_reviews = [[spell(word.lower()) for word in rev] for rev in cleaner_reviews]
    
    if return_df:
        return cleaner_reviews, df
    else:
        return cleaner_reviews#consider, ignore

def parseSEorig(file):
    sideEff = np.genfromtxt(file, delimiter='$', dtype=str)
    clean_SEs = [[spell(word) for word in spacyTokenizer(SE)] for SE in sideEff]
    cleaner_reviews = findTop(clean_SEs, 3)
    
    return clean_SEs
    
def parseSE_FAERs(file, meds):
    sideEff = pd.read_csv(file, sep='$').set_index('Concept ID').dropna(subset=['Percentage observed'])
    sideEff = sideEff.fillna(value='')

    # Looking for the medication name that has the side effects
    meds_obs = sideEff.copy(deep=True)
    meds_obs['Medications observed'] = [obs.split(', ') for obs in meds_obs['Medications observed']]
    
    medList = []
    for obs in meds_obs['Medications observed']: medList += obs
        
    to_check = np.unique(medList)
    
    Found = False
    for med in meds.lower().split(', '):
        if med in to_check: 
            Found = True
            break # Stop when I've found the name

    sideEff = sideEff[[med in obs for obs in meds_obs['Medications observed']]]
    
    sideEff['Joined'] = sideEff['Definition'] + sideEff['Synonyms']
    check_both = lambda combo: sum([c in nlp.vocab for c in combo.split(' ')]) == len(combo.split(' '))
    sideEff['Joined'] = [', '.join([word for word in words.split(', ') if word.find('-') == -1 and check_both(word)]) for words in sideEff['Joined']]
    clean_SEs = [list(set([spell(word) for word in spacyTokenizer(SE)])) for SE in sideEff['Joined']]
    clean_SEs = [[word for word in SE if len(word) > 3] for SE in clean_SEs]

    if clean_SEs:
        clean_SEs = findTop(clean_SEs,5)
    
    return clean_SEs





# Performing sentiment analysis
def find_polarity_scores(reviews):
    VADERscorePos = []
    VADERscoreNeg = []
    for rev in reviews:
        VADERscorePos.append(VADER_SIA.polarity_scores(rev)['pos'])    
        VADERscoreNeg.append(VADER_SIA.polarity_scores(rev)['neg'])            
        
    return VADERscorePos, VADERscoreNeg





# Locating side effects
def find_sideEffects_inReviews_FAERsinformed(revFile, sefile1, sefile2, faers=True):

    # Parsing reviews
    reviews, fullrevs = parseRevnew(revFile, return_df=True)
    if faers:
        cond = sefile1[sefile1.find('faers_results/')+14:sefile1.rfind('/')]
        medication = revFile[revFile.rfind('/')+1:revFile.find('_'+cond)]
        allmeds = pd.read_csv('UniqueMedications/Medications_unique_{:s}.csv'.format(cond), sep='$')['All names']
        meds = [allnames for allnames in allmeds if medication.lower() in [med.strip() for med in allnames.lower().split(', ')]]

        if not meds:
            return False, False, True
        else:
            meds = meds[0]

    # Parsing side effects
    listSEs1 = parseSE_FAERs(sefile1, meds)
    listSEs2 = parseSEorig(sefile2)
    listSEs = listSEs1 + listSEs2
    
    BagOSE = ' '.join([' '.join(SE) for SE in listSEs])

    # Finding review words that exist in the list of side effects
    # Only requiring space at the beginning because of words like nausea-nauseated, etc.
    found = [[word for word in rev if BagOSE.lower().find(' '+word.lower())] for rev in reviews]
    found = []
    for ind, rev in enumerate(reviews):
        item = {}
        for SE in listSEs:
            # Match words in reviews to side effects and then add them to found, build dataframe with this info
            item[', '.join(SE)] = len([word for word in rev if word.lower() in SE])
        found.append(item)
    
    SE_match = pd.DataFrame(found)
    SE_match['Full Review'] = fullrevs['Comment']
    
    # Return the master product
    return SE_match, fullrevs, False




# Screening for reviews that have been found
def screen_for_hits(df):
    newdf = df.drop(columns=['Full Review', 'Positive polarity', 'Negative polarity'])
    review_inds = []
    
    # Allowing for two item side effects UNLESS they contain very generic words
    colLens = np.array([len(col.split(', ')) + 2*((col.find('medication')!=-1)|
                                                  (col.find('reaction')!=-1)|
                                                  (col.find('skin')!=-1)|
                                                  (col.find('feel') != -1)|
                                                  (col.find('pain')!= -1)|
                                                  (col.find('abnormal')!=-1)|
                                                  (col.find('change')!=-1)|
                                                  (col.find('disorder')!=-1)|
                                                  (col.find('problem')!=-1)|
                                                  (col.find('decrease')!=-1)|
                                                  (col.find('increase')!=-1)|
                                                  (col.find('loss')!=-1)) for col in newdf.columns])
    
    # If the column is not generic and has two or fewer words, count one word as a match, otherwise require 2
    for ind in newdf.index:
        if (((colLens < 3) & newdf.loc[ind].gt(0)) | newdf.loc[ind].gt(1)).sum(): 
            review_inds.append(ind)
            
    # Screening based on polarity
    cond = (df['Negative polarity'] != 0)
    diff_inds = df.index[cond][(df['Positive polarity'][cond]/df['Negative polarity'][cond] > 2.5)]

    # Don't know why this is necessary

    
    # Marking hits versus not
    found_reviews = []
    for ind in review_inds:
        conditions = np.logical_or(np.logical_and((colLens < 3), newdf.loc[ind].gt(0)), newdf.loc[ind].gt(1))
        newdf.loc[ind][conditions] = 1
        newdf.loc[ind][np.logical_not(conditions)] = 0
        found_reviews.append(sum(conditions))
    
    # Creating one big dataframe
    toconcat = df.drop(columns=[col for col in df.columns if col not in ['Full Review', 'Positive polarity', 'Negative polarity']])
    master_df = pd.concat([toconcat, newdf], axis=1)
    
    # Not dropping rows with no hits because that is also important infomration
    return master_df





# Sticking it all together

for condition in ['ADHD', 'Anxiety', 'Bipolar-Disorder', 'Depression', 'Schizophrenia']:

    if not glob.glob('Final_processed_reviews/{:s}_processed.csv'.format(condition)):
        medications = pd.read_csv('UniqueMedications/Medications_unique_{:s}.csv'.format(condition), sep='$')['Medication']
        Ive_made_a_monster=False
        for i, med in enumerate(medications):
            # Locate the side effects
            print(condition, med)
            df, fullrevs, failed = find_sideEffects_inReviews_FAERsinformed('ProcessedReviews/{:s}/{:s}_{:s}_parsed_reviews.csv'.format(condition, med.strip(), condition), 
                                                                            'NERstuff/faers_results/{:s}/SideEffectsExtracted.csv'.format(condition),
                                                                            'SideEffects/{:s}_SideEffects.csv'.format(condition))


            if not failed:
                
                # Perform a sentiment analysis of the full reviews
                pos, neg = find_polarity_scores(df['Full Review'])
                df['Positive polarity'] = pos
                df['Negative polarity'] = neg


                # Screen the dataframe for reviews
                master_df = screen_for_hits(df)
                master_df[med.strip()] = 1

                for col in ['Effectiveness', 'Satisfaction', 'Age', 'Gender', 'Length of treatment']:
                    master_df[col] = fullrevs[col].copy(deep=True)
        
        
                # Concatenate
                if i == 0 or not Ive_made_a_monster:
                    monster_df = master_df.copy(deep=True)
                    Ive_made_a_monster=True
                else:
                    monster_df = monster_df.append(master_df, ignore_index=True, sort=False)
                    monster_df = monster_df.fillna(0)


        monster_df.to_csv('Final_processed_reviews/{:s}_processed.csv'.format(condition), sep='$')

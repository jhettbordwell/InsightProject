import pandas as pd
import numpy as np
import glob as glob

import nltk
from nltk.corpus import wordnet

import spacy
from spacy.tokenizer import Tokenizer
import en_core_web_lg
nlp = en_core_web_lg.load()

tokenizer = Tokenizer(nlp.vocab)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from autocorrect import Speller
spell = Speller(lang='en')

from nltk.sentiment import vader
VADER_SIA = vader.SentimentIntensityAnalyzer()


class FindSideEffects:
    def __init__(self):
        pass

    # Magic tokenizer thing
    def spacyTokenizer(self, s):
        doc = tokenizer(s.lower().strip())
        tokens = []
        for token in doc:
            if not token.is_stop and token.is_alpha and token.lemma_ != '-PRON-':
                tokens.append(token.lemma_)
        
        return tokens

    def find_polarity_scores(self,reviews):
        VADERscorePos = []
        VADERscoreNeg = []
        for rev in reviews:
            VADERscorePos.append(VADER_SIA.polarity_scores(rev)['pos'])    
            VADERscoreNeg.append(VADER_SIA.polarity_scores(rev)['neg'])            
        
        return VADERscorePos, VADERscoreNeg

    # Need to collect FAERs results and basic Drugs.com info
    def processSideEffects(self, condition, medFile='MedicationsAndSideEffects.csv'):
    
        # Processing the Drugs.com side effects
        df = pd.read_csv(medFile, sep='$', index_col=0)
        df = df.loc[df[df['Condition'].eq(condition)].index]
        readLine = lambda s: s[2:-2].split('; ')[:-1]
        allSEs = []
        for key in ['More common', 'Less common', 'Incidence not known']:
            for item in [readLine(SE) for SE in df[key]]: allSEs += item
        allSEs = list(np.unique(allSEs))
            
        allSEs = [[spell(word) for word in self.spacyTokenizer(SE)] for SE in allSEs]
    
    
        # Handling the FAERs side effects
        moreSEs = []
        mfile = pd.read_csv('faers_results/{:s}/SideEffectsExtracted.csv'.format(condition),
                            sep='$', index_col=0)
        mfile = mfile.fillna('')
        moreSEs += [mfile.loc[ind]['Definition']+mfile.loc[ind]['Synonyms'] for ind in mfile.index]
        moreSEs = list(np.unique(moreSEs))
        moreSEs = [[spell(word) for word in self.spacyTokenizer(SE)] for SE in moreSEs]

        return allSEs+moreSEs
    
    
    def findTop(self, strList, keeptop=10, extracut=5, topcutoff=0.05, vocab=None):
        tfidf_vectr = TfidfVectorizer(vocabulary=vocab)
            
        corpus = [' '.join(SE) for SE in strList]
        tfidf_score = tfidf_vectr.fit_transform(corpus).toarray()
        features = np.array(tfidf_vectr.get_feature_names())
    
        words = []
        for row in tfidf_score:
            topcut = min([keeptop,len(row)])
            inds = row.argsort()[::-1][:topcut]
            row_words = []
            for ind in inds:
                if row[ind] >= topcutoff:
                    row_words.append(features[ind])
                    
            if not row_words:
                row_words = list(features[inds][:extracut])
            words.append(row_words)

        return words

    def parseRevsbyCond(self, condition, cut=None):
        # Grab review files and associated medications
        reviews = glob.glob('ProcessedReviews/{:s}/clean_*_reviews.csv'.format(condition))
        reviews = np.unique(reviews) # Sorting
        medications = [rev[rev.find('clean_')+6:rev.rfind('_reviews.csv')] for rev in reviews]

        # Stack the reviews on top of one another and create a column for medications
        medstack = []
        if cut:
            c1, c2 = cut
        else:
            c1, c2 = 0, len(reviews)
            
        for i,revFile in enumerate(reviews[c1:c2]):
            df = pd.read_csv(revFile, sep='$', index_col=0)
            df = df[['Full Review', 'Effectiveness', 'Satisfaction']]
            medstack += [medications[i]]*len(df)
            if i == 0:
                master_df = df.copy()
            else:
                master_df = master_df.append(df,ignore_index=True, sort=False)

        master_df = master_df.reset_index()
        master_df['Medication'] = medstack
        return master_df

    def parseRevAndSEs(self, SEvocab, condition,
                       top_cutoff=0.05,
                       topRev=25, topRevextra=10,
                       topSE=3, topSEextra=3, cut=None):
        
        df = self.parseRevsbyCond(condition, cut=cut)
        reviews = df['Full Review']
        print("I've processed the reviews")
        
        # Counting medication mentions as a feature
        medications = np.unique(df['Medication'])
        medications = np.array([med.lower().replace('-',' ') for med in medications])
        med_counts = []
        for rev in df['Full Review']:
            split_rev = rev.split(' ')
            matches = []
            for med in medications:
                medSplit = list(med)
                lenMed = len(medSplit)
                for word in split_rev:
                    wordSplit = list(word.lower())
                    ind = min([lenMed, len(wordSplit)])
                    test = ''.join([w for i,w in enumerate(wordSplit[:ind]) if w == medSplit[i] ])
                    if len(test) >= max([ind,len(medSplit)-2]):
                        matches.append(med)
            med_counts.append(np.unique(matches).size)
    
        pos, neg = self.find_polarity_scores(reviews)
        reviews = [[spell(word) for word in self.spacyTokenizer(rev.replace('/', ' '))] for rev in reviews]
        
        clean_SEs = self.findTop(SEvocab, keeptop=topSE, extracut=topSEextra)

        vocab = []
        for SE in clean_SEs: vocab += SE
        vocab = np.unique(vocab)
        clean_reviews = self.findTop(reviews, keeptop=topRev, extracut=topRevextra,
                                     vocab=vocab)

        
        return list(df['Full Review']), clean_reviews, clean_SEs, med_counts, list(df['Effectiveness'])


    def find_sideEffects_inReviews_FAERsinformed(self, SEvocab, condition, cut=None):

        fullrevs, reviews, listSEs, med_counts, eff = self.parseRevAndSEs(SEvocab, condition,
                                                                          cut=cut)
    
        BagOSE = ' '.join([' '.join(SE) for SE in listSEs])

        # Finding review words that exist in the list of side effects
        # Only requiring space at the beginning because of words like nausea-nauseated, etc.
        found = [[word for word in rev if BagOSE.lower().find(' '+word.lower())] for rev in reviews]
        found = []
        for ind, rev in enumerate(reviews):
            item = {}
            for SE in listSEs:
                # Match words in reviews to side effects and then add them to found,
                # build dataframe with this info
                item[', '.join(SE)] = len([word for word in rev if word.lower() in SE])
            found.append(item)
    
        SE_match = pd.DataFrame(found)
        SE_match['Full Review'] = fullrevs
        SE_match['Medication mentions'] = med_counts

        pos, neg = self.find_polarity_scores(fullrevs)
        SE_match['Positive polarity'] = pos
        SE_match['Negative polarity'] = neg

        SE_match['Effectiveness'] = eff
        
        # Return the master product
        return SE_match

    def screen_for_hits(self, df, posnegRat=6):
        newdf = df.drop(columns=['Full Review', 'Medication mentions',
                                 'Positive polarity', 'Negative polarity', 'Effectiveness'])
        review_inds = []
    
        # Allowing for two item side effects UNLESS they contain very generic words
        colLens = np.array([len(col.split(', ')) + 2*((col.find('skin')!=-1)|
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
        diff_inds = df.index[cond][(df['Positive polarity'][cond]/df['Negative polarity'][cond] > posnegRat)]
    
        # Marking hits versus not
        for ind in review_inds:
            if ind not in diff_inds and df.loc[ind]['Medication mentions'] < 2:
                conditions = np.logical_or(np.logical_and((colLens < 3), newdf.loc[ind].gt(0)), newdf.loc[ind].gt(1))
                newdf.loc[ind] = 0
                newdf.loc[ind][conditions] = 1
            else:
                newdf.loc[ind] = 0

        # Creating one big dataframe
        toconcat = df[['Full Review', 'Positive polarity', 'Negative polarity',
                       'Medication mentions','Effectiveness']]
        master_df = pd.concat([toconcat, newdf], axis=1)
    
        # Not dropping rows with no hits because that is also important information
        return master_df

    def mymethod(self, condition, cut=None):
        SEvocab = self.processSideEffects(condition)
        print("I've processed the side effects")
        
        df = self.find_sideEffects_inReviews_FAERsinformed(SEvocab, condition, cut=cut)
        print("I've found side effects")
        
        master = self.screen_for_hits(df)
        print("I've matched the side effects")

        if cut:
            master.to_csv('ReviewsMatched2SideEffects/{:s}_matched_{:g}_{:g}.csv'.format(condition, cut[0], cut[1]), sep='$')
        else:
            master.to_csv('ReviewsMatched2SideEffects/{:s}_matched.csv'.format(condition), sep='$')
        print("I've saved the results\n\n")

if __name__=="__main__":
    clf = FindSideEffects()
    #for condition in ['ADHD', 'Anxiety', 'Bipolar-Disorder', 'Depression', 'Schizophrenia']:
    for condition in ['Depression']:
        print("I'm working on {:s}".format(condition))
        for cut in [[0,11],[11,22],[22,37]]:
            clf.mymethod(condition, cut=cut)


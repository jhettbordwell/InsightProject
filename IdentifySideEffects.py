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

    def spacyTokenizer(self, s):
        """
        Tokenizing a document s using an english vocab with more breadth
        """
        doc = tokenizer(s.lower().strip())
        tokens = []
        for token in doc:
            if not token.is_stop and token.is_alpha and token.lemma_ != '-PRON-':
                tokens.append(token.lemma_)
        
        return tokens

    def find_polarity_scores(self,reviews):
        """
        Performs VADER sentiment analysis on a list of reviews in unedited form 
        (all capitalization, punctuation, etc. left in place)
        """
        VADERscorePos = []
        VADERscoreNeg = []
        for rev in reviews:
            VADERscorePos.append(VADER_SIA.polarity_scores(rev)['pos'])    
            VADERscoreNeg.append(VADER_SIA.polarity_scores(rev)['neg'])            
        
        return VADERscorePos, VADERscoreNeg

    # Need to collect FAERs results and basic Drugs.com info
    def processSideEffects(self, condition, medFile='MedicationsAndSideEffects.csv'):
        """
        For each condition, processes the side effects from Drugs.com and those that 
        were mentioned in FAERs reports. medFile takes default value from the csv 
        generated by ScrapeDrugsCom.py.
        """

        # Processing the Drugs.com side effects
        #---------------------------------------
        # Treating all side effects equally, mostly just hoping for more low level
        # descriptions I can incorporate for each medication
        df = pd.read_csv(medFile, sep='$', index_col=0)
        df = df.loc[df[df['Condition'].eq(condition)].index]
        readLine = lambda s: s[2:-2].split('; ')[:-1]
        allSEs = []
        for key in ['More common', 'Less common', 'Incidence not known']:
            for item in [readLine(SE) for SE in df[key]]: allSEs += item
        allSEs = list(np.unique(allSEs))
            
        allSEs = [[spell(word) for word in self.spacyTokenizer(SE)] for SE in allSEs]
    
    
        # Handling the FAERs side effects
        #---------------------------------
        # In short, these side effects are side effects that were mentioned in FAERs reports
        # and then verified on SIDER. I chose this method for two reasons. The first is that
        # initially I was considering including frequency information from the FAERs reports
        # in my results (and comparing it against the frequency info on SIDER). The second is
        # that this struck a nice balance in including verified side effects (they had been
        # reported by a doctor usually on FAERs), rather than the wide variety of things that
        # were present on SIDER (which included side effects like alcohol abuse)
        moreSEs = []
        mfile = pd.read_csv('faers_results/{:s}/SideEffectsExtracted.csv'.format(condition),
                            sep='$', index_col=0)
        mfile = mfile.fillna('')
        moreSEs += [mfile.loc[ind]['Definition']+mfile.loc[ind]['Synonyms'] for ind in mfile.index]
        moreSEs = list(np.unique(moreSEs))
        moreSEs = [[spell(word) for word in self.spacyTokenizer(SE)] for SE in moreSEs]

        return allSEs+moreSEs
    
    
    def findTop(self, strList, keeptop=10, extracut=5, topcutoff=0.05, vocab=None):
        """
        Finds the words with the highest TF-IDF scores in every document in a corpus

        Input:
        ------
        strlist = A list of lists of strings, where the whole list is a corpus, and each 
        nested list is a document.

        Optional Keywords (hyperparameters for the SE detection algorithm):
        -------------------------------------------------------------------
        keeptop = The number of words to keep
        extracut = The number of words to keep if no word is above the threshold
        topcutoff = The threshold words must be above to be initially kept
        vocab = A vocabulary to use to score the corpus

        Output:
        -------
        A list of the same size as strlist, with each nested list reduced to the top 
        words as defined by the optional keywords for this function.
        """

        # Generating the TF-IDF scorer and scoring the corpus
        tfidf_vectr = TfidfVectorizer(vocabulary=vocab)
            
        corpus = [' '.join(SE) for SE in strList]
        tfidf_score = tfidf_vectr.fit_transform(corpus).toarray()
        features = np.array(tfidf_vectr.get_feature_names())

        # Cutting down to the top words according to the keywords for this function
        words = []
        for row in tfidf_score:
            # First cutting down based on the number of words to keep
            topcut = min([keeptop,len(row)])
            inds = row.argsort()[::-1][:topcut]
            row_words = []
            for ind in inds:
                # If the word is above the threshold, keep it
                if row[ind] >= topcutoff:
                    row_words.append(features[ind])

            # If no words are above the threshold, keep extracut words
            if not row_words:
                row_words = list(features[inds][:extracut])
            words.append(row_words)

        # Each document in the corpus now has between extracut and keeptop significant words
        return words

    
    def parseRevsbyCond(self, condition, cut=None):
        """
        For each condition, this function creates a dataframe with the full review, effectiveness,
        satisfaction, and medication. 

        Optional Keywords:
        ------------------
        cut = Depression medications had so many reviews that I needed to cut down their number
              otherwise this function would time out. In future iterations, I need to implement
              this in a less hacked-together fashion.
        """
        
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
            medstack += [medications[c1+i]]*len(df)
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
        """
        For each full review, this function goes through and:
            1. Counts how many medications were mentioned
            2. Scores the positive and negative expression in the text
            3. Tokenizes the review
            4. Cuts down to the rarest words in the review using FindTop and the FindTop-ed side 
               effects as a vocabulary (which filters all non-side effect words from the review).

        Input:
        ------
        SEvocab = the parsed side effect information from ProcessSideEffects
        """
        
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

        # Performing sentiment analysis
        pos, neg = self.find_polarity_scores(reviews)
        reviews = [[spell(word) for word in self.spacyTokenizer(rev.replace('/', ' '))] for rev in reviews]

        # Cutting down the number of words in the side effects
        clean_SEs = self.findTop(SEvocab, keeptop=topSE, extracut=topSEextra)

        # Cutting down the number of words in the reviews, using the side effect vocabulary
        vocab = []
        for SE in clean_SEs: vocab += SE
        vocab = np.unique(vocab)
        clean_reviews = self.findTop(reviews, keeptop=topRev, extracut=topRevextra,
                                     vocab=vocab)

        
        return list(df['Full Review']), clean_reviews, clean_SEs, med_counts, list(df['Effectiveness']), list(df['Medication'])


    def find_sideEffects_inReviews_FAERsinformed(self, SEvocab, condition, cut=None):
        """
        Uses a bag of words approach to identify whether side effect words showed up in a review.
        FAERsinformed implies that side effects found in FAERs reports were used as well as Drugs.com
        sourced side effects (using both was found to have the highest accuracy in my experience).

        Input:
        ------
        SEvocab = A corpus of side effects (list of lists, nested lists = tokenized side effects)
        condition = The psychiatric condition being considered.
        cut = A split indicating the range of files to be considered for a given condition (from the alphabetical list),
              this parameter is included due to the massive size of the depression dataset, and is a hack solution.

        Output:
        -------
        SE_match = a dataframe with the basic matching information for each side effect and every review
        """
        
        # Going through and getting the minimal forms (all common words cut out) of the reviews and SEs
        fullrevs, reviews, listSEs, med_counts, eff, meds = self.parseRevAndSEs(SEvocab, condition,
                                                                                cut=cut)

        # Creating a string with every side effect word in it, so that I can search for partial words
        # as in the example on the next line
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

        # Constructing dataframe
        SE_match = pd.DataFrame(found)
        SE_match['Full Review'] = fullrevs
        SE_match['Medication mentions'] = med_counts

        pos, neg = self.find_polarity_scores(fullrevs)
        SE_match['Positive polarity'] = pos
        SE_match['Negative polarity'] = neg

        SE_match['Effectiveness'] = eff
        SE_match['Medication'] = meds
        
        # Return the master product
        return SE_match

    def screen_for_hits(self, df, posnegRat=6):
        """
        This function identifies whether or not an initial side effect match by 
        find_sideEffects_inReviews_FAERsinformed matches the criteria necessary for it to be 
        a truly informative match. These criteria are in their general form derived by
        evaluating the results by eye when I was initially considering the results of the 
        ancestors of find_sideEffects*, but as much as possible now are tuned by a grid 
        search (i.e., posnegRat).

        Initially, the ancestors of the identification function often mismatched because
        they latched on to some words that were especially common in side effects for 
        mental health medications (feel, disorder, decrease, etc.), and these words fit in
        an interesting spot between not rare enough to be significant on their own, but not
        common enough to be completely unnecessary to pull out side effects. To work with this
        interesting section of feature space, I designed this function to essentially treat 
        these "generic" words as 1/2 or less of a side effect, and every other word as 1/2 or 
        more depending on the length of the side effect.

        I further included the condition that only one medication could be mentioned, because
        the likelihood that a review was discussing side effects for another medication, or
        the reviewer couldn't distinguish the two because of a pharmaceutical regimen was 
        fairly high in my sampling of the reviews (~750-1000 reviews of 30,000, selected randomly).

        Finally, I found that the review's sentiment was an important final deciding call, because 
        a review that was extremely positive was usually only mentioning side effects in absence,
        as in, "I experienced no weight gain, it was wonderful!". I made the level of positivity
        a free variable and fit the default value using a grid search of a wide range of values.
        """

        # Isolating to only side effect columns
        newdf = df.drop(columns=['Full Review', 'Medication', 'Medication mentions',
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
        toconcat = df[['Medication', 'Full Review', 'Positive polarity', 'Negative polarity',
                       'Medication mentions','Effectiveness']]
        master_df = pd.concat([toconcat, newdf], axis=1)
    
        # Not dropping rows with no hits because that is also important information
        return master_df

    def mymethod(self, condition, cut=None):
        """
        Processes the side effects and reviews, identifies the side effects, confirms that the IDs are
        true identifications, and writes a file with that information summarized, for each condition.

        Optional keyword:
        -----------------
        cut = The range of files to include in the process (had to be split up for depression)
        """
        # Processing the side effects into a tokenized corpus
        SEvocab = self.processSideEffects(condition)
        print("I've processed the side effects")

        # Locating the side effect words within the processed reviews
        df = self.find_sideEffects_inReviews_FAERsinformed(SEvocab, condition, cut=cut, whoops=whoops)
        print("I've found side effects")

        # Screening those matches against some data-inspired conditions to verify that the matches are informative
        master = self.screen_for_hits(df)
        print("I've matched the side effects")

        # Saving the results
        if cut:
            master.to_csv('ReviewsMatched2SideEffects/{:s}_matched_{:g}_{:g}.csv'.format(condition, cut[0], cut[1]), sep='$')
        else:
            master.to_csv('ReviewsMatched2SideEffects/{:s}_matched.csv'.format(condition), sep='$')
            print("I've saved the results\n\n")

if __name__=="__main__":
    clf = FindSideEffects()
    for condition in ['ADHD', 'Anxiety', 'Bipolar-Disorder', 'Schizophrenia']:
        print("I'm working on {:s}".format(condition))
        clf.mymethod(condition)
        
    for condition in ['Depression']:
        print("I'm working on {:s}".format(condition))
        for cut in [[0,11],[11,22],[22,37]]:
            clf.mymethod(condition, cut=cut)


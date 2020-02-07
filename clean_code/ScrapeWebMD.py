import requests
from bs4 import BeautifulSoup, SoupStrainer
import time
import pandas as pd
import glob
import pdb
import h5py

class WebMDScraper:
    def __init__(self):
        self._readinMeds()
        self.found_medications = {}
        
    def _readinMeds(self, medfile='MedicationsAndSideEffects.csv'):
        self.medications = list(pd.read_csv(medfile, sep='$', index_col=0)['Medication'])

    # Next, we want to pull the detailed "Drug Results" pages from WebMD
    def searchWebMD(self, searchterm,
                    rootUrl='https://www.webmd.com/drugs/2/search?type=drugs&query='):

        self.not_found = {}
        
        # Getting the query page html
        if searchterm.find(' ') != -1:
            searchterm.replace(' ', '-')
        response = requests.get(rootUrl+searchterm)
        soup = BeautifulSoup(response.content, 'html.parser')
    
        # Checking if there are matches for the drug
        content = soup.find('p', attrs={'class':'no-matches'})
        if content:
            return False, False
        else:
            # Going through every match and appending them to lists to write to the metadata file
            searchLink = False
        
            content = soup.find('ul', attrs={'class':'exact-match'})
            if not content: content = soup # In case there is no 'exact match' section
            for link in content.find_all('a', href=True):
                if str(link).find('result_') != -1:
                    searchLink = str(link)
                    drug = searchLink[:searchLink.rfind('/details')]
                    drug = drug[drug.rfind('/')+1:]

                    searchLink = searchLink[searchLink.find('href="')+6:searchLink.rfind('"')]
                    searchLink = 'https://www.webmd.com'+searchLink

                    if searchLink:
                        self.found_medications[drug] = searchLink
                        print(drug, '\t', searchLink)
                else:
                    self.not_found[searchterm] = ''

    def handParseWebMD(self):
        for key in self.not_found:
            link = input(key)
            self.found_medications[key] = link
            
    def getResultsLinks(self):
        # Get the links to the medication pages
        for med in self.medications:
            self.searchWebMD(med)

        # Because some links aren't intuitive
        #self.handParseWebMD()
        # Dropping this because it only brought up one drug and broke something important

        with h5py.File('DrugLinks.h5','w') as F:
            for key in self.found_medications:
                F[key] = self.found_medications[key]

        
    def getReviewsLink(self,url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        content = soup.find('div', attrs={'class':'drug-review-lowest'})
        if content:
            searchLink = ''
            for child in content.findChildren():
                if str(child).find('"drug-review"') != -1:
                    searchLink = str(child)
                    searchLink = searchLink[searchLink.find('href="')+6:searchLink.rfind('">')]
                    searchLink += '&pageIndex=0&sortby=3&conditionFilter=-1' # grabs reviews for every condition
                    searchLink = 'https://www.webmd.com'+searchLink
                    break
            
            return searchLink
        else:
            return False

    def pull_review(self,userPost, revnum=1):
        info = {}
        
        # Pulling basic post info
        info['conditionInfo'] = userPost.find('div', attrs={'class':'conditionInfo'}).text.replace('\r\n\t\t\t\t\t','')
        info['date'] = userPost.find('div', attrs={'class':'date'}).text
        info['reviewer'] = userPost.find('p', attrs={'class':'reviewerInfo'}).text
    
        # Pulling stars info
        content = userPost.find('div', attrs={'id':'ctnStars'})
        catsAndScores = []
        for child in content.findChildren():
            nextisScore=False
            for grandChild in child.findChildren():
                if nextisScore:
                    score = grandChild.find('span', attrs={'class':'current-rating'}).text
                    score = score[score.rfind(' ')+1:]
                    catsAndScores.append((category, score))
                    nextisScore=False
                if str(grandChild).find('category') != -1:
                    category = str(grandChild)
                    category = category[category.find('gory">')+6:category.rfind('</p>')]
                    nextisScore=True
        catsAndScores = dict(catsAndScores)
        info['Effectiveness'] = catsAndScores['Effectiveness']
        info['Satisfaction'] = catsAndScores['Satisfaction']
        info['Ease of Use'] = catsAndScores['Ease of Use']
    
        # Pulling comment and cleaning it up
        text = userPost.find('p', attrs={'style':'display:none',
                                         'id':'comFull{:g}'.format(revnum)}).text
        text = text.replace('<strong>Comment:</strong>','').replace('<br','')
        text = text.replace('Hide Full Comment', '').replace('Comment:','')
        info['Full Review'] = text
    
        return info

    def proceed2NextPage(self,url):
        num = int(url[url.find('pageIndex=')+10:url.rfind('&sortby')])
        newurl = url.replace('pageIndex={:g}'.format(num),
                             'pageIndex={:g}'.format(num+1))
        return newurl

    def scrollReviews(self, reviewPage0, reviewsPerPage=5):
        # Going to the reviews page and finding the total number of reviews
        response = requests.get(reviewPage0)
        soup = BeautifulSoup(response.content, 'html.parser')

        content = soup.find('span', attrs={'class':'totalreviews'})
        if not content: # In case there are no reviews
            return None
        totalreviews = content.text
        totalreviews = totalreviews[:totalreviews.rfind('Total')-1]

        # Counting the number of pages
        totalreviews = int(totalreviews)
        num_pages = totalreviews // reviewsPerPage
        if totalreviews % reviewsPerPage: num_pages += 1
    
        # Iterating through pages and grabbing review data
        all_reviews = []
        for npage in range(num_pages):
            # Finding the relevant page
            if npage == 0:
                url = reviewPage0
            else:
                url = self.proceed2NextPage(url)
                
            # Get heading above userPosts
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.find('div', attrs={'id':'ratings_fmt'})
    
            revnum = 1
            for child in content.findChildren():
                if str(child).find('userPost') != -1:
                    child_info = self.pull_review(child, revnum=revnum)
                    all_reviews.append(child_info)
                    revnum += 1
            
            time.sleep(0.025)
        
        # Process list all_reviews into dataframe and return
        return pd.DataFrame(all_reviews)
        
        
    def _formatExactMed(self,s: str)->str:
        s = s.strip() # remove preceding and trailing white space
        s=s.replace(',','-') # Get rid of commas
        s=s.replace(' ', '-') # Get rid of white space
        s=s.replace('/', 'per') # remove things that make it look like a path
        return s
                        
    def pullReviewsByMed(self):
        # Scrapes WebMD and pulls review for every medication it can
        for key in self.found_medications:
            link = self.found_medications[key]
            drug = key
            
            not_found=False
            name = self._formatExactMed(drug)
            if not glob.glob('Reviews/{:s}_reviews.csv'.format(name)):
                reviewLink = self.getReviewsLink(link)
                if not reviewLink: not_found = True
            else:
                reviewLink = None

            if reviewLink and not not_found:
                print('Scrolling for:\n', reviewLink)
                reviewsDF = self.scrollReviews(reviewLink)
                if type(reviewsDF) != type(None):
                    reviewsDF.to_csv('Reviews/{:s}_reviews.csv'.format(name), sep='$')
            elif not_found:
                reviewLink = input(link)
                print('Scrolling for:\n', reviewLink)
                reviewsDF = self.scrollReviews(reviewLink)
                if type(reviewsDF) != type(None):
                    reviewsDF.to_csv('Reviews/{:s}_reviews.csv'.format(name), sep='$')
                    
if __name__ == '__main__':
    Scraper = WebMDScraper()

    # Grab the medication page for each medication (hand tuning as necessary)
    if glob.glob('DrugLinks.h5'):
        with h5py.File('DrugLinks.h5','r') as F:
            for key in F:
                Scraper.found_medications[key] = F[key][()]
    else:
        Scraper.getResultsLinks()

    # Grab the reviews for each medication
    Scraper.pullReviewsByMed()

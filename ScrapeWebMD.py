import requests
from bs4 import BeautifulSoup, SoupStrainer
import time
import pandas as pd
import glob
import pdb
import h5py

class WebMDScraper:
    """
    This class is the workhorse of the data collection aspect of this project. It scrapes all
    the reviews from WebMD, which are processed by ProcessWebMDReviews.py
    """
    def __init__(self):
        self._readinMeds()
        self.found_medications = {}
        
    def _readinMeds(self, medfile='MedicationsAndSideEffects.csv'):
        """
        Reading in the medication information from the csv generated after scraping Drugs.com
        """
        self.medications = list(pd.read_csv(medfile, sep='$', index_col=0)['Medication'])


    def searchWebMD(self, searchterm,
                    rootUrl='https://www.webmd.com/drugs/2/search?type=drugs&query='):
        """
        Finds the links for every medication by searching WebMD. (the link to the page that
        has a link to the reviews)
        """
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

            
    def getResultsLinks(self):
        """
        Search for links to medication pages
        """
        # Get the links to the medication pages
        for med in self.medications:
            self.searchWebMD(med)

        # Because some links aren't intuitive, I originally went through by hand and tried to correct,
        # but it only turned up a single medication that had 3 reviews, and caused weird Python errors
        # self.handParseWebMD()

        # Saving the links for the medications in an h5 file
        with h5py.File('DrugLinks.h5','w') as F:
            for key in self.found_medications:
                F[key] = self.found_medications[key]

        
    def getReviewsLink(self,url):
        """
        Search the medication pages for links to the reviews, and then alters the url to be usable
        for scraping through all the pages.
        """
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
        """
        Search the html for a single review (userPost) for all the needed information.

        One tricky thing here was handling the "see more" button to show the full reviews. To access the
        full text, I needed to know which review on the page I was looking at, which I tracked (revnum)
        """
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
        """
        Modifying the url for the reviews page to the next page
        """
        num = int(url[url.find('pageIndex=')+10:url.rfind('&sortby')])
        newurl = url.replace('pageIndex={:g}'.format(num),
                             'pageIndex={:g}'.format(num+1))
        return newurl

    def scrollReviews(self, reviewPage0, reviewsPerPage=5):
        """
        Goes through every page of reviews for a medication and pulls all the reviews.

        Input:
        ------
        reviewPage0 = The url for the first page, formatted to be explicit about page number
                      and reference all conditions
        reviewsPerPage = The number of reviews on each page. It's always 5 on WebMD, but I 
                         included this so that it would be easier to jump to other review sites

        Output:
        -------
        A dataframe of all of the reviews
        """
        
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

            # Avoid spamming the site
            time.sleep(0.025)
        
        # Process list all_reviews into dataframe and return
        return pd.DataFrame(all_reviews)
        
        
    def _formatExactMed(self,s: str)->str:
        """
        Formatting choices for how to write the medication in the filename
        """
        s = s.strip() # remove preceding and trailing white space
        s=s.replace(',','-') # Get rid of commas
        s=s.replace(' ', '-') # Get rid of white space
        s=s.replace('/', 'per') # remove things that make it look like a path
        return s
                        
    def pullReviewsByMed(self):
        """
        For every medication, search for reviews, and if they aren't found initially, 
        prompt the user to find the reviews page link (idiosyncracies on certain pages),
        because I would have missed some of the most highly reviewed medications otherwise.
        """

        for key in self.found_medications:
            link = self.found_medications[key]
            drug = key

            # Searching to see if I've already gotten data for that medication
            not_found=False
            name = self._formatExactMed(drug)
            if not glob.glob('Reviews/{:s}_reviews.csv'.format(name)):
                reviewLink = self.getReviewsLink(link)
                if not reviewLink: not_found = True
            else:
                reviewLink = None

            # Scrolling through all the reviews pages and writing the results to a csv
            if reviewLink and not not_found:
                print('Scrolling for:\n', reviewLink)
                reviewsDF = self.scrollReviews(reviewLink)
                if type(reviewsDF) != type(None):
                    reviewsDF.to_csv('Reviews/{:s}_reviews.csv'.format(name), sep='$')
            elif not_found:
                # Same thing as searching for the medication, idiosyncracies in the pages
                # led to a few searches by hand
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

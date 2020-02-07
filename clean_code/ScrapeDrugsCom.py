import requests
from bs4 import BeautifulSoup, SoupStrainer
import time
import pandas as pd
import pdb


class DrugsDotComScraper:
    def __init__(self):
        self._define_conditions()

    def _define_conditions(self):
        # Starting by listing links for each condition on Drugs.com to list of medications
        ADHD = "https://www.drugs.com/condition/attention-deficit-disorder.html?category_id=&include_rx=true&show_off_label=true&submitted=true&page_number=1"
        ADHD_end = 3

        Anxiety = "https://www.drugs.com/condition/generalized-anxiety-disorder.html?category_id=&include_rx=true&show_off_label=true&submitted=true&page_number=1"
        Anxiety_end = 1

        Bipolar = "https://www.drugs.com/condition/bipolar-disorder.html?category_id=&include_rx=true&show_off_label=true&submitted=true&page_number=1"
        Bipolar_end = 3

        Depression = "https://www.drugs.com/condition/major-depressive-disorder.html?category_id=&include_rx=true&show_off_label=true&submitted=true&page_number=1"
        Depression_end = 3

        Schizophrenia = "https://www.drugs.com/condition/schizophrenia.html?category_id=&include_rx=true&show_off_label=true&submitted=true&page_number=1"
        Schizophrenia_end=2

        self.condition_dict = {'ADHD': (ADHD, ADHD_end),
                               'Anxiety': (Anxiety, Anxiety_end),
                               'Bipolar-Disorder': (Bipolar, Bipolar_end),
                               'Depression': (Depression, Depression_end),
                               'Schizophrenia': (Schizophrenia, Schizophrenia_end)}

    def pull_med(self,soup):
        content = soup.find('table', attrs={'class':'condition-table data-list data-list--nostripe ddc-table-sortable'})
        meds = []
        for row in content.find_all('tr')[1::3]:  # Skipping the table column names and the commented out rows
            col1 = row.find_all('td')[0]

            drug = col1.find('a', attrs={"class":"condition-table__drug-name__link"}).text
            drug = drug[drug.find('>')+1:]
            
            meds.append(drug)

        return meds
            
    def scrape_meds(self):
        # Scrape the meds from the dictionary of conditions
        # Create dictionary of medications
        self.med_dict = {}
        for key in self.condition_dict:
            self.med_dict[key] = []
            for i in range(1,self.condition_dict[key][1]+1):
                url = self.condition_dict[key][0]
                url = url[:-1] + str(i)
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                meds = self.pull_med(soup)
                self.med_dict[key] += meds
                time.sleep(0.025)

    def getSELink(self,searchterm, rootUrl='https://www.drugs.com/search.php?searchterm='):
        if searchterm.find(' ') != -1:
            searchterm.replace(' ', '-')
        response = requests.get(rootUrl+searchterm)
        soup = BeautifulSoup(response.content, 'html.parser')
    
        sideEffLink = False
        for link in soup.find_all('a', href=True):
            if str(link).find('>Side Effects') != -1:
                sideEffLink = str(link)
                print(str(link))
                sideEffLink = sideEffLink[sideEffLink.find('"')+1:sideEffLink.rfind('"')]
                sideEffLink = 'https://www.drugs.com' + sideEffLink
                break
            
        if not sideEffLink:
            return False
        else:
            return sideEffLink

    def processSideEffects(self, bulletList):
        # first need to split away from immediate medical side effects (need to go to urgent care)
        # Finding the labels of interest on the webpage
        # Also trying to find the length of the list of bullets following the labels
        locs = []
        for i, string in enumerate(bulletList):
            if (string == '<i>More common</i>' or
                string == '<i>Less common</i>' or
                string == '<i>Incidence not known</i>'):
                locs.append(i)
                count = 0
            elif string[:3] == '<i>' and string[-4:] == '</i>' and len(locs):
                if type(locs[-1]) == int:
                    locs[-1] = (locs[-1], i-locs[-1]) 
        sideEffects = {}
        labelAndSE = []

        # Cleaning up a little bit on the length of the list of bullets
        for i in range(len(locs)-1):
            if type(locs[i]) == int:
                if type(locs[i+1]) != tuple:
                    locs[i] = (locs[i], locs[i+1]-locs[i])
                else:
                    locs[i] = (locs[i], locs[i+1][0]-locs[i])
    
        # Parsing the labels and list of bullets
        locLabels = [loc[0] for loc in locs]
        countINK = 0
        for j, locLabel in enumerate(locLabels):
            label = bulletList[locLabel]
            label = label[label.find('<i>')+3:label.rfind('</i>')]
            sEffects = []
            if label == 'Incidence not known': countINK += 1
            for i in range(locs[j][1]):
                if (bulletList[locLabel+i].find('ul') != -1 and
                    bulletList[locLabel+i].find('<p>') == -1):
                    effects = [string[:string.find('</li>')] for string in bulletList[locLabel+i].split('<li>')]
                    for effect in effects[1:]:
                        # Handling links appropriately
                        if effect.find('<a') != -1:
                            effect = effect[effect.rfind('>', 0, effect.find('</a>')):effect.rfind('</a>')]
                            # Sometimes there are two instances of a link in a link, this fixes that
                            if effect.find('<') != -1 or effect.find('>') != -1:
                                effect = effect[effect.rfind('>', 0, effect.find('</a>')):effect.rfind('</a>')]

                        sEffects.append(effect)
                    
            # Because the non-worrying side effects always come second, will overwrite in dictionary
            sEffectStr = ''
            for sEffect in sEffects: sEffectStr += '{:s}; '.format(sEffect)
            sEffectStr = sEffectStr.replace('>','')
            sideEffects[label] = sEffectStr
        
        if 'More common' not in sideEffects: sideEffects['More common'] = ''
        if 'Less common' not in sideEffects: sideEffects['Less common'] = ''
        if 'Incidence not known' not in sideEffects or countINK == 1:
            sideEffects['Incidence not known'] = ''

        return sideEffects
        
        
    def pullSEfromLink(self,url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
    
        content = soup.find('div', attrs={'class':'contentBox'})
        bulletLists = [str(child) for child in content.findChildren() if str(child).find('<ul>') != -1 or str(child).find('<i>') != -1]
    
        effectDict = self.processSideEffects(bulletLists)
        return effectDict

        
    def scrape_SEs(self):
        # Scrape the side effect information for each med
        self.full_dict = {}
        for condition in self.med_dict:
            se_dict = {}
            for key in self.med_dict[condition]:
                # Getting the three classes of side effects
                se_dict[key] = {}
                se_dict[key]['More common'] = []
                se_dict[key]['Less common'] = []
                se_dict[key]['Incidence not known'] = []

                searchLink = self.getSELink(key)
                time.sleep(0.025)
                
                if not searchLink:
                    se_dict[key]['More common'].append('')
                    se_dict[key]['Less common'].append('')
                    se_dict[key]['Incidence not known'].append('')
                else:
                    print('I just went to:\n',searchLink)
                    side_effects = self.pullSEfromLink(searchLink)
    
                    se_dict[key]['More common'].append(side_effects['More common'])
                    se_dict[key]['Less common'].append(side_effects['Less common'])
                    se_dict[key]['Incidence not known'].append(side_effects['Incidence not known'])
    
                time.sleep(0.025)
            self.full_dict[condition] = se_dict
            
    def create_dataframe(self):
        df_info = []
        for condition in self.full_dict:
            for med in self.med_dict[condition]:
                item = {}
                item['Condition'] = condition
                item['Medication'] = med
                item['More common'] = self.full_dict[condition][med]['More common']
                item['Less common'] = self.full_dict[condition][med]['Less common']
                item['Incidence not known'] = self.full_dict[condition][med]['Incidence not known']
                df_info.append(item)

        df = pd.DataFrame(df_info)
        df.to_csv('MedicationsAndSideEffects.csv', sep='$')


if __name__ == '__main__':
    Scraper = DrugsDotComScraper()

    # Pulling all the medications for each condition
    Scraper.scrape_meds()

    # Scrape the side effects for each medication for each condition
    Scraper.scrape_SEs()

    # Make a dataframe from all of that information and save it to a csv
    Scraper.create_dataframe()

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy import inspect
import psycopg2
import pandas as pd

# Dictionary for me to remember what the tables mean
#   Sourced from ASC_NTS.doc file in the downloads

# Keys of interest for each table (to be joined to table above)
demo_keys = {
    "PRIMARYID": 'Number related to faers report, = CASEID{:g} where g = the number of the check into the event',
    "CASEID": 'ID for the entire ADE case',
    "CASEVERSION": 'Number for the check in on a given case',
    "I_F_COD": 'Whether initial of follow-up',
    "EVENT_DT": 'Date of event',
    "FDA_DT": 'Date FDA received report',
    "AGE": 'Value of patients age',
    "AGE_COD": 'Unit abbreviation for patients age (keep on YR)',
    "GNDR_COD": 'Code for the patients sex (UNK, M, F, NS)',
}

drug_keys = {
    "PRIMARYID": 'Number related to faers report, = CASEID{:g} where g = the number of the check into the event',
    "CASEID": 'ID for the entire ADE case',
    "DRUG_SEQ": 'Unique number for IDing a drug for a case (need both to access ther data)',
    "ROLE_COD": 'PS=Primary suspect, SS=Secondary, C=Concomitant, I=Interacting',
    "DRUGNAME": 'Valid trade or Verbatim name',
    "VAL_VBM": '1 = VTN, 2 = VBN',
    "ROUTE": 'Route of administration',
    "DECHAL": 'Did reaction abate upon discontinuation',
    "RECHAL": 'Did reaction return upon recontinuation',
    "DOSE_AMT": 'Dosage',
    "DOSE_UNIT": 'Unit of drug dosage',
    "DOSE_FREQ": 'Code for frequency (look at doc if using)',
}

reac_keys = {
    "PRIMARYID": 'Number related to faers report, = CASEID{:g} where g = the number of the check into the event',
    "CASEID": 'ID for the entire ADE case',
    "PT": 'The term that is considered significant in the outcome file',
    "OUTCOME": 'The relevant outcome file'
}

outc_keys = {
    "PRIMARYID": 'Number related to faers report, = CASEID{:g} where g = the number of the check into the event',
    "CASEID": 'ID for the entire ADE case',
    "OUTC_COD": 'Code for a patient outcome (will probably always be OT for other for us)',
    "REPORT SOURCE": 'file that codes the source of the report'
}

rpsr_keys = {
    "PRIMARYID": 'Number related to faers report, = CASEID{:g} where g = the number of the check into the event',
    "CASEID": 'ID for the entire ADE case',
    "RPSR_COD": 'LIT=literature, CSM=consumer, HP=health professional, DT=distributor'
}

ther_keys = {
    "PRIMARYID": 'Number related to faers report, = CASEID{:g} where g = the number of the check into the event',
    "CASEID": 'ID for the entire ADE case',
    "DSG_DRUG_SEQ": 'DRUG_SEQ from drug_keys',
    "DUR": 'Numeric value of duration',
    "DUR_COD": 'YR, MON, WK, DAY, HR'
}

indi_keys = {
    "PRIMARYID": 'Number related to faers report, = CASEID{:g} where g = the number of the check into the event',
    "CASEID": 'ID for the entire ADE case',
    "INDI_DRUG_SEQ": 'DRUG_SEQ',
    "INIT_PT": 'Indication for use of drug'
}

tables = {
    "demo": ('Info on demographics',demo_keys),
    "drug": ('Info on the drug for as many medications as were related to the event', drug_keys),
    "indi": ('MedDRA terms coded for the indications to use', indi_keys),
    "outc": ('Patient outcomes from the event', outc_keys),
    "reac": ('MedDRA terms coded for the event', reac_keys),
    "rpsr": ('Repor sources for the event', rpsr_keys),
    "ther": ('drug therapy start and end dates for the reported drugs', ther_keys)
}



# Connect to make queries using psycopg2
con = None
con = psycopg2.connect(database = 'faers', 
                       user = 'postgres', 
                       host = 'localhost', 
                       password = 'jhett' )

# Defining how to get info on each drug
#   Just writing out some basic info
conditions = ['ADHD', 'Anxiety', 'Bipolar-Disorder', 'Depression', 'Schizophrenia']
path2uniqMeds = '~/Insight/PsychProject/UniqueMedications/'

#   Defining some useful functions to work with the data
getfile = lambda condition: pd.read_csv(path2uniqMeds+'Medications_unique_{:s}.csv'.format(condition), sep='$', index_col=0)
pullMeds = lambda ind, df: df.loc[ind]['All names']


# Wow this did magic:
# https://www.datadoghq.com/blog/100x-faster-postgres-performance-by-changing-1-line/

#   Query functions
def grabDrug(drugnames, connector):
    formattednames = "('"+"', '".join(list(set([drug.strip() for drug in drugnames.upper().split(', ')])))+"')"
    query = """SELECT primaryid, role_cod, drugname, drug_seq, val_vbm, route, dose_amt, dose_unit, dose_freq FROM drug WHERE drugname IN {:s};""".format(formattednames)

    df = pd.read_sql_query(query, con).fillna(value='')

    return df


def grabDemog(primaryIDs, con):
    # Shaping query
    keys = ', '.join(["({:s}::varchar)".format(ID) for ID in primaryIDs])
    query = """SELECT primaryid, i_f_code, event_dt, age, age_cod, sex FROM  demo WHERE primaryid = ANY( VALUES {:s});""".format(keys)
    
    # Iterating through and stacking pandas dataframes)
    df = pd.read_sql_query(query, con).fillna(value='')

    return df


def grabReac(primaryIDs, con):
    # Shaping query
    keys = ', '.join(["({:s}::varchar)".format(ID) for ID in primaryIDs])
    query = """SELECT primaryid, pt FROM reac WHERE primaryid = ANY( VALUES {:s});""".format(keys)
    
    # Iterating through and stacking pandas dataframes)
    df = pd.read_sql_query(query, con).fillna(value='')

    return df


def grabTher(primaryIDs, con):
    # Shaping query
    keys = ', '.join(["({:s}::varchar)".format(ID) for ID in primaryIDs])
    query = """SELECT primaryid, dsg_drug_seq, dur, dur_cod FROM ther WHERE primaryid = ANY( VALUES {:s});""".format(keys)
    
    # Iterating through and stacking pandas dataframes)
    df = pd.read_sql_query(query, con).fillna(value='')

    return df


def grabIndi(primaryIDs, con):
    # Shaping query
    keys = ', '.join(["({:s}::varchar)".format(ID) for ID in primaryIDs])
    query = """SELECT primaryid, indi_drug_seq, indi_pt FROM indi WHERE primaryid = ANY( VALUES {:s});""".format(keys)
    
    # Iterating through and stacking pandas dataframes)
    df = pd.read_sql_query(query, con).fillna(value='')

    return df

            
def grabRpsr(primaryIDs, con):
    # Shaping query
    keys = ', '.join(["({:s}::varchar)".format(ID) for ID in primaryIDs])
    query = """SELECT primaryid, rpsr_cod FROM rpsr WHERE primaryid = ANY( VALUES {:s});""".format(keys)
    
    # Iterating through and stacking pandas dataframes)
    df = pd.read_sql_query(query, con).fillna(value='')

    return df


def joinSQLTables(demo, drug, reac, ther, indi, rpsr):
    # Getting everything into pivot tables so we can join them into one monster file
    demoPV = demo.set_index('primaryid')

    drugPV = drug.pivot_table(index='primaryid',
                              values=[col for col in drug.columns if col != 'primaryid'],
                              aggfunc=', '.join)

    reacPV = reac.pivot_table(index='primaryid',
                              values=[col for col in reac.columns if col != 'primaryid'],
                              aggfunc=', '.join)

    therPV = ther.pivot_table(index='primaryid',
                              values=[col for col in ther.columns if col != 'primaryid'],
                              aggfunc=', '.join)

    indiPV = indi.pivot_table(index='primaryid',
                              values=[col for col in indi.columns if col != 'primaryid'],
                              aggfunc=', '.join)

    rpsrPV = rpsr.pivot_table(index='primaryid',
                              values=[col for col in rpsr.columns if col != 'primaryid'],
                              aggfunc=', '.join) 
    

    # Joining them in a more intelligent order
    massiveDF = pd.concat([demoPV, drugPV, therPV, indiPV, rpsrPV, reacPV], axis=1, sort=False)

    return massiveDF


# for condition in conditions:
#     print(condition)
#     uniqDF = getfile(condition)
#     for ind in uniqDF.index:
#         medications = pullMeds(ind, uniqDF)
#         print(medications)

#         drugDF = grabDrug(medications, con)
#         primaryIDs = drugDF['primaryid']
#         print('I pulled the drugs')
        
#         demoDF = grabDemog(primaryIDs, con)
#         print('I pulled the demographics')

#         reacDF = grabReac(primaryIDs, con)
#         print('I pulled the reactions')
        
#         therDF = grabTher(primaryIDs, con)
#         print('I pulled the treatment duration')
        
#         indiDF = grabIndi(primaryIDs, con)
#         print('I pulled the indicated condition')
        
#         rpsrDF = grabRpsr(primaryIDs, con)
#         print('I pulled who reported it')
        

#         finished_product = joinSQLTables(demoDF, drugDF, reacDF, therDF, indiDF, rpsrDF)
#         print('I joined everything magically together')
        
#         savefile = lambda drug, cond: 'faers_results/{:s}/faers_pull_{:s}.csv'.format(cond, drug.strip())

#         fname = savefile(medications.split(', ')[0], condition)
#         finished_product.to_csv(fname, sep='$')
#         print('I saved the file')


for condition in ['Bipolar-Disorder']:
        medications = 'LITHIUM'
        print(medications)

        drugDF = grabDrug(medications, con)
        primaryIDs = drugDF['primaryid']
        print('I pulled the drugs')
        
        demoDF = grabDemog(primaryIDs, con)
        print('I pulled the demographics')

        reacDF = grabReac(primaryIDs, con)
        print('I pulled the reactions')
        
        therDF = grabTher(primaryIDs, con)
        print('I pulled the treatment duration')
        
        indiDF = grabIndi(primaryIDs, con)
        print('I pulled the indicated condition')
        
        rpsrDF = grabRpsr(primaryIDs, con)
        print('I pulled who reported it')
        

        finished_product = joinSQLTables(demoDF, drugDF, reacDF, therDF, indiDF, rpsrDF)
        print('I joined everything magically together')
        
        savefile = lambda drug, cond: 'faers_results/{:s}/faers_pull_{:s}.csv'.format(cond, drug.strip())

        fname = savefile(medications.split(', ')[0], condition)
        finished_product.to_csv(fname, sep='$')
        print('I saved the file')

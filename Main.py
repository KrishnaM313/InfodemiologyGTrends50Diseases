import numpy as np
import pandas as pd
import os
import json

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import CatalyseGetData as cgd
import CatalyseAnalyses as ca

base_folder = r"D:\projects\catalyseNEW\AllDis\results"
# Terms
termsList = {
  "Acute Bronchitis": ["bronchitis", "chest infection", "cough"],
  "Acute respiratory infections (ARI)": ["respiratory infection", "chest infection", "severe cough"],
  "Allergic Rhinitis": ["allergy", "hay fever", "pollen allergy"],
  "Asthma": ["asthma", "wheezing", "inhaler"],
  "Bronchiolitis": ["bronchiolitis", "RSV", "baby cough"],
  "Bronchitis": ["bronchitis", "chronic cough", "lung infection"],
  "Bullous Dermatoses": ["skin blisters", "pemphigoid", "autoimmune skin"],
  "Chickenpox": ["chickenpox", "varicella", "itchy rash"],
  "Common Cold": ["common cold", "runny nose", "stuffy nose"],
  "Conjunctival Disorders": ["pink eye", "conjunctivitis", "red eye"],
  "COVID-19": ["covid", "coronavirus", "long covid"],
  "Croup": ["croup", "barking cough", "child cough"],
  "Denom": ["denom disease", "rare denom", "denom syndrome"],
  "ECLD - Asthma exacerbations": ["asthma attack", "asthma flare", "severe asthma"],
  "ECLD - COPD exacerbations": ["COPD flare", "emphysema flare", "chronic bronchitis flare"],
  "Exacerbations of chronic lung disease": ["lung flare", "respiratory exacerbation", "breathing difficulty"],
  "Herpes Simplex": ["herpes simplex", "cold sores", "oral herpes"],
  "Herpes Zoster": ["shingles", "herpes zoster", "shingles rash"],
  "Impetigo": ["impetigo", "skin sores", "contagious rash"],
  "Infectious Intestinal Diseases": ["stomach infection", "intestinal infection", "foodborne illness"],
  "Infectious Mononucleosis": ["mono", "glandular fever", "epstein-barr virus"],
  "Influenza-like illness": ["flu", "influenza", "fever and chills"],
  "Laryngitis": ["laryngitis", "hoarse voice", "sore throat"],
  "Laryngitis and Tracheitis": ["laryngitis", "tracheitis", "hoarse voice"],
  "Lower Respiratory Tract Infections": ["lung infection", "bronchitis", "pneumonia"],
  "Measles": ["measles", "rubeola", "red rash"],
  "Meningitis and Encephalitis": ["meningitis", "encephalitis", "brain infection"],
  "Mumps": ["mumps", "swollen glands", "parotitis"],
  "Non-infective Enteritis and Colitis": ["enteritis", "colitis", "gut inflammation"],
  "Otitis Media": ["ear infection", "middle ear infection", "ear pain"],
  "Otitis Media Acute New": ["acute ear infection", "ear pain", "middle ear infection"],
  "Peripheral Nervous Disease": ["peripheral neuropathy", "nerve pain", "numbness"],
  "Pleurisy": ["pleurisy", "chest pain", "lung inflammation"],
  "Pneumonia": ["pneumonia", "lung infection", "cough with fever"],
  "Pneumonia and Pneumonitis": ["pneumonia", "lung inflammation", "breathing difficulty"],
  "Respiratory System Diseases": ["lung disease", "breathing problems", "respiratory condition"],
  "Rubella": ["rubella", "german measles", "rash"],
  "Scabies": ["scabies", "itchy rash", "skin mites"],
  "Sinusitis": ["sinus infection", "sinusitis", "facial pain"],
  "Skin and Subcutaneous Tissue Infections": ["skin infection", "cellulitis", "boils"],
  "Strep Throat and Peritonsillar Abscess": ["strep throat", "tonsil infection", "sore throat"],
  "Symptoms involving musckuoskeletal": ["muscle pain", "joint pain", "stiffness"],
  "Symptoms involving musculoskeletal": ["joint pain", "muscle ache", "stiffness"],
  "Symptoms involving Respiratory and Chest": ["cough", "shortness of breath", "chest discomfort"],
  "Symptoms involving Skin and Integument Tissues": ["rash", "skin irritation", "itching"],
  "Tonsillitis and acute Pharyngitis": ["tonsillitis", "sore throat", "throat infection"],
  "Tonsillitis/Pharyngitis": ["tonsillitis", "pharyngitis", "sore throat"],
  "Upper Respiratory Tract Infections": ["URTI", "cold", "sore throat"],
  "Urinary Tract Infections": ["UTI", "urinary infection", "painful urination"],
  "Viral Hepatitis": ["hepatitis", "liver infection", "jaundice"],
  "Whooping Cough": ["whooping cough", "pertussis", "severe cough"]
}

# Get incidence data

fp = r"D:\projects\catalyseNEW\AllDis\data\allDis0.csv"
idf = pd.read_csv(fp).replace("", np.nan).fillna(0)
#print(df.head)
sD = "20200106"
eD = "2024129"
sd = "2020-01-06"
ed = "2024-01-31"
ed1 = "2019-12-31"
sd1 = "2014-12-29"

def buildDF(c):
    incidence = idf[c].to_list()
    date = idf["Row Labels"].to_list()
    #gt = cgd.gtrends(sd, ed, termsList[c])
    #tk = cgd.toksearch(sD, eD, termsList[c])

    '''ml = min(len(incidence), len(date), len(gt), len(tk))
    incidence = incidence[:ml]
    date = date[:ml]
    gt = gt[:ml]
    tk = tk[:ml]

    df = pd.DataFrame({"incidence":incidence, "date":date, "gt":gt, "tk":tk})'''

    gr = cgd.gtrends(sd, ed, termsList[c])
    #print(gt0)
    gr1 = cgd.gtrends(sd1, ed1, termsList[c])
    #print(gt1)
    gr.extend(gr1)
    #print(gt)

    ml = min(len(incidence), len(date), len(gr))
    incidence = incidence[:ml]
    date = date[:ml]
    gr = gr[:ml]

    df = pd.DataFrame({"date":date, "incidence":incidence,  "gr":gr})
    
    return df

#ddf = buildDF("Common Cold")


# testing methods
'''ddf = pd.read_csv(r"D:\projects\catalyseNEW\AllDis\data\ddf.csv")
print(ddf.head)
#ddf.to_csv(r"D:\projects\catalyseNEW\AllDis\data\ddf.csv")
ds = "test"
folder = os.path.join(base_folder, ds)
os.makedirs(folder, exist_ok=True)

ddf1 = ca.normalise(ddf)
ca.plotSeries(ddf1, opf=folder)

ca.findCorrs(ddf1, opf=folder)

warnings.filterwarnings("ignore")
ca.rolling_origin_week_based(ddf1,"incidence", "week", 4, opf=folder)

results = pd.read_csv(folder)
T1 = results.groupby(['model', 'horizon'])[['mae', 'rmse', 'nrmse', 'mase']].mean().round(2)
T2 = results[results["horizon"]==0].round(2)'''

##################

def doAnalyses(base_folder):
    spm = {}
    for c in idf.columns:
        if c in termsList.keys():
            try:
                print("!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!" + c + "!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!")
                folder = os.path.join(base_folder, c)
                os.makedirs(folder, exist_ok=True)

                df0 = buildDF(c)
                ddf1 = ca.normalise(df0)
                ca.plotSeries(ddf1, opf=folder)
                sc = ca.findCorrs(ddf1, opf=folder)
                spm[c] = sc
                ca.rolling_origin_week_based(ddf1,"incidence", "week", 8, opf=folder)
                df0.to_csv(os.path.join(folder, "ds.csv"))
                #results = pd.read_csv(os.path.join(folder, "results.csv"))
            except Exception as E:
                print(E)
                continue
    with open(os.path.join(base_folder, "spearmen.json"), 'w') as f:
        json.dump(spm, f)

#running
doAnalyses(base_folder)





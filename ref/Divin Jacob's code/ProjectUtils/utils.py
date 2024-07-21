import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
_FOLDER = "./"

def TrainTestSplit(readFile, writeFile, split=0.2):
    filterPlateau = pd.read_csv(_FOLDER+readFile)
    drugIDs = list(filterPlateau['DRUG_ID'].unique())
    drugIDs = np.squeeze(drugIDs)
    trainTest = {}
    lowest = 1000
    lowestId = 0
    for drugID in drugIDs:
        drugResponses = list(filterPlateau.loc[filterPlateau['DRUG_ID'] == drugID]['DRUGID_COSMICID'])
        if(len(drugResponses) < lowest):
            lowest = len(drugResponses)
            lowestId = drugID
        data = []
        trainDCID, testDCID = train_test_split(drugResponses, test_size=split, random_state=42)
        data.append(trainDCID)
        data.append(testDCID)
        trainTest[str(drugID)] = data
    with open(_FOLDER + 'dataSplit.txt', 'w') as convert_file:
        convert_file.write(json.dumps(trainTest))
    return "DONE"
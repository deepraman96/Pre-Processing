#pip install pandas
#pip install imblearn
#pip install matplotlib
#pip install tensorflow
#pip install keras

print("Process Started...")
import warnings
warnings.filterwarnings("ignore")

import pandas
import numpy as np
import matplotlib.pyplot as plotG

import socket, struct

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


from sklearn.preprocessing import LabelEncoder
from collections import Counter

pathForTheDataSet = "/Users/mac/Desktop/Project/archive/Dataset-Unicauca-Version2-87Atts.csv"
objReader = pandas.read_csv(pathForTheDataSet)

# Checking if any value in the dataframe is null
if objReader.isnull().values.any():
    # removing NA values
    objReader.dropna()
    print("Some of the values in dataframe are null")
else:
    print("None of the values in dataframe are null")



# Checking types of values
dataTypesInDataSet=objReader.dtypes
columns=objReader.columns.tolist()
for x in range(len(dataTypesInDataSet)):
    print('{} , {}'.format(columns[x],dataTypesInDataSet[x]))


# Checking occurance of each application
print(objReader['ProtocolName'].value_counts())
print('Total no. of protocols used : {}'.format(len(objReader['ProtocolName'])))


index_names = objReader[ (objReader['ProtocolName'] != "EBAY") & (objReader['ProtocolName'] != "NETFLIX") & (objReader['ProtocolName'] != "INSTAGRAM") & (objReader['ProtocolName'] != "SPOTIFY") & (objReader['ProtocolName'] != "OFFICE_365") ].index
objReader.drop(index_names, inplace = True)

# Plot the number of records for individual applications
countOfProtocolUsed = objReader['ProtocolName'].value_counts()
plotG.figure(1)
countOfProtocolUsed.plot(kind='bar', title='Occurance Of Individual Application');
plotG.show()


features = [x for x in objReader.columns if x != 'ProtocolName' and x != 'Flow.ID' and x != 'Timestamp' and x != 'Label' and x != 'Source.IP' and x != 'Destination.IP']
X = objReader[features].astype(float)
Y = objReader['ProtocolName']


SampSize=6000
SlctApps = {"OFFICE_365":   SampSize,
            "EBAY":       SampSize,
            "NETFLIX":    SampSize,
            "INSTAGRAM":  SampSize,
            "SPOTIFY":    SampSize}

# manage unbalanced data
pipe = make_pipeline(
    SMOTE(sampling_strategy=SlctApps)
)

X_resampled, y_resampled = pipe.fit_resample(X, Y)
#over sampling

print("Size of Total dataset " + str(objReader.shape))
print("Size of Actual Features " + str(X.shape))
print("Size of Processed Features " + str(X_resampled.shape))

###################################
#Converting output class to numeric
labelEncoder = LabelEncoder()
labelEncoder.fit(Y)
labelEncoded_Y = labelEncoder.transform(Y)
Y=labelEncoded_Y
print("Labelled data")
print(Y)

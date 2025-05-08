from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import unique
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB

def datasetPrep(dataset):

    #pulls data from csv
    df = read_csv(dataset, header=None)
    
    #add labels to all the columnns 
    df.columns = ['ID', 
                'Label',
                'Statement',
                'Subject',
                'Speaker',
                'Occupation',
                'Location',
                'Party',
                'Barely True Counts',
                'False Counts',
                'Half True Counts',
                'Mostly True Counts',
                'Pants on Fire Counts',
                'Context']
    
    #drops columns that are not helpful.
    df = df.drop(['Location',
                'Barely True Counts',
                'False Counts',
                'Half True Counts',
                'Mostly True Counts',
                'Pants on Fire Counts'], axis=1)
    
    #drops rows that have been labeled "half-true"
    df = df.drop(df[df['Label'] == 'half-true'].index)

    #drops rows with information missing in columns
    df = df.dropna()
    
    #drops rows with duplicates 
    df = df.drop_duplicates()
    
    # print(df.shape)
    # print(df.head())
    

    X = df[['Statement',
            'Subject',
            
            'Party',
            'Occupation',
            'Context']]
    
    # print()
    # print(X.shape)
    # print(X.isna().sum())  
    
    Y = df['Label']

    X = X.astype(str)
    return X, Y 

def prepareInputs(X):
    oe = OrdinalEncoder()
    oe.fit(X)
    xEncode = oe.transform(X)
    return xEncode

def prepareTargets(Y):
    le = LabelEncoder()
    le.fit(Y)
    yEncode = le.transform(Y)
    return yEncode

def chi2featureSelection(x,y):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(x,y)
    # xFeatureSelection = fs.transform(x)
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.show()

def labelConversion(outputs):
    outputs = outputs.replace(["barely-true","pants-fire"], "FALSE")
    outputs = outputs.replace(["mostly-true"], "TRUE")
    return outputs

def logReg(x,y,xTest,yTest):
    lrModel = LogisticRegression(solver='lbfgs')
    lrModel.fit(x,y)
    testModel = lrModel.predict(xTest)
    return accuracy_score(yTest, testModel)

def main():
    x,y = datasetPrep("Dataset/Train csv_version.csv")
    y = labelConversion(y)
    x = prepareInputs(x)
    y = prepareTargets(y)
    xTest, yTest = datasetPrep("Dataset/Test csv_version.csv")
    yTest = labelConversion(yTest)
    xTest = prepareInputs(xTest)
    yTest = prepareTargets(yTest)
    accuracy = logReg(x,y,xTest,yTest)
    print(accuracy)

    chi2featureSelection(x,y)


main()
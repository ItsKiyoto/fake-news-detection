from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB

def prepareDataset(dataset):
    
    #retrieves data from csv file
    rawData = read_csv(dataset, header=None)
    
    #drops any rows that have null as any values
    rawData = rawData.dropna()
    
    #drops any rows that are duplicates to other rows
    rawData = rawData.drop_duplicates()
    
    #converts pandas dataframe into numpy array
    data = rawData.values.astype(str)
    print(data.shape)
    
    #seperates the numpy dataset into two datasets x for inputs and y for output
    xValues = data[:, 2:]
    yValues = data[:, 1]
    
    columnUnique(xValues)
    TwoClassConversion(yValues)
    
    return xValues, yValues

#calculates the uniqueness of the each row 
def columnUnique(data):
    for i in range(data.shape[1]):
        num = len(np.unique(data[:, i]))
        percentage = float(num) / data.shape[0] * 100
        print('%d, %d, %.1f%%' % (i, num, percentage))
        
        
def TwoClassConversion(outputs):
    #converts output classes from 6 to 2
    for y in range(0,len(outputs)):
        if (outputs[y] == "barely-true" or outputs[y] == "pants-fire"):
            outputs[y] = "FALSE"
        elif (outputs[y] == "mostly-true" or outputs[y] == "half-true"):
            outputs[y] = "TRUE"
            
def ThreeClassConversion(outputs):
    #converts output classes from 6 to 3
    for y in range(0,len(outputs)):
        if ( outputs[y] == "pants-fire"):
            outputs[y] = "FALSE"
        elif (outputs[y] == "mostly-true" ):
            outputs[y] = "TRUE"
        elif (outputs[y] == "barely-true" or outputs[y] == "half-true"):
            outputs[y] = "UNSURE"

def piechart(outputs):
    y = np.array([0, 0])
    mylabels = ["TRUE", "FALSE"]
    for i in range(0,len(outputs)):
        if outputs[i] == "TRUE":
            y[0] += 1
        else:
            y[1] += 1

    plt.pie(y, labels = mylabels)
    plt.show()
    
def politicalLabels(inputs):
    values = []
    for i in range(0,len(inputs)):
        if not (np.isin(inputs[i][-1], values)):
            values.append(inputs[i][-1])
    return values

def piechartTrue(inputs,outputs,labels):
    values = np.zeros(len(labels))
    for i in range(0,len(inputs)):
        if outputs[i] == "TRUE":
            index = labels.index(inputs[i][-1])
            values[index] += 1
    
    plt.pie(values, labels = labels)
    plt.title("True statements per political alignment")
    plt.show()

def piechartFalse(inputs,outputs,labels):
    values = np.zeros(len(labels))
    for i in range(0,len(inputs)):
        if outputs[i] == "FALSE":
            index = labels.index(inputs[i][-1])
            values[index] += 1
    
    plt.pie(values, labels = labels)
    plt.title("False statements per political alignment")
    plt.show()

    
def textVectorising(data,outputs):   
    countVect = CountVectorizer()
    xTrainStatement = countVect.fit_transform(data[:,0])
    xTrainSubject = countVect.fit_transform(data[:,1])
    xTrainSpeaker = countVect.fit_transform(data[:,2])
    xTrainParty = countVect.fit_transform(data[:,3])
    # print(xTrainStatement.shape)
    # print(xTrainSubject.shape)
    # print(xTrainSpeaker.shape)
    # print(xTrainParty.shape)
    
    tfTransformer = TfidfTransformer()
    xTrainStmtTfIdf = tfTransformer.fit_transform(xTrainStatement)
    xTrainSbjtTfIdf = tfTransformer.fit_transform(xTrainSubject)
    xTrainSpkrTfIdf = tfTransformer.fit_transform(xTrainSpeaker)
    xTrainPrtyTfIdf = tfTransformer.fit_transform(xTrainParty)
#     print(xTrainStatement.shape)
#     print(xTrainSubject.shape)
#     print(xTrainSpeaker.shape)
#     print(xTrainParty.shape)
    
#     text_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', MultinomialNB()),
    
#     statmentPipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer())])
    
#     subjectPipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer())])
    
#     speakerPipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer())])
    
#     partyPipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer())])
        
#     recombine = ColumnTransformer([
#     ('statement', statmentPipeline, data[:,0]),
#     ('subject', subjectPipeline, data[:,1]),
#     ('speaker', speakerPipeline, data[:,2]),
#     ('party', partyPipeline, data[:,3])
#     ])

    recombine = ([
    ('statement', xTrainStmtTfIdf), ('subject', xTrainSbjtTfIdf), ('speaker', xTrainSpkrTfIdf), ('party', xTrainPrtyTfIdf)
    ])

    
    nbc = Pipeline([("reconbine", recombine),  ('clf', MultinomialNB())])
    print("did it work?")
   

x, y = prepareDataset("Modified Train.csv")
# poliLables = politicalLabels(x)
# piechart(y)
# piechartTrue(x,y,poliLables)
# piechartFalse(x,y,poliLables)
y2 = TwoClassConversion(y)
textVectorising(x,y2)


    # countVect = CountVectorizer()
    # xTrainStatement = countVect.fit_transform(data[:,0])
    # xTrainSubject = countVect.fit_transform(data[:,1])
    # xTrainSpeaker = countVect.fit_transform(data[:,2])
    # xTrainParty = countVect.fit_transform(data[:,3])

    # tfTransformer = TfidfTransformer()
    # xTrainStmtTfIdf = tfTransformer.fit_transform(xTrainStatement)
    # xTrainSbjtTfIdf = tfTransformer.fit_transform(xTrainSubject)
    # xTrainSpkrTfIdf = tfTransformer.fit_transform(xTrainSpeaker)
    # xTrainPrtyTfIdf = tfTransformer.fit_transform(xTrainParty)
    
    # text_clf = Pipeline([
    # ('vect', CountVectorizer()),
    # ('tfidf', TfidfTransformer()),
    # ('clf', MultinomialNB()),


def datasetPrepOriginal(dataset):
    #pulls data from csv
    df = read_csv(dataset, header=None)
    
    df.columns = ['ID', 
                'Label',
                'Statement',
                'Subject',
                'Speaker',
                'Party']

    df = df.dropna()
    
    df = df.drop_duplicates()

    #print(df.head())
    X = df[['Statement',
            'Subject',
            'Speaker',
            'Party']]
    #X = df[['Statement','Subject','Speaker','Party']]
    Y = df['Label']

    #print(X.shape, Y.shape)

    #print(Y.head())
    return X, Y   

def multinomialNB_6labels():
    x,y = datasetPrepOriginal("Modified Train.csv")
    model = textOriginal(x,y)
    testx, testy = datasetPrepOriginal("Modified Test.csv")
    print("Accuracy:")
    testModel(model, testx, testy)

def multinomialNB_3labels():
    x,y = datasetPrepOriginal("Modified Train.csv")
    twoY = ThreeClassConversion(y)
    model = textOriginal(x,twoY)
    testx, testy = datasetPrepOriginal("Modified Test.csv")
    print("Accuracy:")
    twoYTest = ThreeClassConversion(testy)
    testModel(model, testx, twoYTest)

    def multinomialNB_2labels():
    x,y = datasetPrepOriginal("Modified Train.csv")
    twoY = TwoClassConversion(y)
    model = textOriginal(x,twoY)
    testx, testy = datasetPrepOriginal("Modified Test.csv")
    print("Accuracy:")
    twoYTest = TwoClassConversion(testy)
    testModel(model, testx, twoYTest)

    barelyTruePipeline = Pipeline([
    ('vect', DictVectorizer())])
    
    falsePipeline = Pipeline([
    ('vect', DictVectorizer())])
    
    halfTruePipeline = Pipeline([
    ('vect', DictVectorizer())])
    
    mostlyTruePipeline = Pipeline([
    ('vect', DictVectorizer())])

    pantsOnFirePipeline = Pipeline([
    ('vect', DictVectorizer())])

        # ('barelyTrue', barelyTruePipeline, 'Barely True Counts'),
    # ('false', falsePipeline, 'False Counts'),
    # ('halfTrue', halfTruePipeline, 'Half True Counts'),
    # ('mostlyTrue', mostlyTruePipeline, 'Mostly True Counts'),
    # ('pantsOnFire', pantsOnFirePipeline, 'Pants on Fire Counts'),

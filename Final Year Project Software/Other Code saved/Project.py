from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

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
                # 'Occupation',
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

    X = df[['Statement',
            'Subject',
            'Speaker',
            'Party',
            'Occupation',
            'Context'
            ]]
        
    Y = df['Label']
    X = X.astype(str)
    return X, Y 
        
def TwoClassConversion(outputs):
    outputs = outputs.replace(["barely-true","pants-fire"], "FALSE")
    outputs = outputs.replace(["mostly-true"], "TRUE")
    return outputs

def textVectorising():
    statmentPipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())])
    
    subjectPipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())])
    
    speakerPipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())])
    
    partyPipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())])

    occupationPipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())])
    
    contextPipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())])
        
    recombine = ColumnTransformer([ 
    ('statement', statmentPipeline, 'Statement'),
    ('subject', subjectPipeline, 'Subject'),
    ('speaker', speakerPipeline, 'Speaker'),
    ('party', partyPipeline, 'Party'),
    ('occupation', occupationPipeline, 'Occupation'),
    ('context', contextPipeline, 'Context')
    ])

    return recombine


def nbModel(x,y,pipeline):
    nbc = Pipeline([
    ("recombine", pipeline),  
    ('clf', MultinomialNB())])
    nbc.fit(x, y)
    return nbc

def svmModel(x,y,pipeline):
    svm = Pipeline([
    ("recombine", pipeline),  
    ('clf', SGDClassifier(loss = 'hinge', 
                          penalty='l2',
                          alpha=1e-3, 
                          random_state=42,
                          max_iter=5, 
                          tol=None))])
    svm.fit(x, y)
    return svm


def testModel(model, data, output):
    predicted = model.predict(data)
    print(classification_report(output, predicted)) 
    # print((np.mean(predicted == output))*100)

# x,y = datasetPrep("Dataset/Train csv_version.csv")
# model = textVectorising(x,y)

def multinomialNB(trainDataset, testDataset):
    x,y = datasetPrep(trainDataset)
    y = TwoClassConversion(y)
    pipeline = textVectorising()
    nbTrainedModel = nbModel(x,y,pipeline)
    testx, testy = datasetPrep(testDataset)
    print("Accuracy:")
    twoYTest = TwoClassConversion(testy)
    testModel(nbTrainedModel, testx, twoYTest)

def svmClassifying(trainDataset, testDataset):
    x,y = datasetPrep(trainDataset)
    y = TwoClassConversion(y)
    pipeline = textVectorising()
    svmTrainedModel = svmModel(x,y,pipeline)
    testx, testy = datasetPrep(testDataset)
    print("Accuracy:")
    twoYTest = TwoClassConversion(testy)
    testModel(svmTrainedModel, testx, twoYTest)

def hyperParamTuningGridSearch(x,y,textClf,testX,testY):
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }

    param_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
    }

    gsClf = GridSearchCV(textClf, parameters, cv= 5, n_jobs=-1)
    gsClf.fit(x,y)
    testModel(gsClf, testX, testY)

    
def main():
    trainDataset1 = "Dataset/Train csv_version.csv"
    testDataset1 = "Dataset/Test csv_version.csv"
    print("Naive bayes Results")
    multinomialNB(trainDataset1,testDataset1)
    print()
    print('SVM Results')
    svmClassifying(trainDataset1,testDataset1)

main()

# print()
# print("modified dataset")
# multinomialNB(trainDataset2,testDataset2)

#originalModifiedDatasetISOT()


# Statistic code for graphs and other things

#calculates the uniqueness of the each row 
def columnUnique(data):
    for i in range(data.shape[1]):
        num = len(np.unique(data[:, i]))
        percentage = float(num) / data.shape[0] * 100
        print('%d, %d, %.1f%%' % (i, num, percentage))

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

# poliLables = politicalLabels(x)
# piechart(y)
# piechartTrue(x,y,poliLables)
# piechartFalse(x,y,poliLables)


# param_nb = {
# 'var_smoothing': np.logspace(0,-9, num=100)
# }
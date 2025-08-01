import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set_theme()

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#from src.features import CountVectorizer, TfidfTransformer
#from src.preprocessing import *


def nbPipeline():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])
    return pipeline

def svmPipeline():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SVC(probability=True,random_state=42))
    ])
    return pipeline

def decisionTreePipeline():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])
    return pipeline

def randomForestPipeline():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    return pipeline

def xgBoostPipeline():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=42))
    ])
    return pipeline

def parameterTuning(model, x, y, parameters): 
    gsModel = GridSearchCV(model, parameters, cv=10, n_jobs=-1, scoring='accuracy')
    gsModel.fit(x, y)
    return gsModel

def nbSmote():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('smote', SMOTE(random_state=42)),
        ('clf', MultinomialNB())
    ])
    return pipeline

def svmSmote():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('smote', SMOTE(random_state=42)),
        ('clf', SVC(probability=True, random_state=42))
    ])
    return pipeline

def decisionTreeSmote():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('smote', SMOTE(random_state=42)),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])
    return pipeline

def randomForestSmote():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    return pipeline

def xgBoostSmote():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('smote', SMOTE(random_state=42)),
        ('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=42))
    ])
    return pipeline

def testModel(model, xT, yT, modelName):
    # tests predicition of model on test data
    predicted = model.predict(xT)
    
    # prints out performance metrics for a given model
    print(classification_report(yT, predicted))
   
    # creates heatmap confusion matrix 
    classes = ["False", "True"]
    confMatx = confusion_matrix(yT, predicted)
    confHeatMap = sns.heatmap(confMatx.T, square = True, annot = True, fmt = 'd', xticklabels = classes, yticklabels = classes)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.title(modelName)
    plt.show()

    # Uncomment if you want confusion matrix visualization
    # classes = ["FALSE", "TRUE"]
    # confMatx = confusion_matrix(yT, predicted)
    # plt.figure(figsize=(8, 6))
    # confHeatMap = sns.heatmap(confMatx.T, square=True, annot=True, fmt='d', 
    #                          xticklabels=classes, yticklabels=classes)
    # plt.xlabel("True label")
    # plt.ylabel("Predicted label")
    # plt.title(f"{modelName} Confusion Matrix")
    # plt.show()

def kfold(modelName,model, x, y):
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    scores = cross_val_score(model, x, y, cv=kf, scoring='accuracy')
    # prints average accuracy from all 10 folds to 2 decimal places
    #print(modelName + ": %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    #print(modelName + ": %0.2f accuracy" % (scores.mean()))
    print(f"{modelName}: {scores.mean():.2f}")

def rocAUC(model, xT, yT, modelName):
    try:
        modProba = model.predict_proba(xT)
        modProba = modProba[:, 1]
        AucScore = roc_auc_score(yT, modProba)
        print(modelName + " AUC Score: %.3f" % (AucScore))
    except Exception as e:
        print(f"Could not calculate AUC for {modelName}: {e}")

def runPipeline(modelName, modelPipeline, hypParam, smotePipeline, x, y, xT, yT, xV, yV):
    #print(modelName)
    print(f"\n{'='*50}")
    print(f"RUNNING: {modelName}")
    print(f"{'='*50}")
    # x,y,xT,yT,xV,yV = getData()

    # Basic Model
    model = modelPipeline()
    model.fit(x,y)
    testModel(model, xT, yT, modelName)
    rocAUC(model, xT, yT, modelName)
    
    # K-Fold Cross Validation
    print(f"\n{modelName} Cross-Validation:")
    modelKfold = modelPipeline()
    kfold(modelName, modelKfold, x, y)
    
    # Hyperparameter Tuning
    print("")
    print(modelName + " with Hyperparameter Tuning")
    modelTuned = parameterTuning(model, xV, yV, hypParam)
    testModel(modelTuned, xT, yT, modelName + " HP Tuned")
    rocAUC(modelTuned, xT, yT, modelName + " HP Tuned")
    
    print("")
    print(modelName + " with Hyperparameter Tuning and SMOTE")
    modelSmote = smotePipeline()
    modelSmote = modelSmote.fit(x,y)
    smoteTuned = parameterTuning(modelSmote, xV, yV, hypParam)
    testModel(smoteTuned, xT, yT, modelName + " HP Tuned + SMOTE")
    rocAUC(smoteTuned, xT, yT, modelName + " HP Tuned + SMOTE")


    

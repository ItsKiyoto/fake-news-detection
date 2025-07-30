import os
import sys
import numpy as np
from pandas import read_csv
#from sklearn.model_selection import train_test_split

from model import *
from preprocessing import *


def getData(data_dir="Data"):
    """
    Load and preprocess data from CSV files
    """
    try:
        # Use relative paths for better portability
        train_path = os.path.join(data_dir, "Train csv_version.csv")
        test_path = os.path.join(data_dir, "Test csv_version.csv")
        valid_path = os.path.join(data_dir, "Valid csv_version.csv")
        
        print("Loading datasets...")
        trainDataset = read_csv(train_path, header=None)
        testDataset = read_csv(test_path, header=None)
        validDataset = read_csv(valid_path, header=None)
        
        print("Processing datasets...")
        x, y = read_dataset(trainDataset)
        xT, yT = read_dataset(testDataset)
        xV, yV = read_dataset(validDataset)

        return x, y, xT, yT, xV, yV
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        print("Please make sure your CSV files are in the 'Data' directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # # retrieve dataset from csv files
    # trainDataset = read_csv("/Users/Kishan/Documents/Kishan/Repositories/fake-news-detection/Final Year Project Software/Data/Train csv_version.csv", header=None)
    # testDataset = read_csv("/Users/Kishan/Documents/Kishan/Repositories/fake-news-detection/Final Year Project Software/Data/Test csv_version.csv", header=None)
    # validDataset = read_csv("/Users/Kishan/Documents/Kishan/Repositories/fake-news-detection/Final Year Project Software/Data/Valid csv_version.csv", header=None)

    # x,y = read_dataset(trainDataset)
    # xT, yT = read_dataset(testDataset)
    # xV, yV = read_dataset(validDataset)

    # return x,y,xT,yT,xV,yV


def runAll():
    mnbParameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        "tfidf__norm": ("l1", "l2"), # maybe remove this
        'clf__alpha': (1e-2, 1e-3)
    }
        
    SvmSvcParameters = {        
        "vect__ngram_range": ((1, 1), (1, 2)),
        "tfidf__use_idf": (True, False),
        "tfidf__norm": ("l1", "l2"),
        'clf__C': (0.1, 1, 10, 100),
        'clf__gamma': (0.01, 0.1, 1, 10),
        'clf__kernel': ('rbf', 'linear', 'poly')}
    
    decTreeParam = {
        "vect__ngram_range": ((1, 1), (1, 2)),
        "tfidf__use_idf": (True, False),
        "tfidf__norm": ("l1", "l2"),
        "clf__max_depth": (np.arange(2, 10, 1))
    }
    
    randForestParam = {
        "vect__ngram_range": ((1, 1), (1, 2)),
        "tfidf__use_idf": (True, False),
        "tfidf__norm": ("l1", "l2"),
        "clf__max_features": (1, 5),
        "clf__n_estimators": (10, 100, 1000)
    }
    
    x,y,xT,yT,xV,yV = getData()   

    runPipeline("Multinomial Naive Bayes", nbPipeline, mnbParameters, nbSmote, x,y,xT,yT,xV,yV)
    runPipeline("Support Vector Machine", svmPipeline, SvmSvcParameters, svmSmote, x,y,xT,yT,xV,yV)
    runPipeline("Decision Tree", decisionTreePipeline, decTreeParam, decisionTreeSmote, x,y,xT,yT,xV,yV)
    runPipeline("Random Forest", randomForestPipeline, randForestParam, randomForestSmote, x,y,xT,yT,xV,yV)
     
    print(f"\nRunning XGBoost...")
    print("XGBoost")
    xgBoostModel = xgBoostPipeline()
    xgBoostModel = xgBoostModel.fit(x,y)
    testModel(xgBoostModel, xT, yT, "XGBoost")
    rocAUC(xgBoostModel, xT, yT, "XGBoost")
    
    xgBoostKfold = xgBoostPipeline()
    kfold("XGBoost", xgBoostKfold, x,y)

def runQuickTest():
    """
    Quick test with just basic Naive Bayes
    """
    print("Running quick test...")
    x, y, xT, yT, xV, yV = getData()
    
    # Simple Naive Bayes
    model = nbPipeline()
    model.fit(x, y)
    testModel(model, xT, yT, "Quick NB Test")
    rocAUC(model, xT, yT, y, "Quick NB Test")
    
    print("\nK-fold validation:")
    modelKfold = nbPipeline()
    kfold("Quick NB Test", modelKfold, x, y)

# if __name__ == "__main__":
#     if len(sys.argv) > 1 and sys.argv[1] == "quick":
#         runQuickTest()
#     else:
#         runAll()

runAll()
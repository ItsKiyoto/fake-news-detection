import os
import sys
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split

from model import *
#from features import *
from preprocessing import *


def getData(data_dir="."):
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
        
        print("Processing training data...")
        x, y = read_dataset(trainDataset)
        print("Processing test data...")
        xT, yT = read_dataset(testDataset)
        print("Processing validation data...")
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

def getHyperparameters():
    """
    Define hyperparameters for each model
    """
    # Reduced parameter grids for faster tuning
    mnbParameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2'],
        'clf__alpha': [1e-2, 1e-3, 1e-1]
    }
        
    SvmSvcParameters = {        
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2'],
        'clf__C': [0.1, 1, 10],  # Reduced options
        'clf__gamma': ['scale', 'auto'],  # Simplified
        'clf__kernel': ['rbf', 'linear']  # Removed poly for speed
    }
    
    decTreeParam = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2'],
        'clf__max_depth': [3, 5, 7, 10, None],
        'clf__min_samples_split': [2, 5, 10]
    }
    
    randForestParam = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2'],
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [3, 5, 10, None],
        'clf__min_samples_split': [2, 5]
    }
    
    xgBoostParam = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2'],
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.1, 0.01],
        'clf__max_depth': [3, 5]
    }
    
    return {
        'mnb': mnbParameters,
        'svm': SvmSvcParameters,
        'dt': decTreeParam,
        'rf': randForestParam,
        'xgb': xgBoostParam
    }

def runSingleModel(model_name, model_pipeline, smote_pipeline, hyperparams, x, y, xT, yT, xV, yV):
    """
    Run a single model with all variations
    """
    try:
        runPipeline(model_name, model_pipeline, hyperparams, smote_pipeline, x, y, xT, yT, xV, yV)
    except Exception as e:
        print(f"Error running {model_name}: {e}")

def runAll():
    """
    Run all models with hyperparameter tuning and SMOTE
    """
    print("Starting Fake News Detection Model Training...")
    print("="*60)
    
    # Load data
    x, y, xT, yT, xV, yV = getData("Data")
    
    # Get hyperparameters
    hyperparams = getHyperparameters()
    
    # Get text statistics
    print("\nTraining Data Statistics:")
    get_text_statistics(x)
    
    # Define models to run
    models_to_run = [
        ("Multinomial Naive Bayes", nbPipeline, nbSmote, hyperparams['mnb']),
        ("Support Vector Machine", svmPipeline, svmSmote, hyperparams['svm']),
        ("Decision Tree", decisionTreePipeline, decisionTreeSmote, hyperparams['dt']),
        ("Random Forest", randomForestPipeline, randomForestSmote, hyperparams['rf']),
        # ("XGBoost", xgBoostPipeline, xgBoostSmote, hyperparams['xgb']),  # Uncomment if needed
    ]
    
    # Run each model
    for model_name, model_pipeline, smote_pipeline, model_hyperparams in models_to_run:
        try:
            runSingleModel(model_name, model_pipeline, smote_pipeline, model_hyperparams, 
                          x, y, xT, yT, xV, yV)
        except KeyboardInterrupt:
            print(f"\nTraining interrupted by user during {model_name}")
            break
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Training Complete!")

def runQuickTest():
    """
    Run a quick test with just Naive Bayes for debugging
    """
    print("Running quick test...")
    x, y, xT, yT, xV, yV = getData()
    
    # Simple Naive Bayes without hyperparameter tuning
    model = nbPipeline()
    model.fit(x, y)
    testModel(model, xT, yT, "Quick NB Test")
    rocAUC(model, xT, yT, "Quick NB Test")

if __name__ == "__main__":
    # You can switch between runAll() and runQuickTest() for debugging
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        runQuickTest()
    else:
        runAll()

# def runAll():
#     mnbParameters = {
#         'vect__ngram_range': [(1, 1), (1, 2)],
#         'tfidf__use_idf': (True, False),
#         "tfidf__norm": ("l1", "l2"), # maybe remove this
#         'clf__alpha': (1e-2, 1e-3)
#     }
        
#     SvmSvcParameters = {        
#         "vect__ngram_range": ((1, 1), (1, 2)),
#         "tfidf__use_idf": (True, False),
#         "tfidf__norm": ("l1", "l2"),
#         'clf__C': (0.1, 1, 10, 100),
#         'clf__gamma': (0.01, 0.1, 1, 10),
#         'clf__kernel': ('rbf', 'linear', 'poly')}
    
#     decTreeParam = {
#         "vect__ngram_range": ((1, 1), (1, 2)),
#         "tfidf__use_idf": (True, False),
#         "tfidf__norm": ("l1", "l2"),
#         "clf__max_depth": (np.arange(2, 10, 1))
#     }
    
#     randForestParam = {
#         "vect__ngram_range": ((1, 1), (1, 2)),
#         "tfidf__use_idf": (True, False),
#         "tfidf__norm": ("l1", "l2"),
#         "clf__max_features": (1, 5),
#         "clf__n_estimators": (10, 100, 1000)
#     }
    
#     x,y,xT,yT,xV,yV = getData()   

#     runPipeline("Multinomial Naive Bayes", nbPipeline, mnbParameters, nbSmote, x,y,xT,yT,xV,yV)
#     # runPipeline("Support Vector Machine", svmPipeline, SvmSvcParameters, svmSmote, x,y,xT,yT,xV,yV)
#     # runPipeline("Decision Tree", decisionTreePipeline, decTreeParam, decisionTreeSmote, x,y,xT,yT,xV,yV)
#     # runPipeline("Random Forest", randomForestPipeline, randForestParam, randomForestSmote, x,y,xT,yT,xV,yV)
     
#     # print()
#     # print("XGBoost")
#     # xgBoostModel = xgBoostPipeline()
#     # xgBoostModel = xgBoostModel.fit(x,y)
#     # testModel(xgBoostModel, xT, yT, "XGBoost")
#     # rocAUC(xgBoostModel, xT, yT, y, "XGBoost")
    
#     # xgBoostKfold = xgBoostPipeline()
#     # kfold("XGBoost", xgBoostKfold, x,y)

# runAll()
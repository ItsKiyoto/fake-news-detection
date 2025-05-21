from src.model import *
from src.features import *
from src.preprocessing import *

def getData():
    # retrieve dataset from csv files
    trainDataset = read_csv("/Users/Kishan/Documents/Kishan/Repositories/fake-news-detection/Final Year Project Software/Data/Train csv_version.csv", header=None)
    testDataset = read_csv("/Users/Kishan/Documents/Kishan/Repositories/fake-news-detection/Final Year Project Software/Data/Test csv_version.csv", header=None)
    validDataset = read_csv("/Users/Kishan/Documents/Kishan/Repositories/fake-news-detection/Final Year Project Software/Data/Valid csv_version.csv", header=None)

    x,y = read_dataset(trainDataset)
    xT, yT = read_dataset(testDataset)
    xV, yV = read_dataset(validDataset)

    return x,y,xT,yT,xV,yV

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
     
    print()
    print("XGBoost")
    xgBoostModel = xgBoostPipeline()
    xgBoostModel = xgBoostModel.fit(x,y)
    testModel(xgBoostModel, xT, yT, "XGBoost")
    rocAUC(xgBoostModel, xT, yT, y, "XGBoost")
    
    xgBoostKfold = xgBoostPipeline()
    kfold("XGBoost", xgBoostKfold, x,y)

runAll()


import numpy as np
import pandas as pd


def read_dataset(df):
    '''
    Clean and preprocess the dataset
    '''
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
    
    # Drop columns that are not helpful.
    df = df.drop(['Barely True Counts',
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

    print(f"\nDataset shape after cleaning: {df.shape}")
    print(f"Label distribution:\n{df['Label'].value_counts()}")

    df['Combined Data'] = (
        df['Statement'].fillna('').astype(str) + '\n' + 
        df['Subject'].fillna('').astype(str) + '\n' + 
        df['Speaker'].fillna('').astype(str) + '\n' + 
        df['Party'].fillna('').astype(str) + '\n' + 
        df['Occupation'].fillna('').astype(str) + '\n' + 
        df['Context'].fillna('').astype(str)
    )

    
    X = df['Combined Data']
    Y = df['Label']
    
    Y = TwoClassConversion(Y)

    print(f"\nFinal label distribution:\n{Y.value_counts()}")

    return X, Y

# Class Conversion Method
def TwoClassConversion(outputs):
    outputs = outputs.replace(["barely-true","pants-fire"], "FALSE")
    outputs = outputs.replace(["mostly-true"], "TRUE")
    return outputs

def get_text_statistics(X):
    """
    Get basic statistics about the text data
    """
    word_counts = X.apply(lambda x: len(str(x).split()))
    char_counts = X.apply(lambda x: len(str(x)))
    
    print(f"\nText Statistics:")
    print(f"Average words per document: {word_counts.mean():.2f}")
    print(f"Average characters per document: {char_counts.mean():.2f}")
    print(f"Min words: {word_counts.min()}, Max words: {word_counts.max()}")
    
    return word_counts, char_counts

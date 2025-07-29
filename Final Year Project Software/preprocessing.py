import re
import spacy
import string
import pandas as pd

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please install it using:")
    print("python -m spacy download en_core_web_sm")
    nlp = None

#Dataset Cleaning Method
def read_dataset(df):
    # Add labels to all the columnns 
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

    for column in df.columns:
        df['Cln_' + column] = clean_data(df[column].to_string())

    df['Combined Data'] = (
        df['Cln_Statement'] + ' ' +
        df['Cln_Subject'] + ' ' + 
        df['Cln_Speaker'] + ' ' + 
        df['Cln_Location'] + ' ' + 
        df['Cln_Party'] + ' ' + 
        df['Cln_Occupation'] + ' ' + 
        df['Cln_Context']
    )

    # combines all the different text information into one string.
    # df['Combined Data'] = df['Statement'] + '\n' + df['Subject'] + '\n' + df['Speaker'] + '\n' + df['Location'] + '\n' + df['Party'] + '\n' + df['Occupation'] + '\n' + df['Context']

    X = df['Combined Data']
    Y = df['Label']
    
    Y = TwoClassConversion(Y)

    return X, Y

def clean_data(text):
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower().replace("\n", " ").replace("\r", " ")

    text = re.sub(r'\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\s+', ' ', text).strip()
    
    # doc = nlp(text)
    # tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and len(token) > 2]

    if nlp is not None:
        try:
            doc = nlp(text)
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and token.is_alpha and len(token) > 2]
            return " ".join(tokens)
        except:
            # Fallback to basic cleaning if spacy fails
            pass
    
    # Basic fallback cleaning without spacy
    words = text.split()
    # Basic stop words (you could expand this list)
    stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'an', 'be', 'or', 'by'}
    cleaned_words = [word for word in words if len(word) > 2 and word not in stop_words]

    return " ".join(cleaned_words)

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

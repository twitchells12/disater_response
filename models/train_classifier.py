import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
import sqlite3
import nltk
import re
import pickle

nltk.download(['punkt', 'stopwords'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import warnings
warnings.simplefilter('ignore')

def load_data(db_filepath):
    """ Load data from SQL database and split into X and y

    Args:
    db_filepath: SQLite databse file contained message databse

    Returns:
    X = Features DataFrame
    Y = Labels Dataframe
    """
    # load data from database
    engine = create_engine('sqlite:///' + db_filepath)

    inspector = inspect(engine)
    # Get table information
    print('tables in the db {}'.format(inspector.get_table_names()))

    df = pd.read_sql("SELECT * FROM Messages ", engine)

    # create X and Y datasets
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)

    # create a list of cat names
    category_names = list(Y.columns.values)

    return X, Y, category_names

def tokenize(text):
    """ Tokenize and stem text strings

    Args:
    text: Message data used for processing

    Returns:
    stemmed: List of normalized and stemmed strings
    """
    # convert text to lowercase and remove puctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())

    # tokenize the words
    tokens = word_tokenize(text)

    #stem and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words('english')

    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return stemmed



def build_model():
    """ Builds an sklearn machine learning pipeline

    Args:
    None

    Returns:
    cv: a gridsearchcv object. The gridsearchcv object transforms the data, finds
    the optimal model parameters.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize, min_df = 5)),
        ('tfidf', TfidfTransformer(use_idf = True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10,
                                                             min_samples_split = 10)))
    ])

    # Create parameters dictionary
    parameters = {'vect__min_df': [1, 5],
                  'tfidf__use_idf':[True, False],
                  'clf__estimator__n_estimators':[10, 25],
                  'clf__estimator__min_samples_split':[2, 5, 10]}

    # create grid search
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluates the model
    Args:
    model: machine learning model
    X-test: feature dataset for testeing
    Y-test: label dataset for testing
    category_names: category Labels

    Returns:
    None
    """
    # predicts the labels for the test dataset
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_path):
    """ Save machine learing model to a pikle file

    Args:
    model: machine learning model
    model_path: file to save model

    Returns:
    None
    """
    pickle.dump(model.best_estimator_,open(model_path,'wb'))


def main():
    db_filepath,model_path = sys.argv[1:]
    print(db_filepath)
    X, y , cat_names= load_data(db_filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

    print('buiding the model...')
    # model = build_model()
    print('training the model...')
    # model.fit(X_train, y_train)

    print('evaluating model...')
    evaluate_model(model,X_test,y_test,cat_names)

    print('saving the model...')
    save_model(model,model_path)

    print('model saved')

if __name__ == '__main__':
    main()

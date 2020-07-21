import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_path,cat_path):
    """Load merge messages and category datasets

    Args:
    messages_path: messages filepath
    cat_path: category filepath

    Returns:
    df: Dataframe of merged messages and cagetories
    """
    # load datasets
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(cat_path)
    # merge datasets
    df = messages.merge(categories,on='id')

    return df


def clean_data(df):
    """Clean dataframe by removing duplicates and converting categories from strings
    to binary values.

    Args:
    df:  Merged messages and categories dataframe.

    Returns:
    df: Cleaned version of input dataframe.
    """
    categories = df.categories.str.split(pat=';',expand=True)

    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: str(x)[:-2])

    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    # print (df.head())
    return df


def save_data(df,db_filename):
    """Save cleaned dataframe to an SQL database

    Args:
    df: Cleaned Dataframe
    db_filename: Output database filename

    Returns:
    None
    """
    engine = create_engine('sqlite:///' + db_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')
    df.to_csv('messages_clean.csv')

def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

# %% Imports
import importlib
import logging
import os
import re
import sys
import traceback
from contextlib import contextmanager  # To make a cd function

import matplotlib.pyplot as plt
import numpy as np  # Arrays
import pandas as pd  # Dataframes
from IPython.display import display
from tqdm import tqdm  # Progress bar

# Pandas fancy tables
pd.set_option('display.notebook_repr_html', True)
pd.set_option('max_rows', 10)
# Logger setup
importlib.reload(logging)
logging.basicConfig(format='%(levelname)s | %(asctime)s | %(message)s',
                    level=logging.INFO, stream=sys.stdout, datefmt='%H:%M:%S')
# Numpy printing setup
np.set_printoptions(threshold=10, linewidth=79, edgeitems=5)

# %% Making some helper functions


@contextmanager
def cd(dir):  # Must be a generator
    '''
        Easy to use 'cd' function, which goes back after exiting a 'with' statement.
        Arguments:
            dir (str): the directory you want to cd to
        Examples:
            with cd('mydir'):
                do_something() # Working in the mydir
            do_something() # Working in the previous dir
    '''
    old_dir = os.getcwd()
    os.chdir(os.path.join(old_dir, dir))
    try:
        yield
    finally:
        os.chdir(old_dir)


def list_dir(path=None):
    '''Lists the items in the directory, but ommits hidden files.
        Arguments:
            path (str, optional): path to a directory you want to list
        Examples:
        >>> print(list_dir()) # Omits files that start with '.'
        ['folder', 'file.ext']
    '''
    return [item for item in os.listdir(path) if not item.startswith('.')]


# WORD SELECTION SETUP
# Only processing 3+ letter words
WORD_PATTERN = re.compile('(?:[a-z][a-z][a-z]+)')
# Removing "Key: value" stuff
HEADER_PATTERN = re.compile('^[A-Z][a-z]+([- ]\w+){0,2}:.+$')
EMAIL_PATTERN = re.compile('(?:\w+?@\w+?\.\w+?)')
WEBSITE_PATTERN = re.compile('((?:http://|https://){0,1}(?:www.){0,1}(?:[a-z0-9_]+?\.)'
                             '(?:[a-z0-9_]+(?:\.){0,1}){1,3})')


def get_word_dict(filename):
    '''
    Counts the occurances of a word in a file given filename
    Arguments:
        filename (str): The name or path of a file you want to process
    Returns:
        dict: The {word: occurences} dictionary for that file
    Examples:
        >>> counts = get_word_dict('to_process.txt')
        {'word1': 1, 'word2': 2, }
    '''
    return_dict = {}
    with open(filename, 'r', errors='ignore') as f:
        linelist = list(f)
    cleaned_linelist = [
        WEBSITE_PATTERN.sub('', EMAIL_PATTERN.sub('', line.lower().strip()))
        for line in linelist
        if not line.startswith('*') and not HEADER_PATTERN.match(line)
    ]
    for line in cleaned_linelist:
        for word in WORD_PATTERN.findall(line):
            try:
                return_dict[word] += 1
            except KeyError:  # First occurence of a word
                return_dict[word] = 1
    return return_dict


def get_word_dict_proper(filename, proper_words):
    '''
    Counts the occurances of a word in a file given filename
    Arguments:
        filename (str): The name or path of a file you want to process
        proper_words (list): The list of the words that we want to select
    Returns:
        dict: The {word: occurences} dictionary for that file
    Examples:
        >>> counts = get_word_dict_proper('to_process.txt', popular_words)
        {'word1': 1, 'word2': 2, }
    '''
    return_dict = {word: 0 for word in proper_words}
    with open(filename, 'r', errors='ignore') as f:
        linelist = list(f)
    cleaned_linelist = [
        EMAIL_PATTERN.sub('', line).lower().strip() for line in linelist
        if not line.startswith('*') and not HEADER_PATTERN.match(line)
    ]
    for line in cleaned_linelist:
        for word in WORD_PATTERN.findall(line):
            try:
                return_dict[word] += 1
            except KeyError:  # We don't care about other words
                pass
    return return_dict


def generate_classes(folder_name):
    '''
        Generates the class dictionary given a folder_name.
        Arguments:
            folder_name (str): The name or path of a data folder you want to process
        Returns:
            dict: A dictionary {class_name: class_value}
        Examples:
            >>> class_dict = generate_classes('train')
            {'sci.electronics':0, }
    '''
    return_dict = {}
    i = 0
    with cd(folder_name):
        for data_class in [item for item in list_dir() if os.path.isdir(item)]:
            return_dict[data_class] = i
            i += 1
    return return_dict


# Word column names in the dataframe
def check_df(dataframe, class_dict, df_type='train', coverage_pct=.05):
    '''
        Checks whether a dataframe is valid based on random selection of rows
        Arguments:
            dataframe (pd.DataFrame): The dataframe to be checked
            class_dict (dict): The {'class_name': class_number} dictionary
            df_type (string): 'test' or 'train'
            coverage_pct (float): The percentage of rows to check
        Returns:
            bool: True if it's valid, False if it's not
        Examples:
            >>> is_valid = check_df(my_df, class_dict, rows_to_check=10, coverage_pct=0.25)
            True
    '''
    word_columns = dataframe.columns.tolist()[2:]  # This is faster than regex
    class_dict_rev = {v: k for k, v in class_dict.items()}
    try:
        with cd(os.path.join('data', df_type)):
            for idx, row in dataframe.sample(
                    int(coverage_pct * dataframe.shape[0])).iterrows():
                with cd(class_dict_rev[row['prediction_class']]):
                    counts = get_word_dict(str(row['document_id']))
                    for word in word_columns:
                        if counts[word] != row[word]:
                            logging.error(
                                'Word {} appears in the file {} times, and is in the df {} times'
                                .format(word, counts[word], row[word]))
                            logging.error('Row:\n{}'.format(row))
                            return False
    except KeyError:
        pass
    except Exception:
        logging.error(traceback.format_exc())
        logging.error('Row:\n{}'.format(row))
        return False
    return True


def get_total_word_count(folder_name):
    '''
        Checks whether a dataframe is valid based on random selection of rows
        Arguments:
            folder_name (str): The class folder you want to parse
        Returns:
            bool: Dictionary of word counts in all of the files.
                Dict structure: {'word': (total_occurances, total_occured_files)}
        Examples:
            >>> twc = get_total_word_count('test_folder')
            {'the': (1005, 21), }
    '''
    twc_dict = {}
    logging.info('Getting the total word count in {}'.format(folder_name))
    with cd(folder_name):
        for data_class in tqdm(
            [item for item in list_dir()
             if os.path.isdir(item)], desc='Classes processed', leave=False,
                file=sys.stdout, unit='class'):
            logging.debug('get_total_word_cound | Processing class "{}"...'
                          .format(data_class))
            with cd(data_class):
                for filename in [
                        item for item in list_dir() if os.path.isfile(item)
                ]:
                    temp_word_dict = get_word_dict(filename)
                    # twc_dict: {'word': [total_count, occured_in_N_files]}
                    for word, count in temp_word_dict.items():
                        try:
                            twc_dict[word][0] += count
                            twc_dict[word][1] += 1
                        except KeyError:  # First occurence of a word
                            twc_dict[word] = [count, 1]
    return twc_dict


def process_data_folder(folder_name, class_dict, popular_words):
    '''
    Counts the occurances of a word in a file given filename
    Arguments:
        folder_name (str): The name or path of a data folder you want to process
        class_dict (dict): A dictionary of classes {class_name: class_number}
        popular_words (list<string>): A list of words to count
    Returns:
        np.array(str): The column names for every file
        np.array(np.array(int)): The array of values corresponding to the column names
    '''
    # We have the same return_cols for every file
    return_cols = np.array(['document_id', 'prediction_class'] +
                           list(popular_words), dtype=np.dtype('str'))
    return_vals = []
    logging.info('Parsing the folder {}'.format(folder_name))
    with cd(folder_name):
        for data_class in tqdm(
            [item for item in list_dir()
             if os.path.isdir(item)], desc='Classes processed', leave=False,
                file=sys.stdout, unit='class'):
            logging.debug('process_data_folder | Processing class "{}"...'
                          .format(data_class))
            with cd(data_class):
                for filename in [
                        item for item in list_dir() if os.path.isfile(item)
                ]:
                    temp_word_dict = get_word_dict_proper(
                        filename, popular_words)
                    # Creating an array of values
                    return_vals.append(
                        np.array([filename, class_dict[data_class]] + list(
                            temp_word_dict.values()), dtype=np.dtype('int')))
    return (np.array(return_cols), np.array(return_vals))


def top_rows(word_df, num_rows, add_counts=False):
    if add_counts:
        return pd.DataFrame(
            data=(
                (word_df[:NUMBER_TOP_WORDS])
                .reset_index(drop=True)  # Resetting index to be tidy
            )
        )
    return pd.DataFrame(
        data=(
            (word_df[:NUMBER_TOP_WORDS])['words']
            .reset_index(drop=True)  # Resetting index to be tidy
        )
    )


# %% Reading the files - getting the popular words
if 'data' not in list_dir():
    sys.exit('Could not find the data folder. '
             'Make sure that the notebook is run from the "News-Learning" dir.')

if 'test' not in list_dir('data') or 'train' not in list_dir('data'):
    sys.exit('Bad structure of the data folder. '
             'Make sure that the data folder contains "test" and "train" dirs.')

with cd('data'):
    logging.info('Generating classes...')
    class_dict = generate_classes('train')

    logging.info('Getting the total word counts...')
    twc_train = get_total_word_count('train')

# %% Transforming the data - selecting the most common words
NUMBER_TOP_WORDS = 10_000

# Making the 2d array from a dictionary
wordlist = [[w, v[0], v[1]] for (w, v) in twc_train.items()]
word_arr = np.stack(tuple(wordlist), axis=0)  # vertical stack

# Making a df out of it
word_df = pd.DataFrame(data=word_arr, columns=('words', 'count', 'count_files'))

# Change the cell types so we can properly compare them
word_df['count'] = pd.to_numeric(word_df['count'])
word_df['count_files'] = pd.to_numeric(word_df['count_files'])

# Sort the values
word_df.sort_values(by=['count', 'count_files'], axis=0, inplace=True,
                    ascending=False)
word_df.reset_index(inplace=True, drop=True)

# %% Visualizing the data
every_other = 1  # taking every element

temp_df = word_df.iloc[:, 1:3]
display(temp_df)
plt.show(temp_df.plot.scatter(x='count', y='count_files'))

temp_df = temp_df[:NUMBER_TOP_WORDS]
display(temp_df)
plt.show(temp_df.plot.scatter(x='count', y='count_files'))

# %%Resulting dataframe
logging.info(f'Selecting {NUMBER_TOP_WORDS} most popular words:')
uwords_df = top_rows(word_df, NUMBER_TOP_WORDS, False)
logging.info('Final dataframe')
display(uwords_df)

# %% Getting the data - Parsing train and test folders

with cd('data'):
    logging.info('Getting the train data...')
    train_cols, train_vals = process_data_folder('train', class_dict,
                                                 uwords_df.values[:, 0])
    logging.info('Getting the test data...')
    test_cols, test_vals = process_data_folder('test', class_dict,
                                               uwords_df.values[:, 0])

logging.info('Train data sample:')
logging.info('Columns:')
display(pd.DataFrame(train_cols).T)
logging.info('Values:')
display(pd.DataFrame(train_vals[0]).T)

logging.info('Test data sample:')
logging.info('Columns:')
display(pd.DataFrame(test_cols).T)
logging.info('Values:')
display(pd.DataFrame(test_vals[0]).T)

# %% Transforming the data - Making dataframes

# Train
logging.info('Making the train dataframe...')
train_df = pd.DataFrame(data=np.stack(train_vals), columns=train_cols)

# Test
logging.info('Making the test dataframe...')
test_df = pd.DataFrame(data=np.stack(test_vals), columns=test_cols)

# Checking
PCT_TO_CHECK = 0.10

logging.info(
    'Checking the train_df ({}/{} rows): {}!'.format(
        int(PCT_TO_CHECK * train_df.shape[0]),
        train_df.shape[0],
        'Passed the check'
        if check_df(train_df, class_dict, coverage_pct=PCT_TO_CHECK) else
        'Didn\'t pass the check!',
    ),
)

logging.info('Checking the test_df ({}/{} rows): {}!'.format(
    int(PCT_TO_CHECK * test_df.shape[0]),
    test_df.shape[0],
    'Passed the check'
    if check_df(test_df, class_dict, df_type='test', coverage_pct=PCT_TO_CHECK)
    else 'Didn\'t pass the check!',
))


# Final transforms
logging.info('Final train DF:')
train_df = train_df.iloc[:, 1:]
display(train_df)

logging.info('Final test DF:')
test_df = test_df.iloc[:, 1:]
display(test_df)

# %% Making a class name df
class_df = pd.DataFrame(data=[name for name, _ in class_dict.items()],
                        index=[val for _, val in class_dict.items()],
                        columns=['class_name'])

display(class_df)
# %% Saving the results

# Change this at will
options = {
    'save': True,
    'archive': True,
    'save_folder': 'dataframes',
    'train_fname': 'train_df',
    'test_fname': 'test_df',
    'class_fname': 'class_df'
}

if options['save']:
    # If there's no save folder, make it
    if not os.path.isdir(options['save_folder']):
        os.makedirs(options['save_folder'])

    # Generating options that we can pass in a df's save function
    options = {
        'train': {
            'path_or_buf':
                os.path.join(options['save_folder'],
                             (('{}.csv.gz' if options['archive'] else '{}.csv')
                              .format(options['train_fname']))),
            'compression':
                'gzip' if options['archive'] else None
        },
        'test': {
            'path_or_buf':
                os.path.join(options['save_folder'],
                             (('{}.csv.gz' if options['archive'] else '{}.csv')
                              .format(options['test_fname']))),
            'compression':
                'gzip' if options['archive'] else None
        },
        'class': {
            'path_or_buf':
                os.path.join(options['save_folder'],
                             (('{}.csv.gz' if options['archive'] else '{}.csv')
                              .format(options['class_fname']))),
            'compression':
                'gzip' if options['archive'] else None
        }
    }

    # Class df
    logging.info(
        f'Saving the Class DF to "{options["class"]["path_or_buf"]}"...')
    class_df.to_csv(**options['class'])
    logging.info('Finished saving the class DF')

    # Train df
    logging.info(
        f'Saving the train DF to "{options["train"]["path_or_buf"]}"...'
    )
    train_df.to_csv(**options['train'])
    logging.info('Finished saving the train DF')

    # Test df
    logging.info(f'Saving the test DF to "{options["test"]["path_or_buf"]}"...')
    test_df.to_csv(**options['test'])
    logging.info('Finished saving the test DF')

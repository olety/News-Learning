# %% Imports
import importlib
import logging
import os
import re
import sys
from contextlib import contextmanager  # To make a cd function

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
np.set_printoptions(threshold=10, linewidth=79, edgeitems=20)

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


def list_dir():
    '''Lists the items in the directory, but ommits hidden files.'''
    return [item for item in os.listdir() if not item.startswith('.')]


# WORD SELECTION SETUP
# Only processing 3+ letter words
WORD_PATTERN = re.compile('[a-z][a-z][a-z]+')
# Removing "Key: value" stuff
HEADER_PATTERN = re.compile('^[A-Z][a-z]+([- ]\w+){0,2}:.+$')
EMAIL_PATTERN = re.compile('\w+?@.+?')


def get_word_dict(filename):
    '''
    Counts the occurances of a word in a file given filename
    Arguments:
        filename (str): The name or path of a file you want to process
    Returns:
        dict: The {word: occurences} dictionary for that file
    Examples:
        counts = get_word_dict('to_process.txt')
    '''
    return_dict = {}
    with open(filename, 'r', errors='ignore') as f:
        linelist = list(f)
    cleaned_linelist = [EMAIL_PATTERN.sub('', line).lower().strip() for line in linelist
                        if not line.startswith('*') and not HEADER_PATTERN.match(line)]
    for line in cleaned_linelist:
        for word in WORD_PATTERN.findall(line):
            try:
                return_dict[word] += 1
            except KeyError:  # First occurence of a word
                return_dict[word] = 1
    return return_dict


def generate_classes(folder_name):
    '''
        Generates the class dictionary given a folder_name.
        Arguments:
            folder_name (str): The name or path of a data folder you want to process
        Returns:
            dict: A dictionary {class_name: class_value}
        Examples:
            class_dict = generate_classes('train') # {'sci.electronics':0, }
    '''
    return_dict = {}
    i = 0
    with cd(folder_name):
        for data_class in [item for item in os.listdir() if os.path.isdir(item)]:
            return_dict[data_class] = i
            i += 1
    return return_dict


def process_data_folder(folder_name, class_dict, total_word_count=False):
    '''
    Counts the occurances of a word in a file given filename
    Arguments:
        folder_name (str): The name or path of a data folder you want to process
        class_dict (dict): A dictionary of classes {class_name: class_number}
        total_word_count (bool, optional): A flag that determines whether you want to
            return an additional parameter. Defaults to False.
    Returns:
        np.array: The column names for every processed file in the class folder
        np.array: The values corresponding to the column names
        dict (optional): A dictionary {word_name: (occurence_count, file_count)}.
            Is only returned when the total_word_count = True
    Examples:
        class_dict = generate_classes('train')
        train_cols, train_vals, twc_train =
            process_data_folder('train', class_dict, total_word_count=True)
    '''
    return_cols = []
    return_vals = []
    if total_word_count:
        # For selecting k top words
        twc_dict = {}
    with cd(folder_name):
        for data_class in tqdm([item for item in os.listdir() if os.path.isdir(item)],
                               desc='Classes processed', leave=False,
                               file=sys.stdout, unit='classes'):
            logging.debug('Processing class "{}"...'.format(data_class))
            with cd(data_class):
                for filename in [item for item in os.listdir() if os.path.isfile(item)]:
                    temp_word_dict = get_word_dict(filename)
                    # Creating an array of column names
                    return_cols.append(np.array(['document_id',
                                                 'prediction_class'] +
                                                list(temp_word_dict.keys()),
                                                dtype=np.dtype('str')))
                    # Creating an array of values
                    return_vals.append(np.array([filename,
                                                 class_dict[data_class]] +
                                                list(temp_word_dict.values()),
                                                dtype=np.dtype('int')))
                    if total_word_count:
                        # twc_dict: {'word': [total_count, occured_in_N_files]}
                        for word, count in temp_word_dict.items():
                            try:
                                twc_dict[word][0] += count
                                twc_dict[word][1] += 1
                            except KeyError:  # First occurence of a word
                                twc_dict[word] = [count, 1]
    if total_word_count:
        return (np.array(return_cols),
                np.array(return_vals),
                twc_dict)
    return (np.array(return_cols),
            np.array(return_vals))


# %% Reading the files


if 'data' not in os.listdir():
    sys.exit('Could not find the data folder. '
             'Make sure that the notebook is run from the "News-Learning" dir.')

if 'test' not in os.listdir('data') or 'train' not in os.listdir('data'):
    sys.exit('Bad structure of the data folder. '
             'Make sure that the data folder contains "test" and "train" dirs.')

# if 'data.npy' in os.listdir():
#     # TODO: Implement reading data from a file as an arg
#     sys.exit(0)


with cd('data'):
    logging.info('Generating classes...')
    class_dict = generate_classes('train')

    logging.info('Processing train data...')
    train_cols, train_vals, twc_train = process_data_folder(
        'train', class_dict, total_word_count=True)

    logging.info('Processing test data...')
    test_cols, test_vals = process_data_folder('test', class_dict)

logging.info('Train data sample:')
logging.info('Columns:')
display(pd.DataFrame(train_cols[0]))
logging.info('Values:')
display(pd.DataFrame(train_vals[0]))

logging.info('Test data sample:')
logging.info('Columns:')
display(pd.DataFrame(test_cols[0]))
logging.info('Values:')
display(pd.DataFrame(test_vals[0]))


# %% Transforming the data - selecting the dataframes with k top words

NUMBER_TOP_WORDS = 10_000

# Making the 2d array
wordlist = [[w, v[0], v[1]] for (w, v) in twc_train.items()]
word_arr = np.vstack(tuple(wordlist))

# Making a df out of it
word_df = pd.DataFrame(data=word_arr,
                       columns=('words', 'count', 'count_files'))

# Change the cell types so we can properly compare them
word_df['count'] = pd.to_numeric(word_df['count'])
word_df['count_files'] = pd.to_numeric(word_df['count_files'])

# Sort the values
word_df.sort_values(by=['count', 'count_files'],
                    axis=0, inplace=True, ascending=False)

# Selecting the top words
word_df = word_df[:NUMBER_TOP_WORDS]

logging.info('Selected {} most popular words:'.format(NUMBER_TOP_WORDS))
display(word_df)


# Resetting index to be tidy
uwords_series = word_df['words'].reset_index(drop=True)

# Resulting dataframe
uwords_df = pd.DataFrame(data=uwords_series)
logging.info('Final dataframe')
display(uwords_df)


# %% Transforming the data - Making dataframes

# Used in the transformations
column_headers = np.concatenate((
    np.array(['document_id',
              'prediction_class']),
    uwords_df.T.values[0])
).astype(np.dtype('str'))
# Used in the resulting dataframes
column_headers_sorted = np.sort(column_headers)

# Numpy processing method gave ~10-100x speed boost compared to the dataframe one
# (from 1-10 DFs per second to ~100 arrays per second)

# Processing the train data
train_arrs = []

total_len = len(train_cols)
iter_seq = zip(train_cols, train_vals)
logging.info('Transforming the train data into an arraylist...')
for col, val in tqdm(iter_seq, desc='Arrays processed', total=total_len, leave=True, file=sys.stdout, unit='arrs'):
    # First, sort the col/vals
    idx = np.argsort(col)
    col = col[idx]
    val = val[idx]
    # Get the set of extra words
    sd = np.setdiff1d(col, column_headers)
    # Find the extra words in cols
    # We sorted them in the start to use this function
    del_index = np.searchsorted(col, sd)
    # Get rid of the extra words
    col = np.delete(col, del_index)
    val = np.delete(val, del_index)
    # get the missing words
    sd = np.setdiff1d(column_headers, col)
    # horizontal stack with the set of words that we're missing
    col = np.concatenate((col, sd))
    val = np.concatenate((val, np.zeros(sd.shape)))
    # Argsort returns indices that sort an array
    # then we apply the column sort to the values to preserve order
    # (alphabetical)
    train_arrs.append(val[np.argsort(col)])

train_arrs = np.array(train_arrs, dtype='int')
logging.info('Transforming the train arraylist into a dataframe...')
train_df = pd.DataFrame(data=train_arrs, columns=column_headers_sorted)

# Processing the test data
test_arrs = []

total_len = len(test_cols)
iter_seq = zip(test_cols, test_vals)
logging.info('Transforming the test data into an arraylist...')
for col, val in tqdm(iter_seq, desc='Arrays processed', total=total_len, leave=True, file=sys.stdout, unit='arrs'):
    # First, sort the col/vals
    idx = np.argsort(col)
    col = col[idx]
    val = val[idx]
    # Get the set of extra words
    sd = np.setdiff1d(col, column_headers)
    # Find the extra words in cols
    # We sorted them in the start to use this function
    del_index = np.searchsorted(col, sd)
    # Get rid of the extra words
    col = np.delete(col, del_index)
    val = np.delete(val, del_index)
    # get the missing words
    sd = np.setdiff1d(column_headers, col)
    # horizontal stack with the set of words that we're missing
    col = np.concatenate((col, sd))
    val = np.concatenate((val, np.zeros(sd.shape)))
    # Argsort returns indices that sort an array
    # then we apply the column sort to the values to preserve order
    # (alphabetical)
    test_arrs.append(val[np.argsort(col)])

test_arrs = np.array(test_arrs, dtype='int')
logging.info('Transforming the test arraylist into a dataframe...')
test_df = pd.DataFrame(data=test_arrs, columns=column_headers_sorted)


# Checking the results
logging.info('Train DF:')
display(train_df)
logging.info('Test DF:')
display(test_df)

# %% Saving the results

# Change this at will
options = {
    'save': False,  # Hopefully this will make the notebook viewable on github
    'archive': True,
    'save_folder': 'dataframes',
    'train_fname': 'train_df',
    'test_fname': 'test_df'
}

if options['save']:
    # If there's no save folder, make it
    if not os.path.isdir(options['save_folder']):
        os.makedirs(options['save_folder'])

    # Generating options that we can pass in a df's save function
    options = {
        'train': {
            'path_or_buf': os.path.join(
                options['save_folder'],
                (
                    ('{}.csv.gz' if options['archive'] else '{}.csv')
                    .format(options['train_fname'])
                )
            ),
            'compression': 'gzip' if options['archive'] else None
        },

        'test': {
            'path_or_buf': os.path.join(
                options['save_folder'],
                (
                    ('{}.csv.gz' if options['archive'] else '{}.csv')
                    .format(options['test_fname'])
                )
            ),
            'compression': 'gzip' if options['archive'] else None
        }
    }

    # Train df
    logging.info('Saving the train DF to "{}"...'
                 .format(options['train']['path_or_buf']))
    train_df.to_csv(**options['train'])
    logging.info('Finished saving the train DF')

    # Test df
    logging.info('Saving the test DF to "{}"...'
                 .format(options['test']['path_or_buf']))
    train_df.to_csv(**options['test'])
    logging.info('Finished saving the test DF')

import importlib
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

# Pandas fancy tables
pd.set_option('display.notebook_repr_html', True)
# Logger setup
importlib.reload(logging)
logging.basicConfig(format='%(levelname)s | %(asctime)s | %(message)s',
                    level=logging.INFO, stream=sys.stdout, datefmt='%H:%M:%S')
# %% Loading dataframes
print('Loading dataframes...')
print('Loading train dataframe...')
train_df = pd.read_csv('dataframes/train_df.csv.gz', index_col=0)
print('Loading test dataframe...')
test_df = pd.read_csv('dataframes/test_df.csv.gz', index_col=0)
print('Finished loading dataframes!')

# %% Converting dfs to numpy arrays
print('Converting the dataframes...')
USE_ONEHOT = False
# Train data
# Shuffling the data
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

total_x = np.concatenate((train_df.iloc[:, 1:].as_matrix(),
                          test_df.iloc[:, 1:].as_matrix()))

total_y = np.concatenate((train_df.iloc[:, 0].as_matrix(),
                          test_df.iloc[:, 0].as_matrix()))
if USE_ONEHOT:
    temp_arr = []
    for class_ in total_y:
        zeros = [0] * 10
        zeros[class_] = 1
        temp_arr.append(zeros)
        total_y = np.array(temp_arr)

print(f'Train data:\n{total_x}\nTrain labels:\n{total_y}')

# %%


def make_classifiers():
    models = []
    models.append({'name': 'Logistic regression',
                   'clf': LogisticRegression(verbose=1, random_state=42)})
    models.append({'name': 'MLP NN lr=1e-4',
                   'clf': MLPClassifier(activation='relu', solver='adam', verbose=1, learning_rate='invscaling', random_state=42)})
    models.append({'name': 'MLP NN lr=1e-3',
                   'clf': MLPClassifier(activation='relu', solver='adam', verbose=1, learning_rate='invscaling', learning_rate_init=1e-3, random_state=42)})
    models.append({'name': 'MLP NN lr=1e-2',
                   'clf': MLPClassifier(activation='relu', solver='adam', verbose=1, learning_rate='invscaling', learning_rate_init=1e-2, random_state=42)})
    models.append({'name': 'Multinomial Naive Bayes',
                   'clf': MultinomialNB()})
    return models


# Feature selection
def make_feature_selectors():
    feature_selectors = []

    feature_selectors.append(
        {'name': 'Percentile selector (30%)',
         'selector': SelectPercentile(percentile=30)}
    )

    feature_selectors.append({
        'name': 'Model selector(random forest)',
        'selector': SelectFromModel(
            RandomForestClassifier(n_estimators=100),
            threshold="median")
    })

    feature_selectors.append(
        {'name': 'All features',
         'selector': SelectPercentile(percentile=100)}
    )
    return feature_selectors


results = []
# Running the loop
for select_dict in tqdm(make_feature_selectors(), desc='Feature selectors', unit='selector', file=sys.stdout):
    logging.info(f'Started processing feature_selector {select_dict["name"]}')
    logging.info('Selecting features...')
    select = select_dict['selector']
    select.fit(total_x, total_y)
    train_x = select.transform(total_x)
    logging.info('Finished selecting features!')
    for classifier in tqdm(make_classifiers(), desc='Models', unit='model', file=sys.stdout):
        logging.info(f'Started processing classifier {classifier["name"]}')
        # 10-fold cv
        scores = cross_val_score(
            classifier['clf'], train_x, total_y, cv=10, scoring='accuracy')
        logging.info(f'Scores vector: {scores}')
        results.append({'name': classifier['name'],
                        'score': scores.mean(),
                        'feature_selector': select_dict['name']})
        logging.info((f'[{classifier["name"]}] Score with only selected features for selector'
                      f'{select_dict["name"]}: {scores.mean()}'))

# joblib.dump(clf, 'filename.pkl')
pd.DataFrame(results).to_csv('res.csv')

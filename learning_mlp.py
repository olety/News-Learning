# %% Imports
import math

import mglearn
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.neural_network import MLPClassifier

import scikit_learn as sklearn

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

total_x = np.concatenate(train_df.iloc[:, 1:].as_matrix(),
                         test_df.iloc[:, 1:].as_matrix())

total_y = np.concatenate(train_df.iloc[:, 0].as_matrix(),
                         test_df.iloc[:, 0].as_matrix())
if USE_ONEHOT:
    temp_arr = []
    for class_ in total_y:
        zeros = [0] * 10
        zeros[class_] = 1
        temp_arr.append(zeros)
        total_y = np.array(temp_arr)

print(f'Train data:\n{total_x}\nTrain labels:\n{total_y}')


# Test data
test_x = test_df.iloc[:, 1:].as_matrix().astype(np.float32)
test_y = test_df.iloc[:, 0].as_matrix()
temp_arr = []
for class_ in test_y:
    zeros = [0] * 10
    zeros[class_] = 1
    temp_arr.append(zeros)
test_y = np.array(temp_arr, dtype=np.float32)
print(f'Test data:\n{test_x}\nTrain labels:\n{test_y}')
print('Finished converting')
total_batch = math.ceil(train_x.shape[0] / batch_size)
print(f'Total number of batches = {total_batch}')

# %% sklearn nns

mlp = MLPClassifier(algorithm='l-bfgs',
                    random_state=0).fit(train_x, train_y)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

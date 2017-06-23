# %% Imports
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display

# %% Basic setup & loading dataframes
print('Setting variables...')
learning_rates = [1e-05, 1e-04, 5e-04, 1e-03]
results = {'rate': 0, 'costs': [], 'test': -1.}
batch_size = 100
training_iteration = 50
display_step = 2
save_step = 1
method = 'logreg'
print('Finished setting...')

# %% Loading dataframes
print('Loading dataframes...')
print('Loading train dataframe...')
train_df = pd.read_csv('dataframes/train_df.csv.gz', index_col=0)
print('Loading test dataframe...')
test_df = pd.read_csv('dataframes/test_df.csv.gz', index_col=0)
print('Finished loading dataframes!')

# %% Converting dfs to numpy arrays
print('Converting the dataframes...')

# Train data

train_df = train_df.sample(frac=1).reset_index(drop=True)
train_x = train_df.iloc[:, 1:].as_matrix().astype(np.float32)
train_y = train_df.iloc[:, 0].as_matrix()
temp_arr = []
for class_ in train_y:
    zeros = [0] * 10
    zeros[class_] = 1
    temp_arr.append(zeros)
train_y = np.array(temp_arr, dtype=np.float32)
print(f'Train data:\n{train_x}\nTrain labels:\n{train_y}')


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


# %% tf setup & run - Logistic regression
for learning_rate in learning_rates:
    # Setup
    results['rate'] = learning_rate
    print('Starting tf setup...')
    x = tf.placeholder(tf.float32, [None, 10_000], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')

    # Weights
    W = tf.Variable(tf.zeros([10_000, 10]))
    b = tf.Variable(tf.zeros([10]))

    with tf.name_scope('Wx_b') as scope:
        # Softmax is used for multiclass
        model = tf.nn.softmax(tf.matmul(x, W) + b)

    with tf.name_scope('cost_function') as scope:
        cost_function = tf.losses.softmax_cross_entropy(
            onehot_labels=y, logits=model)

    with tf.name_scope('train') as scope:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(cost_function)

    init = tf.global_variables_initializer()
    print('Finished tf setup')

    # Run
    print(f'Starting tf training session, learning_rate={learning_rate}')
    with tf.Session() as sess:
        sess.run(init)
        print('Started iteration...')
        for iteration in range(training_iteration):
            avg_cost = 0.
            for i in range(total_batch):
                batch_xs = train_x[i * batch_size:(i + 1) * batch_size]
                batch_ys = train_y[i * batch_size:(i + 1) * batch_size]
                _, c = sess.run([optimizer, cost_function],
                                feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += c / total_batch
            if iteration % save_step == 0:
                results['costs'].append(avg_cost)
            if iteration % display_step == 0:
                print(f'Iteration: {iteration+1}, cost={avg_cost:09f}')

        print('Saving the model...')
        saver = tf.train.Saver()
        saver.save(sess, f'tf_outs/{method}{learning_rate}')
        print('Finished saving!')

        print('Finished training!')
        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
        test_acc = accuracy.eval({x: test_x, y: test_y})
        results['test'] = test_acc
        print(f'Test set accuracy: {test_acc}')
        res_df = pd.DataFrame(results)
        res_df.to_csv(f'tf_outs/{method}_{learning_rate}.csv')

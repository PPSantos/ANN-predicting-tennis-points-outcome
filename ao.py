from time import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier

# Read csv file.
# Train data.
df_mens = pd.read_csv('data/mens_train_file.csv', sep=',',header=0)
df_womens = pd.read_csv('data/womens_train_file.csv', sep=',',header=0)
frames = [df_mens, df_womens]
df = pd.concat(frames)

# Submission data.
df_mens_test = pd.read_csv('data/mens_test_file.csv', sep=',',header=0)
df_womens_test = pd.read_csv('data/womens_test_file.csv', sep=',',header=0)
frames = [df_mens_test, df_womens_test]
df_test = pd.concat(frames)
df_test['submission_id'] = df_test['id'].map(str) + '_' + df_test['gender'].map(str)
df_submission = pd.read_csv('data/AUS_SubmissionFormat.csv', sep=',',header=0)
df_test = pd.merge(df_submission, df_test, how='outer', on=['submission_id', 'submission_id'])
df_test.drop(['submission_id', 'train_x', 'UE', 'FE', 'W'], axis=1, inplace=True)

# Check data.
#print(df.head())
#print(df_test.head())

X = df.iloc[:, 0:24].values
Y = df.iloc[:, 26].values
X_pred = df_test.iloc[:, 0:24].values

"""
    Pre-processing.
"""
# Encoding categorical data.
labelEncoder = LabelEncoder()
for col in [2,7,8,20,21,23]:
    X[:, col] = labelEncoder.fit_transform(X[:, col])
    X_pred[:, col] = labelEncoder.fit_transform(X_pred[:, col])

"""
from sklearn.feature_selection import SelectPercentile, f_classif
p = SelectPercentile(f_classif, percentile=80)
X = p.fit_transform(X, Y)
X_pred = p.transform(X_pred)
"""

# Categorical representation: ['FE', 'UE', 'W']
Y = keras.utils.to_categorical(labelEncoder.fit_transform(Y), num_classes=3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, shuffle=True)
#X_train, Y_train = X, Y

# Feature Scaling.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_pred = sc.transform(X_pred)

# Check shapes.
#print(X_train)
#print(Y_train)
#print(X_test)
#print(Y_test)
#print(X_train.shape)
#print(Y_train.shape)
#print(X_test.shape)
#print(Y_test.shape)

"""
    Model.
"""
def classifier():

    model = Sequential()

    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    return model

model = classifier()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(X_train, Y_train,
          epochs=120,
          batch_size=25,
          callbacks=[tensorboard])


print('Testing:')
score = model.evaluate(X_test, Y_test)
print(model.metrics_names[0], ': ', score[0], '\n', model.metrics_names[1], ': ',score[1])


"""
print('X_test')
print(X_test)
print('Correct label:')
print(Y_test)
print('predicted label:')
print(model.predict(X_test))
"""
"""

#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

clf = KerasClassifier(build_fn=classifier)

parameters = {'batch_size': [25, 32, 64],
              'epochs': [100, 200]}
grid_search = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           scoring = 'neg_log_loss',
                           cv = 4)
grid_search = grid_search.fit(X_train, Y_train)

print('Best params:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)
"""

"""
ada = AdaBoostClassifier(clf)

ada.fit(X_train, Y_train)

Y_test_pred = ada.predict(X_test)

print('Accuracy:', ada.score(X_test, Y_test))
print('loss:', log_loss(Y_test_pred, Y_test))
"""

#scores = cross_val_score(clf, X_train, Y_train, cv=4)
#print('Cross-val scores:', scores)
#print('Classifying:')
#print(model.predict(X_test))


"""
    Predicting.
"""
"""
Y_pred = model.predict(X_pred)
print(Y_pred)

for row in range(len(Y_pred)):
    df_submission.iloc[row, 3], df_submission.iloc[row, 2], df_submission.iloc[row, 4] = Y_pred[row]

print(df_submission)

df_submission.to_csv('out.csv', index=False)
"""
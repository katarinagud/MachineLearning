import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random
import pickle
import csv


file = '/Users/katarinaguderud/Desktop/Roma/ML/Homework 1/train_dataset.jsonl'

def getDataset(file):
    number = 0
    instructions_low = []  # X matrix that will be passed to count vectorizer
    labels_low = []  # Y array which contains the target values
    instructions_high = []  # X matrix that will be passed to count vectorizer
    labels_high = []
    y_all = []
    x_all = []
    data = []
    new_item = []
    with open(file) as json_file:
        for line in json_file:
            data.append(json.loads(line))
    for item in data:
        label = item['opt']
        instructions = item['instructions']
        y_all.append(item['opt']) #y_all = optimization

        if label == 'L':
            number +=  1
            labels_low.append(label)
            list_inst = []
            for inst in instructions:
                list_inst.append(inst.split(' ', 1)[0])
                to_be_added = ' '.join(list_inst)
            instructions_low.append(to_be_added)

        elif label == 'H':
            labels_high.append(label)
            list_inst = []
            for inst in instructions:
                list_inst.append(inst.split(' ', 1)[0])
                to_be_added = ' '.join(list_inst)
            instructions_high.append(to_be_added)

    needed_low = random.sample(instructions_low, k=12076)
    x_all = instructions_high + needed_low
    y_all = labels_high + labels_low[0:12076]

    return x_all, y_all

def vectorizer(x_all):
    #vectorizer.pickle.load()
    vector = CountVectorizer(ngram_range=(1,3))
    x_transformed = vector.fit_transform(x_all)
    filename = 'finalized_vectorizer.sav'
    pickle.dump(vector, open(filename, 'wb'))
    print('Vectorizer dumped')

    return x_transformed


def gaussianNB(x_train, x_test, y_train, y_test):
    #For model 1 - GaussianNB
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = model.score(x_test, y_test)
    print("Accuracy for Gaussian NB is %.3f" %acc)
    print(classification_report(y_test, y_pred, labels=None, digits=3))
    cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    print(cm)
    filename_lr = 'finalized_model.sav'
    pickle.dump(model, open(filename_lr, 'wb'))
    print('Model dumped')

def logisticRegression(x_train, x_test, y_train, y_test):
    #For model 2 - Logistic regression
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = model.score(x_test, y_test)
    print("Accuracy for Logistic Regression is %.3f" %acc)
    print(classification_report(y_test, y_pred, labels=None, digits=3))
    cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    print(cm)
    filename_lr = 'finalized_model.sav'
    pickle.dump(model, open(filename_lr, 'wb'))
    print('Model dumped')
    return y_pred


x_all, y_all = getDataset(file)
x_all = np.array(x_all)
y_all = np.array(y_all)
x_all = vectorizer(x_all)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.333, random_state=14)


#gaussianNB(x_train, x_test, y_train, y_test)
y_pred_lr = logisticRegression(x_train, x_test, y_train, y_test)


filename1 = 'finalized_vectorizer.sav'
vectorizer = pickle.load(open(filename1, 'rb'))
print('Vectorizer loaded')

filename2 = 'finalized_model.sav'
model = pickle.load(open(filename2, 'rb'))
print('Model loaded')

def predict_optimization(file, y_predicted):
    x_all = []
    data = []
    list_inst = []
    with open(file) as json_file:
        for line in json_file:
            data.append(json.loads(line))
    for item in data:
        instructions = item['instructions']
        for inst in instructions:
            list_inst.append(inst.split(' ', 1)[0])
            list_inst = ' '.join(list_inst)
            x_all.append(list_inst)
            list_inst = []
    x_all = np.array(x_all)
    y_all = np.array(y_predicted)
    x_all = vectorizer(x_all)

with open('blindtest.csv', mode='w') as f:
    f.truncate()
    w = csv.writer(f)
    for opt in y_pred_lr:
        w.writerow(opt)
    print("opt written to csv")
    f.close()



testset = '/Users/katarinaguderud/Desktop/Roma/ML/Homework 1/test_dataset_blind.jsonl'
#predict_optimization(testset, y_pred_lr)

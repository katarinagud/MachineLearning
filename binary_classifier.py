import json
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np


file = '/Users/katarinaguderud/Desktop/Roma/ML/Homework 1/train_dataset_small.jsonl'

def getDataset(file):
    y_all = []
    x_all = []
    data = []
    new_item = []
    with open(file) as json_file:
        for line in json_file:
            data.append(json.loads(line))
    for item in data:
        y_all.append(item['opt']) #y_all = optimization
        for i in item['instructions']:
            i = i.split(' ', 1)[0]
            new_item.append(i)
        new_item = ' '.join(new_item)
        x_all.append(new_item)
        new_item = []

    x_all = [[el] for el in x_all]
    print(x_all)
    return x_all, y_all

def vectorizer(x_all):
    vectorizer = CountVectorizer(ngram_range=(1,1))
    x_transformed = vectorizer.fit_transform(x_all)
    print(vectorizer.get_feature_names())
    for i in range(1):
        print(x_transformed[i].toarray())
    return x_transformed

def to_binary(y_all): #convert H to 1 and L to 0
    y_num = np.array(y_all)
    y_num[y_num == 'H'] = int(1)
    y_num[y_num == 'L'] = int(0)
    return y_num

#split training set
#x_train, x_test, y_train, y_test = train_test_split(vectorizer(x_all), y_all, test_size = 0.333, random_state=14)

#x_train = np.array(x_train).reshape(1,-1)
#y_train = np.array(y_train)

#print(x_train.shape)
#print(y_train.shape)

def gaussianNB(x_train, x_test, y_train, y_test):
    #For model 1 - GaussianNB
    model1 = GaussianNB()
    model1.fit(x_train, y_train)
    y1_pred = model1.predict(x_test)
    acc1 = model1.score(x_test, y_test)
    print("Accuracy for Gaussian NB is %.3f" %acc1)

def logisticRegression(x_train, x_test, y_train, y_test):
    #For model 2 - Logistic regression
    model2 = LogisticRegression()
    model2.fit(x_train, y_train)
    y2_pred = model2.predict(x_test)
    acc2 = model2.score(x_test, y_test)
    print("Accuracy for Logistic Regression is %.3f" %acc2)

def main():
    file = '/Users/katarinaguderud/Desktop/Roma/ML/Homework 1/train_dataset_small.jsonl'
    x_all, y_all = getDataset(file)
    x_all = vectorizer(x_all)
    x_train, x_test, y_train, y_test = train_test_split(vectorizer(x_all), y_all, test_size=0.333, random_state=14)
    print('Size of training set: %d ' % np.shape(x_train))
    print("Size of test set: %d " % np.shape(x_test))
    gaussianNB(x_train, x_test, y_train, y_test)
    logisticRegression(x_train, x_test, y_train, y_test)
    print('DONE')

main()

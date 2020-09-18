import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv


def getDataset(file):
    instructions = []
    x = []
    y = []
    with open(file, 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        instructions.append(result["instructions"])
        y.append(result["compiler"])
    newString = ""
    for i in range(len(instructions)):
        newString = " ".join(instructions[i])
        x.append(newString)

    #vectorizer = TfidfVectorizer()
    vectorizer = CountVectorizer(max_features=100)
    x = np.array(x)
    y = np.array(y)
#    x = x.reshape(x.shape[1:])
 #   x = x.transpose()

    x_train_vec = vectorizer.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_train_vec, y, test_size=0.333, random_state=14)

    return x_train, x_test, y_train, y_test

def supportVectorMachine(x_train, x_test, y_train, y_test):
    model = LinearSVC(random_state=0, multi_class='ovr').fit(x_train, y_train)
    #model = svm.SVC(random_state=0, gamma='scale', decision_function_shape='ovr', degree=3).fit(x_train, y_train)
    svm_pred = model.predict(x_test)
    accuracy = round(model.score(x_test, y_test), 4)
    cm = confusion_matrix(y_test, svm_pred)
    print('Accuracy SVM: ', accuracy)
    #print('Predicted value SVM: ', svm_pred)
    print(classification_report(y_test, svm_pred, labels=None, digits=3))
    cm = confusion_matrix(y_test, svm_pred, labels=None, sample_weight=None)
    print(cm)

def logisticRegression_multi(x_train, x_test, y_train, y_test):
    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train, y_train)
    lr_pred = model.predict(x_test)
    accuracy = round(model.score(x_test, y_test), 4)
    print('Accuracy LR: ', accuracy)
    #print('Predicted value LR: ', lr_pred)
    print(classification_report(y_test, lr_pred, labels=None, digits=3))
    cm = confusion_matrix(y_test, lr_pred, labels=None, sample_weight=None)
    print(cm)
    return lr_pred

def kNearestNeighbors(x_train, x_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    y_pred = model.predict(x_train)
    accuracy = round(model.score(x_test, y_test), 4)
    print('Accuracy kNN: ', accuracy)
    print(classification_report(y_test, y_pred, labels=None, digits=3))
    cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    print(cm)

def randomForest(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier(bootstrap=True, max_features = 'auto', n_estimators=100, max_depth=2, random_state = 0,max_leaf_nodes=None)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    accuracy = round(model.score(x_test, y_test), 4)
    print('Accuracy Random Forest: ', accuracy)
    print(classification_report(y_test, y_pred, labels=None, digits=3))
    cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    print(cm)

file = '/Users/katarinaguderud/Desktop/Roma/ML/Homework 1/train_dataset.jsonl'
x_train, x_test, y_train, y_test = getDataset(file)
#supportVectorMachine(x_train, x_test, y_train, y_test)
y_pred_lr = logisticRegression_multi(x_train, x_test, y_train, y_test)
#print(y_pred_lr)
#kNearestNeighbors(x_train, x_test, y_train, y_test)
#randomForest(x_train, x_test, y_train, y_test)

result = []
with open('blindtest.csv', mode='r') as f:
    for opt in f:
        result += opt

result = [line.rstrip('\n') for line in result]
result = [x for x in result if x]


to_be_delivered = []
index = 0
string = ''

with open('blindtest.csv', mode='a') as f:
    f.truncate()
    w = csv.writer(f)

    for item in range(len(result)):
        opt = result[item]
        comp = y_pred_lr[item]
        to_be_delivered.append(comp)
        to_be_delivered.append(opt)
        w.writerow(to_be_delivered)
        index += 1
        to_be_delivered = []

    f.close()

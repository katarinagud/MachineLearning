import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm

def getDataset():
    instructions = []
    x = []
    y = []
    with open('./train_dataset.jsonl', 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        instructions.append(result["instructions"])
        y.append(result["opt"])
    newString =""
    for i in range(len(instructions)):
        newString = " ".join(instructions[i])
        x.append(newString)

    vectorizer = CountVectorizer(max_features = 200)
    x_train_vec = vectorizer.fit_transform(x)
    print (vectorizer.get_feature_names())
    for i in range(1):
        print(x_train_vec[i].toarray())
    x_train, x_test, y_train, y_test = train_test_split(x_train_vec, y, test_size=0.333, random_state=14)
    return x_train, x_test, y_train, y_test

def logisticRegression(x_train, x_test, y_train, y_test):
    model = LogisticRegression(random_state=0,max_iter = 3000, solver='lbfgs', multi_class='ovr').fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = model.score(x_test, y_test)
    print("Accuracy logisticRegression %.3f" %acc)
    #Precision and recall
    print(classification_report(y_test, y_pred, labels=None, digits=3))
    cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    print(cm)

def supportVectorMachine(x_train, x_test, y_train, y_test):
    model = svm.SVC(gamma='scale')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = model.score(x_test, y_test)
    print("Accuracy SVM %.3f" %acc)
    #Precision and recall
    print(classification_report(y_test, y_pred, labels=None, digits=3))

def gridSearch(x_train, x_test, y_train, y_test):
    parameters = {'kernel':['linear', 'poly', 'rbf'], 'C':[0.01, 0.1, 1, 10, 100]  }
    modelclass = svm.SVC(gamma='scale')
    gridmodel = GridSearchCV(modelclass, parameters, cv=5, iid=False)
    gridmodel.fit(x_train, y_train)
    #print(gridmodel.cv_results_)
    for i in range(0,len(gridmodel.cv_results_['params'])):
        print("[%2d] params: %s  \tscore: %.3f +/- %.3f" %(i,
            gridmodel.cv_results_['params'][i],
            gridmodel.cv_results_['mean_test_score'][i],
            gridmodel.cv_results_['std_test_score'][i] ))

    a = np.argmax(gridmodel.cv_results_['mean_test_score'])
    bestparams = gridmodel.cv_results_['params'][a]
    bestscore = gridmodel.cv_results_['mean_test_score'][a]

    print("Best configuration [%d] %r  %.3f" %(a,bestparams,bestscore))
    print("Best kernel: %s" %(bestparams['kernel']))
    print("Best C: %s" %(bestparams['C']))

def main():
    x_train, x_test, y_train, y_test = getDataset()
    print("Starting learning")
    logisticRegression(x_train, x_test, y_train, y_test)
    supportVectorMachine(x_train, x_test, y_train, y_test)
    gridSearch(x_train, x_test, y_train, y_test)
main()

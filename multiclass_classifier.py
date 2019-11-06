import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import metrics


def getDataset(file):
    y_all = [] #compilers, output
    x_all = [] #instructions, input
    data = []
    new_item = []
    with open(file) as json_file:
        for line in json_file:
            data.append(json.loads(line))
    for item in data:
        y_all.append(item['compiler']) #y_all = compiler
        for i in item['instructions']:
            i = i.split(' ', 1)[0]
            new_item.append(i)
        new_item = ' '.join(new_item)
        x_all.append(new_item)
        new_item = []

    x_all = [[el] for el in x_all]
    print(x_all)
    print(y_all)
    return x_all, y_all

def vectorizer(x_all):
    vectorizer = CountVectorizer(ngram_range=(1,1))
    x_transformed = vectorizer.fit_transform(x_all) #HER ER FEILEN
    print(vectorizer.get_feature_names())
    for i in range(1):
        print(x_transformed[i].toarray())
    return x_transformed


def main():
    file = '/Users/katarinaguderud/Desktop/Roma/ML/Homework 1/train_dataset_small.jsonl'
    x_all, y_all = getDataset(file)
    x_all = vectorizer(x_all)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.333, random_state=14)
    model = linear_model.SGDClassifier() #
    model.fit(x_train, y_train)
    print();    print(model)
    expected_y = y_test
    predicted_y = model.predict(x_test)
    print();    print(metrics.confusion_matrix(expected_y, predicted_y))

main()
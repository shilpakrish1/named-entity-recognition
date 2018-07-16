import sklearn
from sklearn.feature_extraction import DictVectorizer
import numpy
import os
import math
import scipy.sparse
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#Creates each of the six classifiers
class Classifier(object):
    def __init__(self, algorithm, x_train, y_train, iterations=20, averaged=False, eta=1, alpha=1.1, limit=0):
        # Get features from examples; this line figures out what features are present in
        # the training data, such as 'w-1=dog' or 'w+1=cat'
        features = {feature for xi in x_train for feature in xi.keys()}
        v = DictVectorizer(sparse=True)
        range1 = 0
        if (limit != 0):
            range1 = limit
        else:
            range1 = len(x_train)
        if algorithm == 'Perceptron' and not averaged:
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Iterate over the training data n times
            for i in range(iterations):
                # Check each training example
                for j in range(range1):
                    xi, yi = x_train[j], y_train[j]
                    y_hat = self.predict(xi)
                    # Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] + yi * eta * value
                        self.w['bias'] = self.w['bias'] + yi * eta
        if algorithm == 'Winnow' and not averaged:
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 1.0 for feature in features}, -len(features)
            # Iterate over the training data n times
            for i in range(iterations):
                # Check each training example
                for i in range(range1):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    # Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] * math.pow(alpha, yi * value)
        if algorithm == 'Adagrad' and not averaged:
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            g = {feature: 0.0 for feature in features}
            g['bias'] = 0.0
            #Iterate over the training data n times
            for i in range(iterations):
            #Check each training example
                for i in range(range1):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    #Update the feature vector
                    if yi != y_hat:
                        #Updates the gradients for each feature
                        for feature, value in xi.items():
                            update = math.pow(value * -yi, 2)
                            g[feature] = g[feature] + update
                            g['bias'] = math.pow(-yi, 2)
                            self.w[feature] = self.w[feature] + (eta * yi * value)/(math.pow(g[feature], 1/2))
                        self.w['bias'] = self.w['bias'] + (eta * yi * 1)/(math.pow(g['bias'], 1/2))
        if algorithm == 'Perceptron' and averaged:
            # Represents the number of mistakes
            k = 0
            # Represents the counter
            counter = 0
            # Represents the current weight vector and current bias term
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Represents the cumulative bias
            self.cumBias = 0
            # Represents the active features
            self.active = {}
            # Represents the cumulative active features
            self.cumAct = {}
            # Iterates over the training data n times
            counts = 0
            for i in range(iterations):
                print(i)
                # Checks each training example
                for i in range(range1):
                    print('working')
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    if yi == y_hat:
                        # Increments the counter if the prediction is correct
                        counter = counter + 1
                    elif yi != y_hat:
                        counts = counts + 1
                        for key in self.cumAct:
                            self.cumAct[key] = self.cumAct[key] + self.active[key] * counter
                        self.cumBias = self.cumBias + self.w['bias'] * counter
                        for feature, value in xi.items():
                            # Updates the current weight vector
                            self.w[feature] = self.w[feature] + yi * eta * value
                            k = k + 1
                            counter = 1
                            if (self.active.get(feature) is None):
                                self.active[feature] = yi * eta * value
                                self.cumAct[feature] = 0
                            else:
                                self.active[feature] = self.active[feature] + yi * eta * value
                        self.w['bias'] = self.w['bias'] + yi * eta
        if algorithm == 'Winnow' and averaged:
            # Represents the number of mistakes
            k = 0
            # Represents the counter
            counter = 0
            # Represents the current weight vector and current bias term
            self.w, self.w['bias'] = {feature: 1.0 for feature in features}, -len(features)
            # Represents the cumulative bias
            self.cumBias = 0
            # Represents the active features
            self.active = {}
            # Represents the cumulative active features
            self.cumAct = {}
            # Iterates over the training data n times
            counts = 0
            for i in range(iterations):
                # Checks each training example
                for i in range(range1):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    if yi == y_hat:
                        # Increments the counter if the prediction is correct
                        counter = counter + 1
                    elif yi != y_hat:
                        counts = counts + 1
                        for key in self.cumAct:
                            self.cumAct[key] = self.cumAct[key] + self.active[key] * counter
                        self.cumBias = self.cumBias + self.w['bias'] * counter
                        for feature, value in xi.items():
                            # Updates the current weight vector
                            self.w[feature] = self.w[feature] * math.pow(alpha, yi * value)
                            k = k + 1
                            counter = 1
                            if (self.active.get(feature) is None):
                                self.active[feature] = math.pow(alpha, yi * value)
                                self.cumAct[feature] = 0.0
                            else:
                                self.active[feature] = self.active[feature] * math.pow(alpha, yi * value)
        if algorithm == 'Adagrad' and averaged:
            # Represents the number of mistakes
            k = 0
            # Represents the counter
            counter = 0
            # Represents the current weight vector and current bias term
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Represents the cumulative bias
            self.cumBias = 0
            # Represents the active features
            self.active = {}
            # Represents the cumulative active features
            self.cumAct = {}
            # Iterates over the training data n times
            counts = 0
            g = {feature: 0.0 for feature in features}
            g['bias'] = 0.0
            for i in range(iterations):
                # Checks each training example
                for i in range(range1):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    if yi == y_hat:
                        # Increments the counter if the prediction is correct
                        counter = counter + 1
                    elif yi != y_hat:
                        counts = counts + 1
                        for key in self.cumAct:
                            self.cumAct[key] = self.cumAct[key] + self.active[key] * counter
                        self.cumBias = self.cumBias + self.w['bias'] * counter
                        for feature, value in xi.items():
                            update = math.pow(value * -yi, 2)
                            g[feature] = g[feature] + update
                            g['bias'] = math.pow(-yi, 2)
                            # Updates the current weight vector
                            self.w[feature] = self.w[feature] + (eta * yi * value)/(math.pow(g[feature], 1/2))
                            k = k + 1
                            counter = 1
                            if (self.active.get(feature) is None):
                                self.active[feature] = (eta * yi * value)/(math.pow(g[feature], 1/2))
                                self.cumAct[feature] = 0
                            else:
                                self.active[feature] = self.active[feature] + (eta * yi * value)/(math.pow(g[feature], 1/2))
                        self.w['bias'] = self.w['bias'] + (eta * yi * 1)/(math.pow(g['bias'], 1/2))
    def predict(self, x, algorithm = None):
        if (algorithm is None):
            s = sum([self.w[feature] * value for feature, value in x.items()]) + self.w['bias']
            return 1 if s > 0 else -1
        elif (algorithm == 'avg' or algorithm == 'avgW'):
            dotProd = 0
            for feature, value in x.items():
                if self.cumAct.get(feature) is not None:
                    dotProd = dotProd + self.cumAct[feature] * value
                elif (algorithm == 'avgW'):
                    dotProd = dotProd + 1 * value
            return 1 if (dotProd + self.cumBias > 0) else -1

# Parse the real-world data to generate features,
# Returns a list of tuple lists
def parse_real_data(path):
    # List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        if (filename != '.DS_Store'):
            with open(path + filename, 'r') as file:
                sentence = []
                for line in file:
                    if line == '\n':
                        data.append(sentence)
                        sentence = []
                    else:
                        sentence.append(tuple(line.split()))
    return data

# Returns a list of labels
def parse_synthetic_labels(path, iterations=1 * 10^10):
    # List of tuples for each sentence
    labels = []
    counter = iterations
    with open(path + 'y.txt', 'rb') as file:
        if (counter > 0):
            for line in file:
                labels.append(int(line.strip()))
                counter = counter - 1
    return labels

# Returns a list of features
def parse_synthetic_data(path, iterations=1 * 10^10):
    # List of tuples for each sentence
    data = []
    counter = iterations
    with open(path + 'x.txt') as file:
        if (counter > 0):
            features = []
            for line in file:
                for ch in line:
                    if ch == '[' or ch.isspace():
                        continue
                    elif ch == ']':
                        data.append(features)
                        features = []
                        counter = counter - 1
                    else:
                        features.append(int(ch))
    return data

# Parse the real-world data to generate features,
# Returns a list of tuple lists
def parse_real_data(path):
    # List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        if (filename != '.DS_Store'):
            with open(path + filename, 'r') as file:
                sentence = []
                for line in file:
                    if line == '\n':
                        data.append(sentence)
                        sentence = []
                    else:
                        sentence.append(tuple(line.split()))
    return data


# Returns a list of labels
def parse_synthetic_labels(path):
    # List of tuples for each sentence
    labels = []
    with open(path + 'y.txt', 'rb') as file:
        for line in file:
            labels.append(int(line.strip()))
    return labels


# Returns a list of features
def parse_synthetic_data(path):
    # List of tuples for each sentence
    data = []
    with open(path + 'x.txt') as file:
        features = []
        for line in file:
            for ch in line:
                if ch == '[' or ch.isspace():
                    continue
                elif ch == ']':
                    data.append(features)
                    features = []
                else:
                    features.append(int(ch))
    return data

if __name__ == '__main__':
    print('Loading data...')
    # Load data from folders.
    # Real world data - lists of tuple lists
    news_train_data = parse_real_data('Data/Real-World/CoNLL/train/')
    news_dev_data = parse_real_data('Data/Real-World/CoNLL/dev/')
    news_test_data = parse_real_data('Data/Real-World/CoNLL/test/')
    email_dev_data = parse_real_data('Data/Real-World/Enron/dev/')
    email_test_data = parse_real_data('Data/Real-World/Enron/test/')

    # #Load dense synthetic data
    syn_dense_train_data = parse_synthetic_data('Data/Synthetic/Dense/train/')
    syn_dense_train_labels = parse_synthetic_labels('Data/Synthetic/Dense/train/')
    syn_dense_dev_data = parse_synthetic_data('Data/Synthetic/Dense/dev/')
    syn_dense_dev_labels = parse_synthetic_labels('Data/Synthetic/Dense/dev/')

    # Load sparse synthetic data
    syn_sparse_train_data = parse_synthetic_data('Data/Synthetic/Sparse/train/')
    syn_sparse_train_labels = parse_synthetic_labels('Data/Synthetic/Sparse/train/')
    syn_sparse_dev_data = parse_synthetic_data('Data/Synthetic/Sparse/dev/')
    syn_sparse_dev_labels = parse_synthetic_labels('Data/Synthetic/Sparse/dev/')

    #Load synthetic test data
    syn_sparse_test_data = parse_synthetic_data('Data/Synthetic/Sparse/Test/')
    syn_dense_test_data = parse_synthetic_data('Data/Synthetic/Dense/Test/')

    # Convert to sparse dictionary representations.
    # Examples are a list of tuples, where each tuple consists of a dictionary
    # and a lable. Each dictionary contains a list of features and their values,
    # i.e a feature is included in the dictionary only if it provides information.

    # You can use sklearn.feature_extraction.DictVectorizer() to convert these into
    # scipy.sparse format to train SVM, or for your Perceptron implementation.
    print('Converting Synthetic data...')
    syn_dense_train = zip(*[({'x' + str(i): syn_dense_train_data[j][i]
                              for i in range(len(syn_dense_train_data[j])) if syn_dense_train_data[j][i] == 1},
                             syn_dense_train_labels[j])
                            for j in range(len(syn_dense_train_data))])
    syn_dense_train_x, syn_dense_train_y = syn_dense_train
    syn_dense_dev = zip(*[({'x' + str(i): syn_dense_dev_data[j][i]
                            for i in range(len(syn_dense_dev_data[j])) if syn_dense_dev_data[j][i] == 1},
                           syn_dense_dev_labels[j])
                          for j in range(len(syn_dense_dev_data))])
    syn_dense_dev_x, syn_dense_dev_y = syn_dense_dev

    syn_sparse_train = zip(*[({'x' + str(i): syn_sparse_train_data[j][i]
                               for i in range(len(syn_sparse_train_data[j])) if syn_sparse_train_data[j][i] == 1},
                              syn_sparse_train_labels[j])
                             for j in range(len(syn_sparse_train_data))])
    syn_sparse_train_x, syn_sparse_train_y = syn_sparse_train
    syn_sparse_dev = zip(*[({'x' + str(i): syn_sparse_dev_data[j][i]
                             for i in range(len(syn_sparse_dev_data[j])) if syn_sparse_dev_data[j][i] == 1},
                            syn_sparse_dev_labels[j])
                           for j in range(len(syn_sparse_dev_data))])
    syn_sparse_dev_x, syn_sparse_dev_y = syn_sparse_dev

    syn_sparse_test = [({'x' + str(i): syn_sparse_test_data[j][i]
                              for i in range(len(syn_sparse_test_data[j])) if syn_sparse_test_data[j][i] == 1})
                              for j in range(len(syn_sparse_test_data))]

    syn_dense_test = [({'x' + str(i): syn_dense_test_data[j][i]
                              for i in range(len(syn_dense_test_data[j])) if syn_dense_test_data[j][i] == 1})
                              for j in range(len(syn_dense_test_data))]

    # Feature extraction
    print('Extracting features from real-world data...')
    news_train_y = []
    news_train_x = []
    train_features = set([])
    for sentence in news_train_data:
        padded = sentence[:]
        padded.insert(0, ('pad', None))
        padded.append(('pad', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2, len(padded) - 2):
            news_train_y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-1+w-2=' + str(padded[i - 1][0]) + str(padded[i - 2][0])
            feat6 = 'w+1+w+2=' + str(padded[i + 1][0]) + str(padded[i + 2][0])
            feat7 = 'w+1+w-1=' + str(padded[i + 1][0]) + str(padded[i - 1][0])
            feats = [feat1, feat2]
            train_features.update(feats)
            feats = {feature: 1 for feature in feats}
            news_train_x.append(feats)
    news_dev_y = []
    news_dev_x = []
    for sentence in news_dev_data:
        padded = sentence[:]
        padded.insert(0, ('pad', None))
        padded.append(('pad', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2, len(padded) - 2):
            news_dev_y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-1+w-2=' + str(padded[i - 1][0]) + str(padded[i - 2][0])
            feat6 = 'w+1+w+2=' + str(padded[i + 1][0]) + str(padded[i + 2][0])
            feat7 = 'w+1+w-1=' + str(padded[i + 1][0]) + str(padded[i - 1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            feats = {feature: 1 for feature in feats if feature in train_features}
            news_dev_x.append(feats)
    news_test_x = []
    for sentence in news_test_data:
        padded = sentence[:]
    #    padded.insert(0, ('pad', None))
     #   padded.append(('pad', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2, len(padded) - 2):
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-1+w-2=' + str(padded[i - 1][0]) + str(padded[i - 2][0])
            feat6 = 'w+1+w+2=' + str(padded[i + 1][0]) + str(padded[i + 2][0])
            feat7 = 'w+1+w-1=' + str(padded[i + 1][0]) + str(padded[i - 1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            feats = {feature: 1 for feature in feats if feature in train_features}
            news_test_x.append(feats)
    email_test_x = []
    for sentence in email_test_data:
        padded = sentence[:]
        padded.insert(0, ('pad', None))
        padded.append(('pad', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2, len(padded) - 2):
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feat3 = 'w-2=' + str(padded[i - 2][0])
            feat4 = 'w+2=' + str(padded[i + 2][0])
            feat5 = 'w-1+w-2=' + str(padded[i - 1][0]) + str(padded[i - 2][0])
            feat6 = 'w+1+w+2=' + str(padded[i + 1][0]) + str(padded[i + 2][0])
            feat7 = 'w+1+w-1=' + str(padded[i + 1][0]) + str(padded[i - 1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            feats = {feature: 1 for feature in feats if feature in train_features}
            email_test_x.append(feats)


def plotPoints(classifier, limit=0, averaged=False, sparse=True):
    if (not sparse):
        p = Classifier(classifier, syn_dense_train_x, syn_dense_train_y, limit=limit, averaged=averaged)
        accuracy = sum([1 for i in range(len(list(syn_dense_dev_y))) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(syn_dense_dev_y) * 100
    else:
        print('sparse')
        p = Classifier(classifier, syn_sparse_train_x, syn_sparse_train_y, limit=limit, averaged=averaged)
        accuracy = sum([1 for i in range(len(list(syn_sparse_dev_y))) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(syn_sparse_dev_y) * 100
    return accuracy


def subsetData(xtr, ytr, endIndex):
    x = xtr[:(endIndex)]
    y = ytr[:(endIndex)]
    return x, y

def runSVM(index, density):
    v = DictVectorizer(sparse=True)
    clf = LinearSVC()
    if (density == 'sparse'):
        sparse_train_x, sparse_train_y = subsetData(syn_sparse_train_x, syn_sparse_train_y, index)
        clf.fit(v.fit_transform(sparse_train_x), list(sparse_train_y))
        xtest = v.transform(list(syn_sparse_dev_x))
        ytest = list(syn_sparse_dev_y)
    else:
        dense_train_x, dense_train_y = subsetData(syn_dense_train_x, syn_dense_train_y, index)
        clf.fit(v.fit_transform(dense_train_x), list(dense_train_y))
        xtest = v.transform(list(syn_sparse_dev_x))
        ytest = list(syn_sparse_dev_y)
    labels = clf.predict(xtest)
    count = 0
    for i in range(len(ytest)):
        if (labels[i] == ytest[i]):
            count = count + 1
    return count / len(syn_sparse_dev_y)

def predictLabelsSVM():
    v = DictVectorizer(sparse=True)
    clf = LinearSVC()
    clf.fit(v.fit_transform(list(syn_sparse_train_x)), list(syn_sparse_train_y))

#Generates learning curves for the dense
def generateLearningCurvesDense():
    plt.subplot(211)
    plt.ylim([50, 120])
    accuracies = [plotPoints('Perceptron', 500, False), plotPoints('Perceptron', 1000, False),
                  plotPoints('Perceptron', 1500, False), plotPoints('Perceptron', 2000, False),
                  plotPoints('Perceptron', 2500, False), plotPoints('Perceptron', 3000, False),
                  plotPoints('Perceptron', 3500, False), plotPoints('Perceptron', 4000, False),
                  plotPoints('Perceptron', 4500, False), plotPoints('Perceptron', 5000, False),
                  plotPoints('Perceptron', 5500, False), plotPoints('Perceptron', 50000, False)]
    print('Perceptron Dense Synthetic Accuracy is' + str(plotPoints('Perceptron', 50000, False)))
    accuracies2 = [plotPoints('Winnow', 500, False), plotPoints('Winnow', 1000, False),
                  plotPoints('Winnow', 1500, False), plotPoints('Winnow', 2000, False),
                  plotPoints('Winnow', 2500, False), plotPoints('Winnow', 3000, False),
                  plotPoints('Winnow', 3500, False), plotPoints('Winnow', 4000, False),
                  plotPoints('Winnow', 4500, False), plotPoints('Winnow', 5000, False),
                  plotPoints('Winnow', 5500, False), plotPoints('Winnow', 50000, False)]
    print('Winnow Dense Synthetic Accuracy is' + str(plotPoints('Winnow', 50000, False)))
    accuracies3 = [plotPoints('Adagrad', 500, False), plotPoints('Adagrad', 1000, False),
                  plotPoints('Adagrad', 1500, False), plotPoints('Adagrad', 2000, False),
                  plotPoints('Adagrad', 2500, False), plotPoints('Adagrad', 3000, False),
                  plotPoints('Adagrad', 3500, False), plotPoints('Adagrad', 4000, False),
                  plotPoints('Adagrad', 4500, False), plotPoints('Adagrad', 5000, False),
                  plotPoints('Adagrad', 5500, False), plotPoints('Adagrad', 50000, False)]
    print('Adagrad Dense Synthetic Accuracy is' + str(plotPoints('Adagrad', 50000, False)))
    accuracies4 = [plotPoints('Adagrad', 500, True), plotPoints('Adagrad', 1000, True),
                   plotPoints('Adagrad', 1500, True), plotPoints('Adagrad', 2000, True),
                   plotPoints('Adagrad', 2500, True), plotPoints('Adagrad', 3000, True),
                   plotPoints('Adagrad', 3500, True), plotPoints('Adagrad', 4000, True),
                   plotPoints('Adagrad', 4500, True), plotPoints('Adagrad', 5000, True),
                   plotPoints('Adagrad', 5500, True), plotPoints('Adagrad', 50000, True)]
    print('Averaged Adagrad Dense Synthetic Accuracy is' + str(plotPoints('Adagrad', 50000, True)))
    accuracies5 = [plotPoints('Winnow', 500, True), plotPoints('Winnow', 1000, True),
                   plotPoints('Winnow', 1500, True), plotPoints('Winnow', 2000, True),
                   plotPoints('Winnow', 2500, True), plotPoints('Winnow', 3000, True),
                   plotPoints('Winnow', 3500, True), plotPoints('Winnow', 4000, True),
                   plotPoints('Winnow', 4500, True), plotPoints('Winnow', 5000, True),
                   plotPoints('Winnow', 5500, True), plotPoints('Winnow', 50000, True)]
    print('Averaged Winnow Dense Synthetic Accuracy is' + str(plotPoints('Winnow', 50000, True)))
    accuracies6 = [plotPoints('Perceptron', 500, True), plotPoints('Perceptron', 1000, True),
                   plotPoints('Perceptron', 1500, True), plotPoints('Perceptron', 2000, True),
                   plotPoints('Perceptron', 2500, True), plotPoints('Perceptron', 3000, True),
                   plotPoints('Perceptron', 3500, True), plotPoints('Perceptron', 4000, True),
                   plotPoints('Perceptron', 4500, True), plotPoints('Perceptron', 5000, True),
                   plotPoints('Perceptron', 5500, True), plotPoints('Perceptron', 50000, True)]
    print('Averaged Perceptron Dense Synthetic Accuracy is' + str(plotPoints('Perceptron', 50000, False)))
    accuracies7 = [runSVM(500, 'dense') * 100, runSVM(1000, 'dense')* 100, runSVM(1500, 'dense')* 100,
                   runSVM(2000, 'dense')* 100, runSVM(2500, 'dense')* 100, runSVM(3000, 'dense')* 100, runSVM(3500, 'dense')* 100,
                   runSVM(4000, 'dense')* 100, runSVM(4500, 'dense')* 100, runSVM(5000, 'dense')* 100, runSVM(5500, 'dense')* 100,
                   runSVM(50000, 'dense')* 100]
    print('SVM Dense Synthetic Accuracy is' + str(runSVM(50000, 'dense')))
    plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies, label = 'Perceptron', color='orange')
    plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies2, label = 'Winnow', color='red')
    plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies3, label='Adagrad', color='blue')
   # plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies4, label='Adagrad Average', color='pink')
  #  plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies5, label='Winnow Average', color='yellow')
  #  plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies6, label='Perceptron Average', color='green')
    plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies7, label='SVM Average', color='purple')
    plt.legend(bbox_to_anchor=(1, 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=5.)
    plt.show()
generateLearningCurvesDense()

#Generates learning curves for the dense
def generateLearningCurvesSparse():
    plt.subplot(211)
    plt.ylim([50, 120])
    accuracies = [plotPoints('Perceptron', 500, False, 'sparse'), plotPoints('Perceptron', 1000, False, 'sparse'),
                  plotPoints('Perceptron', 1500, False, 'sparse'), plotPoints('Perceptron', 2000, False, 'sparse'),
                  plotPoints('Perceptron', 2500, False, 'sparse'), plotPoints('Perceptron', 3000, False, 'sparse'),
                  plotPoints('Perceptron', 3500, False, 'sparse'), plotPoints('Perceptron', 4000, False, 'sparse'),
                  plotPoints('Perceptron', 4500, False, 'sparse'), plotPoints('Perceptron', 5000, False, 'sparse'),
                  plotPoints('Perceptron', 5500, False, 'sparse'), plotPoints('Perceptron', 50000, False, 'sparse')]
    print('Perceptron Sparse Synthetic Accuracy is' + str(plotPoints('Perceptron', 50000, False)))
    accuracies2 = [plotPoints('Winnow', 500, False, 'sparse'), plotPoints('Winnow', 1000, False, 'sparse'),
                  plotPoints('Winnow', 1500, False, 'sparse'), plotPoints('Winnow', 2000, False, 'sparse'),
                  plotPoints('Winnow', 2500, False, 'sparse'), plotPoints('Winnow', 3000, False, 'sparse'),
                  plotPoints('Winnow', 3500, False,'sparse'), plotPoints('Winnow', 4000, False, 'sparse'),
                  plotPoints('Winnow', 4500, False, 'sparse'), plotPoints('Winnow', 5000, False, 'sparse'),
                  plotPoints('Winnow', 5500, False, 'sparse'), plotPoints('Winnow', 50000, False, 'sparse')]
    print('Winnow Sparse Synthetic Accuracy is' + str(plotPoints('Winnow', 50000, False)))
    accuracies3 = [plotPoints('Adagrad', 500, False, 'sparse'), plotPoints('Adagrad', 1000, False, 'sparse'),
                  plotPoints('Adagrad', 1500, False, 'sparse'), plotPoints('Adagrad', 2000, False, 'sparse'),
                  plotPoints('Adagrad', 2500, False, 'sparse'), plotPoints('Adagrad', 3000, False, 'sparse'),
                  plotPoints('Adagrad', 3500, False, 'sparse'), plotPoints('Adagrad', 4000, False, 'sparse'),
                  plotPoints('Adagrad', 4500, False, 'sparse'), plotPoints('Adagrad', 5000, False,'sparse'),
                  plotPoints('Adagrad', 5500, False, 'sparse'), plotPoints('Adagrad', 50000, False, 'sparse')]
    print('Adagrad Sparse Synthetic Accuracy is' + str(plotPoints('Adagrad', 50000, False)))
    accuracies4 = [plotPoints('Adagrad', 500, True, 'sparse'), plotPoints('Adagrad', 1000, True, 'sparse'),
                   plotPoints('Adagrad', 1500, True, 'sparse'), plotPoints('Adagrad', 2000, True, 'sparse'),
                   plotPoints('Adagrad', 2500, True, 'sparse'), plotPoints('Adagrad', 3000, True, 'sparse'),
                   plotPoints('Adagrad', 3500, True, 'sparse'), plotPoints('Adagrad', 4000, True, 'sparse'),
                   plotPoints('Adagrad', 4500, True, 'sparse'), plotPoints('Adagrad', 5000, True, 'sparse'),
                   plotPoints('Adagrad', 5500, True, 'sparse'), plotPoints('Adagrad', 50000, True, 'sparse')]
    print('Adagrad Averaged Sparse Synthetic Accuracy is' + str(plotPoints('Adagrad', 50000, False)))
    accuracies5 = [plotPoints('Winnow', 500, True, 'sparse'), plotPoints('Winnow', 1000, True, 'sparse'),
                   plotPoints('Winnow', 1500, True, 'sparse'), plotPoints('Winnow', 2000, True, 'sparse'),
                   plotPoints('Winnow', 2500, True, 'sparse'), plotPoints('Winnow', 3000, True, 'sparse'),
                   plotPoints('Winnow', 3500, True, 'sparse'), plotPoints('Winnow', 4000, True, 'sparse'),
                   plotPoints('Winnow', 4500, True, 'sparse'), plotPoints('Winnow', 5000, True, 'sparse'),
                   plotPoints('Winnow', 5500, True, 'sparse'), plotPoints('Winnow', 50000, True, 'sparse')]
    print('Winnow Averaged Sparse Synthetic Accuracy is' + str(plotPoints('Winnow', 50000, False)))
    accuracies6 = [plotPoints('Perceptron', 500, True, 'sparse'), plotPoints('Perceptron', 1000, True, 'sparse'),
                   plotPoints('Perceptron', 1500, True, 'sparse'), plotPoints('Perceptron', 2000, True, 'sparse'),
                   plotPoints('Perceptron', 2500, True, 'sparse'), plotPoints('Perceptron', 3000, True, 'sparse'),
                   plotPoints('Perceptron', 3500, True, 'sparse'), plotPoints('Perceptron', 4000, True, 'sparse'),
                   plotPoints('Perceptron', 4500, True, 'sparse'), plotPoints('Perceptron', 5000, True, 'sparse'),
                   plotPoints('Perceptron', 5500, True, 'sparse'), plotPoints('Perceptron', 50000, True, 'sparse')]
    print('Perceptron Sparse Synthetic Accuracy is' + str(plotPoints('Perceptron', 50000, False)))
    accuracies7 = [runSVM(500, 'sparse') * 100, runSVM(1000, 'sparse')* 100, runSVM(1500, 'sparse')* 100,
                   runSVM(2000, 'sparse')* 100, runSVM(2500, 'sparse')* 100, runSVM(3000, 'sparse')* 100, runSVM(3500, 'sparse')* 100,
                   runSVM(4000, 'sparse')* 100, runSVM(4500, 'sparse')* 100, runSVM(5000, 'sparse')* 100, runSVM(5500, 'sparse')* 100,
                   runSVM(50000, 'sparse')* 100]
    print('SVM Sparse Synthetic Accuracy is' + str(plotPoints('Perceptron', 50000, False)))
    plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies,  label = 'Perceptron', color='orange')
    plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies2, label = 'Winnow', color='red')
    plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies3, label='Adagrad', color='blue')
  #  plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies4, label='Adagrad Average', color='pink')
   # plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies5, label='Winnow Average', color='yellow')
  #  plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies6, label='Perceptron Average', color='green')
    plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies7, label='SVM', color='purple')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=3)
    plt.show()
generateLearningCurvesSparse()

#Gets the predicted labels for each of the classes
def predictSyntheticDataSparse():
    #Predicts the labels using the averaged perceptron
    p = Classifier('Perceptron', syn_sparse_train_x, syn_sparse_train_y, averaged=True)
    file1 = open('p-sparse.txt', 'w')
    for i in range(len(syn_sparse_test)):
        label = p.predict(syn_sparse_test[i])
        if (label == 1):
            file1.write(str(1))
            file1.write('\n')
        else:
            file1.write(str(-1))
            file1.write('\n')
    file2 = open('svm-sparse.txt', 'w')
    v = DictVectorizer(sparse=True)
    clf = LinearSVC()
    clf.fit(v.fit_transform(list(syn_sparse_train_x)), list(syn_sparse_train_y))
    xtest = v.transform(list(syn_sparse_test))
    labels = clf.predict(xtest)
    print(labels)
    for i in range(len(labels)):
        if (labels[i] == 1):
            file2.write(str(1))
            file2.write('\n')
        else:
            file2.write(str(-1))
            file2.write('\n')
predictSyntheticDataSparse()

#Gets the predicted labels for each of the classes
def predictSyntheticDataDense():
    #Predicts the labels using the averaged perceptron
    p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y, averaged=True)
    file1 = open('p-dense.txt', 'w')
    for i in range(len(syn_dense_test)):
        label = p.predict(syn_dense_test[i])
        if (label == 1):
            file1.write(str(1))
            file1.write('\n')
        else:
            file1.write(str(-1))
            file1.write('\n')
    file2 = open('svm-dense.txt', 'w')
    v = DictVectorizer(sparse=True)
    clf = LinearSVC()
    clf.fit(v.fit_transform(list(syn_dense_train_x)), list(syn_dense_train_y))
    xtest = v.transform(list(syn_dense_test))
    labels = clf.predict(xtest)
    print(labels)
    for i in range(len(labels)):
        if (labels[i] == 1):
            file2.write(str(1))
            file2.write('\n')
        else:
            file2.write(str(-1))
            file2.write('\n')
predictSyntheticDataDense()

def generateLearningCurvesSVM():
    plt.subplot(211)
    plt.ylim([50, 120])
    accuracies7 = [runSVM(500, 'sparse') * 100, runSVM(1000, 'sparse')  * 100, runSVM(1500, 'sparse')  * 100,
                   runSVM(2000, 'sparse') * 100, runSVM(2500, 'sparse') * 100, runSVM(3000, 'sparse') * 100, runSVM(3500, 'sparse')  * 100,
                   runSVM(4000, 'sparse') * 100, runSVM(4500, 'sparse') * 100, runSVM(5000, 'sparse') * 100, runSVM(5500, 'sparse') * 100,
                   runSVM(50000, 'sparse') * 100]
    print(accuracies7)
    plt.plot([2.7, 3, 3.2, 3.3, 3.4, 3.5, 3.54, 3.60, 3.65, 3.70, 3.74, 4.7], accuracies7, label='SVM Average', color='purple')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
generateLearningCurvesSVM()


#Gets the predicted labels for each of the classes
def predictRealDataNews():
    print(email_test_x)
    #Predicts the labels using the averaged perceptron
    print(news_train_x)
    print(news_train_y)
    k = 5
    p = Classifier('Perceptron', news_train_x, news_train_y, averaged=True)
    with open('pickle', 'wb') as handle:
        pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
    file1 = open('p-conll.txt', 'w')
    for i in range(len(news_test_x)):
        label = p.predict(news_test_x[i])
        if (label == 1):
            file1.write(str('I'))
            file1.write('\n')
        else:
            file1.write(str('O'))
            file1.write('\n')
    file2 = open('p-enron.txt', 'w')
    for i in range(len(email_test_x)):
        print('here')
        label = p.predict(email_test_x[i])
        print(label)
        if (label == 1):
            print(label)
            file2.write(str('I'))
            file2.write('\n')
        else:
            file2.write(str('O'))
            file2.write('\n')

predictRealDataNews()

def runSVMReal():
    v = DictVectorizer(sparse=True)
    clf1 = LinearSVC()
    clf1.fit(v.fit_transform(list(news_train_x)), list(news_train_y))
    xtest2 = v.transform(list(news_dev_x))
    score = clf1.score(xtest2, list(news_dev_y))
    print(score)
    print(clf1.predict(xtest2))
    xtest1 = v.transform(list(news_test_x))
    print(xtest2)
    labels = clf1.predict(xtest1)
    print(labels)
    file2 = open('svm-conll.txt', 'w')
    for i in range(len(labels)):
        if (labels[i] == 1):
            file2.write(str('I'))
            file2.write('\n')
        else:
            file2.write(str('O'))
            file2.write('\n')
    file3 = open('svm-enron.txt', 'w')
    x1test = v.transform(list(email_test_x))
    labels2 = clf1.predict(x1test)
    for i in range(len(labels2)):
        if (labels2[i] == 1):
            file3.write(str('I'))
            file3.write('\n')
        else:
            file3.write(str('O'))
            file3.write('\n')
runSVMReal()




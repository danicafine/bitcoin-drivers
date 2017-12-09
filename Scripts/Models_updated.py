
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# load the data
filename = 'data/data2.dat'
allData = np.genfromtxt(filename, delimiter=',', skip_header=1)
allData = allData[~np.isnan(allData).any(axis=1)]
allData = allData[1:,:]
n, d = allData.shape
print n, d
#np.random.shuffle(allData)

# Data was not shuffled but sequentially ordered
# Training and test data 8/2 split
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = allData[np.arange(train_start, train_end), :]
data_test = allData[np.arange(test_start, test_end), :]

# Scale data
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

X_train = data_train[:, :-2] #exclude delta_5day
y_train = data_train[:, -1]

X_test = data_test[:, :-2] #exclude delta_5day
y_test = data_test[:, -1]

#models
def performClassification(X_train, y_train, X_test, y_test, method, parameters):
    """
    performs classification on returns using serveral algorithms
    """
    if method == 'RF':
        return performRFClass(X_train, y_train, X_test, y_test, parameters)

    elif method == 'KNN':
        return performKNNClass(X_train, y_train, X_test, y_test, parameters)

    elif method == 'SVM':
        return performSVMClass(X_train, y_train, X_test, y_test, parameters)

    elif method == 'ADA':
        return performAdaBoostClass(X_train, y_train, X_test, y_test, parameters)

    elif method == 'LOG':
        return performLogistics(X_train, y_train, X_test, y_test, parameters)

####### Classifier Arsenal ####################################################

def performRFClass(X_train, y_train, X_test, y_test):
    """
    Random Forest Binary Classification
    """
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy

def performKNNClass(X_train, y_train, X_test, y_test, parameters):
    """
    KNN binary Classification
    """
    clf = neighbors.KNeighborsClassifier(parameters[0])
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performSVMClass(X_train, y_train, X_test, y_test, parameters):
    """
    SVM binary Classification
    """
    c = parameters[0]
    k =  parameters[1]
    clf = SVC(C=c, kernel=k)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    return accuracy

def performAdaBoostClass(X_train, y_train, X_test, y_test, parameters):
    """
    Ada Boosting binary Classification
    """
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    return accuracy

def performLogistics(X_train, y_train, X_test, y_test, parameters):
    """
    Logistics regression Prediction
    """
    c = parameters[0]
    clf = LogisticRegression(C=c)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy


def main():
   print 'SVM linear accuracy: ' + str(performSVMClass(X_train, y_train, X_test, y_test, [1, 'linear']))
   print 'SVM rbf accuracy: ' + str(performSVMClass(X_train, y_train, X_test, y_test, [1, 'rbf']))
   print 'SVM sigmoid accuracy: ' + str(performSVMClass(X_train, y_train, X_test, y_test, [1, 'sigmoid']))
   print 'SVM poly accuracy: ' + str(performSVMClass(X_train, y_train, X_test, y_test, [1, 'poly']))
   print 'Logistics: ' + str(performLogistics(X_train, y_train, X_test, y_test, [1]))
   print 'Random Forest: ' + str(performRFClass(X_train, y_train, X_test, y_test))



if __name__ == "__main__": main()
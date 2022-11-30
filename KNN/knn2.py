import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, precision_score)
from statistics import mean
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

alternative = pd.read_csv("alternative.csv")
hiphop = pd.read_csv("hiphop.csv")
blues = pd.read_csv("blues.csv")


alternative['class'] = 0
hiphop['class'] = 1
blues['class'] = 2

listaVariada = pd.concat([alternative, hiphop, blues], axis=0, join='inner')
normalizar = preprocessing.MinMaxScaler()

caracteristica = listaVariada[['danceability', 'energy', 'loudness',
                               'tempo']]

classes = listaVariada['class']

caracteristica = normalizar.fit_transform(caracteristica)

X, y = make_classification(random_state=0)

xTrain, xTest, yTrain, yTest = train_test_split(
    caracteristica, classes)
knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")

clf = SVC(random_state=0)
clf.fit(xTrain, yTrain)

SVC(random_state=0)

predictions = clf.predict(xTest)

cf_matrix = confusion_matrix(yTest, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cf_matrix, display_labels=clf.classes_)
disp.plot()
plt.show()

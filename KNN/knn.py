import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, precision_score)
from statistics import mean

listaVariada_df = pd.read_csv("listaVariada.csv")

normalizar = preprocessing.MinMaxScaler()

caracteristica = listaVariada_df[['danceability', 'energy', 'loudness',
                                  'speechiness', 'acousticness',
                                  'instrumentalness', 'liveness', 'valence',
                                  'tempo']]
classes = listaVariada_df['Classe']

caracteristica = normalizar.fit_transform(caracteristica)

accList = []
precList = []
f1List = []
recallList = []

for i in range(20):
    xTrain, xTest, yTrain, yTest = train_test_split(
        caracteristica, classes)
    knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")

    knn.fit(xTrain, yTrain)
    yPred = knn.predict(xTest)

    accList.append(accuracy_score(yTest, yPred))
    precList.append(precision_score(yTest, yPred, average='macro'))
    f1List.append(f1_score(yTest, yPred, average='macro'))
    recallList.append(recall_score(yTest, yPred, average='macro'))

    print(mean(accList), mean(precList), mean(f1List), mean(recallList))

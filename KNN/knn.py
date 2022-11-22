import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, precision_score)
from statistics import mean


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

fig, axs = plt.subplots(2, 3, figsize=(13, 13))
axs[0, 0].scatter(listaVariada["danceability"],
                  listaVariada["energy"], c=listaVariada["class"], alpha=0.5)
axs[0, 0].set_xlabel("danceability")
axs[0, 0].set_ylabel("energy")

axs[0, 1].scatter(listaVariada["danceability"],
                  listaVariada["loudness"], c=listaVariada["class"], alpha=0.5)
axs[0, 1].set_xlabel("danceability")
axs[0, 1].set_ylabel("loudness")

axs[0, 2].scatter(listaVariada["danceability"],
                  listaVariada["tempo"], c=listaVariada["class"], alpha=0.5)
axs[0, 2].set_xlabel("danceability")
axs[0, 2].set_ylabel("tempo")

axs[1, 0].scatter(listaVariada["energy"],
                  listaVariada["loudness"], c=listaVariada["class"], alpha=0.5)
axs[1, 0].set_xlabel("energy")
axs[1, 0].set_ylabel("loudness")

axs[1, 1].scatter(listaVariada["energy"], listaVariada["tempo"],
                  c=listaVariada["class"], alpha=0.5)
axs[1, 1].set_xlabel("energy")
axs[1, 1].set_ylabel("tempo")

axs[1, 2].scatter(listaVariada["loudness"],
                  listaVariada["tempo"], c=listaVariada["class"], alpha=0.5)
axs[1, 2].set_xlabel("loudness")
axs[1, 2].set_ylabel("tempo")


plt.show()


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

# импортируем необходимые библиотеки
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from scipy import stats
import pydot 
from mpl_toolkits.mplot3d import Axes3D 
from sklearn import preprocessing
    
# Загрузим данные из xls документа
# sheet_name - название листа, с котрого надо загрузить данные
data = pd.read_excel('Dannye.xlsx', sheet_name='Лист1')
#если не учитывать столбец f1, то получается 4 кластера
data = data.iloc[:, 1:9]

#выделим данные для анализа
data_for_clust = data.values

# эта библиотека автоматически приведен данные к нормальным значениям
dataNorm = preprocessing.scale(data_for_clust)

# Вычислим расстояния между каждым набором данных,
# т.е. строками массива data_for_clust
# Вычисляется евклидово расстояние (по умолчанию)
data_dist = pdist(dataNorm, 'euclidean')
# Объедение элементов в кластера и сохранение в
# специальной переменной (используется ниже для визуализации
# и выделения количества кластеров
data_linkage = linkage(data_dist, method='average')

# Метод локтя. Позволячет оценить оптимальное количество сегментов.
# Показывает сумму внутри групповых вариаций
last = data_linkage[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
#нашли оптимальное количество кластеров
k = acceleration_rev.argmax() + 2
print("clusters:", k)

# строим кластеризаци методом KMeans
#
# n_clusters - кол-во кластеров
#

km = KMeans(n_clusters=k).fit(dataNorm)
k_means_predicted = km.predict(dataNorm)

dataK=data
#приписываем, к какому кластеру относится каждая строка
dataK['group_no']=km.labels_

#сохраняем данный файл
writer = pd.ExcelWriter('result.xlsx')
dataK.to_excel(writer,'KMeans')
writer.save()

#загружаем кластеризованные данные
dataset = pd.read_excel('result.xlsx')

X = dataset.drop('group_no', 1)
y = dataset['group_no']

#строим 3d график кластеризации
centroids = km.cluster_centers_
plt.figure('K-Means on Dataset', figsize=(10,10))
ax = plt.axes(projection = '3d')
ax.scatter(dataNorm[:,3],dataNorm[:,0],dataNorm[:,2], c=y , cmap='Set2', s=50)

ax.scatter(centroids[0,3],centroids[0,0],centroids[0,2] ,c='r', s=50, label='centroid')
ax.scatter(centroids[1,3],centroids[1,0],centroids[1,2] ,c='r', s=50)
ax.scatter(centroids[2,3],centroids[2,0],centroids[2,2] ,c='r', s=50)

#Делим наши данные на тестовые и тренировачные(это нужно, чтобы проверить как работает PCA)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#Использум RandomForestClassifier для классификации
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
estimators = classifier.estimators_

#экспортируем полученные деревья в файл
classes = [str(s) for s in classifier.classes_]

for i in range (len(classifier.estimators_)):
    export_graphviz(classifier.estimators_[i], out_file='E:/trees/' + str(i) + '.dot', 
               rounded = True, proportion = False, feature_names=data.columns,
                precision = 2, filled = True, class_names=classes)
    (graph, ) = pydot.graph_from_dot_file('E:/trees/' + str(i) + '.dot')
    graph.write_png('E:/trees/' + str(i) + '.png')

# предсказываем результаты
y_pred = classifier.predict(X_test)

#смотрим точность предсказания
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy = ' + str(accuracy_score(y_test, y_pred)))

#применяем анализ главынх компонентов(PCA), чтобы узнать, какие компоненты играют главную роль в кластеризации
pca = PCA(n_components=6)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

#также используем RandomForestClassifier, но уже применив PCA
classifier_PCA = RandomForestClassifier(max_depth=2, random_state=0)
classifier_PCA.fit(X_train, y_train)

classes = [str(s) for s in classifier.classes_]
comps = pca.components_

components = (pd.DataFrame(pca.components_,columns=data.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6']))

#выводим деревья
for i in range (len(classifier.estimators_)):
    export_graphviz(classifier_PCA.estimators_[i], out_file='E:/trees/pca/' + str(i) + '.dot', 
               rounded = True, proportion = False, feature_names=data.iloc[:, 0:6].columns,
                precision = 2, filled = True, class_names=classes)
    (graph, ) = pydot.graph_from_dot_file('E:/trees/pca/' + str(i) + '.dot')
    graph.write_png('E:/trees/pca/' + str(i) + '.png')


(graph, ) = pydot.graph_from_dot_file('E:/tree_pca.dot')

graph.write_png('E:/tree_pca.png')

y_pred = classifier_PCA.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy PCA = ' + str(accuracy_score(y_test, y_pred)))

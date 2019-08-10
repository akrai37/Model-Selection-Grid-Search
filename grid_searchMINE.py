import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dataset= pd.read_csv('Social_Network_Ads.csv')
x= dataset.iloc[:, [2, 3]].values
y= dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)

from sklearn.svm import SVC
classifier= SVC(kernel= 'rbf' , random_state= 0)
classifier.fit(x_train, y_train)

y_pred= classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

'''# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator= classifier, X= x_train , y= y_train,  cv=10)
accuracies.mean()
accuracies.std()'''

from sklearn.model_selection import GridSearchCV
parameters= [{'C':[1,10,100,1000], 'kernel':['linear']},
             {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
            ]
grid_search=GridSearchCV ( estimator= classifier,
                           param_grid= parameters,
                           scoring= 'accuracy',
                           cv=10,
                           )
                                                   
grid_search.fit(x_train, y_train)
best_accuracy= grid_search.best_score_
best_parameters= grid_search.best_params_

from matplotlib.colors import ListedColormap
x_set, y_set= x_train, y_train
x1, x2= np.meshgrid(np.arange(start= x_set[:, 0].min()- 1 , stop= x_set[:, 0].max()+ 1,step=0.01),
                    np.arange(start= x_set[:, 1].min()- 1 , stop= x_set[:, 1].max()+ 1,step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape) , 
             alpha=0.75, cmap=ListedColormap(('red','green')))
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j, 0], x_set[y_set==j, 1],
                c=ListedColormap(('red','green'))(i), label=j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('age')
plt.ylabel('salary')
plt.legend()
plt.show()    

from matplotlib.colors import ListedColormap
x_set, y_set= x_test, y_test
x1, x2= np.meshgrid(np.arange(start= x_set[:, 0].min()- 1 , stop= x_set[:, 0].max()+ 1,step=0.01),
                    np.arange(start= x_set[:, 1].min()- 1 , stop= x_set[:, 1].max()+ 1,step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape) , 
             alpha=0.75, cmap=ListedColormap(('red','green')))
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j, 0], x_set[y_set==j, 1],
                c=ListedColormap(('red','green'))(i), label=j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('age')
plt.ylabel('salary')
plt.legend()
plt.show() 








































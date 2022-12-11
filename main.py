import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

data = pd.read_csv("student-mat.csv", sep=';')+pd.read_csv("student-mat.csv", sep=';')
data['avgGrade'] = (data['G1'] + data['G2'] + data['G3'])/3
sns.set_theme()

#sns.heatmap(data.corr(), annot=True)

X = data[['absences', 'failures', 'Medu', "Fedu", 'studytime', 'age', 'traveltime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']]
y = data['avgGrade']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size = 0.01, random_state = 0)

regr = MLPRegressor(random_state=0, max_iter=100000, max_fun=100000, activation='relu', solver='lbfgs').fit(Xtrain, Ytrain)
print(regr.score(Xtrain, Ytrain))

#My data:
print(regr.predict([[0, 0, 4, 4, 4, 17, 1, 5, 5, 3, 1, 3, 5],
                    [0, 0, 4, 4, 4, 17, 1, 5, 5, 3, 1, 3, 5]])[0]/5)
#My GPA: 4.4
#Prediction: 4.50187536861295

plt.show()

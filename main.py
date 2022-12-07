import pandas as pd
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_csv("student-mat.csv", sep=';')
data['avgGrade'] = (data['G1'] + data['G2'] + data['G3'])/3
sns.set_theme()

#sns.heatmap(data.corr(), annot=True)
#make an algorithm to determine which factors have the highest correlation with grades

lmdata = []

for i in data.columns:
    if data[i].dtype == 'int64':
        lmdata.append((i, abs(data[i].corr(data['avgGrade']))))

sns.lmplot(data=data, x="Medu", y="avgGrade", ci=None)


x = data.select_dtypes(include=['int64'])
y = data['avgGrade']

regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

plt.show()

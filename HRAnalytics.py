import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('HR_comma_sep.csv', delimiter=',')
data['sales'].replace(['sales','accounting', 'hr', 'technical', 'support', 'management', 'IT', 'product_mng', 'marketing', 'RandD'],
                      [0,1,2,3,4,5,6,7,8,9], inplace=True)
data['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace = True)

df = pd.DataFrame(data)
X = df.ix[:, df.columns != 'left']
Y = df.ix[:, df.columns == 'left']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=5)

clf = RandomForestClassifier()
clf.fit(X_train,Y_train)
pred = clf.predict(X_test)
acc = accuracy_score(Y_test,pred)
print "Accuracy of my algorithm: ", acc
print "Total number of employees which are likely to leave: ", sum(pred)

important_feat = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
print important_feat

a = sns.FacetGrid(data, hue="left",aspect=4)
a.map(sns.kdeplot,'satisfaction_level',shade= True)
a.set(xlim=(0, data['satisfaction_level'].max()))
a.add_legend()
plt.show()

sns.barplot(x = 'number_project', y = 'left', data = data)
sns.plt.title('Employees that left over Number of project')
plt.show()

sns.barplot(x = 'time_spend_company', y = 'left', data = data)
sns.plt.title('Employees that left over Time spent in the company')
plt.show()

a = sns.FacetGrid(data, hue="left",aspect=4)
a.map(sns.kdeplot,'last_evaluation',shade= True)
a.set(xlim=(0, data['last_evaluation'].max()))
a.add_legend()
plt.show()

a = sns.FacetGrid(data, hue="left",aspect=4)
a.map(sns.kdeplot,'average_montly_hours',shade= True)
a.set(xlim=(0, data['average_montly_hours'].max()))
a.add_legend()
plt.show()


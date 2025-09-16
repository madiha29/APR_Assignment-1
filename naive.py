import pandas as pd

col_names = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Class"]

iris = pd.read_csv(r"C:\Users\hp\Desktop\Apr_Ass1\iris.data", header=None, names=col_names)

"""First five rows of dataframe"""
print(iris.head())


"""# **EDA**

Showing the shape of dataframe
"""
print(":djkjhgg")
print(iris.shape)

"""Info"""

print(iris.info())

"""Stat details"""

print(iris.describe())

"""Visualizing using seaborn"""
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plots to understand relationship  between variables and segregation of classes
# plt.figure(figsize=(20, 10))
# sns.pairplot(iris)
# plt.show()

"""Checking for null values in each column"""

print(iris.isnull().sum())

print("sepal length",iris['Sepal_Length'].isnull().sum())

# iris['Sepal_Width'].isnull().sum()

# iris['Petal_Length'].isnull().sum()

# iris['Petal_Width'].isnull().sum()

corr = iris.select_dtypes(include=['float64', 'int64']).corr()
plt.figure(figsize=(16,8))
sns.heatmap(corr, cmap="YlGnBu", annot=True)
plt.show()


"""Checking for duplicate rows"""

print("duplicated values are",iris.duplicated().sum())

"""Counting unique values of each column"""

print("unique values of sepal length",iris.Sepal_Length.value_counts())

print("unique values of sepal width",iris.Sepal_Width.value_counts())

"""Removal of duplicate rows"""

print("after removing duplicate rows",iris.drop_duplicates(inplace=True))

print(iris)

"""Separation of 'Class' column and storing in variable Y and other columns in variable X."""

X=iris.loc[:,'Sepal_Length':'Petal_Width']
Y=iris.loc[:,'Class']

from sklearn.model_selection import train_test_split

"""Train-test split with test size=0.2"""

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=25)

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

"""# **using library**

Performing training(fitting) the train data into the model and then giving the predictions.
"""
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_train, Y_train)
SkTrain = clf.predict(X_train) 
SkTest = clf.predict(X_test)

print(SkTest)


from sklearn.metrics import accuracy_score
print("The accuracy is",accuracy_score(SkTest,Y_test))
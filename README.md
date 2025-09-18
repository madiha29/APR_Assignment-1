#** Python used is - Python 3.13.5**


#** need to download basic libraries like pandas, matplotlib for running the code**

# **Naive bayes classifier**
import pandas as pd

#**Loading the data**
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
iris =  pd.read_csv(csv_url, names = col_names)


**First five rows of dataframe**

![First five rows of dataframe](image-1.png)


# **EDA**

(2)Dataset Info
iris.info()


![information of dataset](image-2.png)



(3)iris.describe()


![alt text](image-4.png)

**(4)Visualizing using seaborn**
plt.figure(figsize=(20, 10))
sns.pairplot(iris)
plt.show()
![alt text](image-5.png)

**(5)Showing correlation among features(non-categorical) via heatmap**
![alt text](image-7.png)

**(6)Checking for null values in each column**

**(7)Checking for duplicate rows**

**(8)Counting unique values of each column using value_counts function**

**(9)Removal of duplicate rows**
iris.drop_duplicates(inplace=True)

**(10)Final Dataset after performing EDA**
![alt text](image-6.png)

**(11)Separation of 'Class' column and storing in variable Y and other columns in variable X.**

**(12)Train-test split with test size=0.2**

**(13)Performing training(fitting) the train data into the naive_bayes model and then giving the predictions.**


**(14)Printing the predictions**

**(15)Printing the accuracy score**








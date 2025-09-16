# **naive bayes classifier**
import pandas as pd

#**Loading the data**
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
iris =  pd.read_csv(csv_url, names = col_names)

**First five rows of dataframe**
iris.head()#saved as image-1.png
![alt text](image-1.png)

# **EDA**
iris.shape






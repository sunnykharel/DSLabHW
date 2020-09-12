import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Reading data and making the dataframe
data = pd.read_csv('DF2')
testdata = pd.read_csv('DF2')
df = pd.DataFrame(data)

# From looking at the dataframe, the first column seemed to just count num. of data entries. Second and third columns are actual data.
print(list(df), '\n')

# Plotting scatter plot of the data
scatterPlot = df.plot.scatter(x='0', y='1', c = None)

# Covariance matrix of dataframe
np.cov(df['0'], df['1'])

# Transformation Matrix
q = np.asarray([[1, -1],
                [-1, 1]])

# Transforming all data by z = Qy where z is output data, y is input data (data graphed above), Q is transformation matrix, Qy is dot product
x = np.asarray(df['0'])
y = np.asarray(df['1'])
z = np.zeros(len(x))
for i in range(0, len(x)):
    z = [x[i], y[i]]
    c = np.dot(q, z)
    x[i] = c[0]
    y[i] = c[1]

# Graph new matrix and show covariance matrix.
scatterPlot = df.plot.scatter(x= '0', y= '1', c = None)
np.cov(df['0'], df['1'])

# Outlier we want to separate
a = [-1, 1]
c = np.dot(q, a)
print(c)

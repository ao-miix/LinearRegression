import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('sat.csv')
print(data.shape)
print(data.head())

X = data['SAT'].values
Y = data['GPA'].values

#mean of X and Y
x_mean = np.mean(X)
y_mean = np.mean(Y)

#total numbers of values
n = len(X)

#find slope and y-intercept
numerator = 0
denominator = 0

for i in range(n):
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    denominator += (X[i]- x_mean) ** 2
slope = numerator / denominator
intercept = y_mean - (slope * x_mean)

print(slope, intercept)

x_max = np.max(X) + 100
x_min = np.min(X) - 100

#using rsquare to find goodness of fit
correlation_matrix = np.corrcoef(X, Y)
correlation_xy = correlation_matrix[0, 1]
r_squared = correlation_xy ** 2
print(r_squared)

#line values of x and y
x = np.linspace(x_max, x_min, 1000)
y = slope * x + intercept

plt.plot(x, y, label = 'regression')
plt.scatter(X, Y, label = 'scatter')

plt.xlabel('sat')
plt.ylabel('gpa')
plt.show()


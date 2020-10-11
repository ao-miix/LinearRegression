{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(84, 2)\n",
      "    SAT   GPA\n",
      "0  1714  2.40\n",
      "1  1664  2.52\n",
      "2  1760  2.54\n",
      "3  1685  2.74\n",
      "4  1693  2.83\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "data = pd.read_csv('sat.csv')\n",
    "print(data.shape)\n",
    "print(data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = data['SAT'].values\n",
    "Y = data['GPA'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.001655688050092815 0.2750402996602781\n"
     ]
    }
   ],
   "source": [
    "#mean of X and Y\n",
    "x_mean = np.mean(X)\n",
    "y_mean = np.mean(Y)\n",
    "\n",
    "#total numbers of values\n",
    "n = len(X)\n",
    "\n",
    "#find slope and y-intercept\n",
    "numerator = 0\n",
    "denominator = 0\n",
    "\n",
    "for i in range(n):\n",
    "    numerator += (X[i] - x_mean) * (Y[i] - y_mean)\n",
    "    denominator += (X[i]- x_mean) ** 2\n",
    "slope = numerator / denominator\n",
    "intercept = y_mean - (slope * x_mean)\n",
    "\n",
    "print(slope, intercept)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "x_max = np.max(X) + 100\n",
    "x_min = np.min(X) - 100\n",
    "\n",
    "#line values of x and y\n",
    "x = np.linspace(x_max, x_min, 1000)\n",
    "y = slope * x + intercept\n",
    "\n",
    "plt.plot(x, y, label = 'regression')\n",
    "plt.scatter(X, Y, label = 'scatter')\n",
    "\n",
    "plt.xlabel('sat')\n",
    "plt.ylabel('gpa')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.40600391479679826\n"
     ]
    }
   ],
   "source": [
    "#using rsquare to find goodness of fit\n",
    "correlation_matrix = np.corrcoef(X, Y)\n",
    "correlation_xy = correlation_matrix[0, 1]\n",
    "r_squared = correlation_xy ** 2\n",
    "print(r_squared)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "X = x.reshape((m, 1))\n",
    "Y = "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

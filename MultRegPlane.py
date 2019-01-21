import csv
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('example.csv', 'rb') as csvfile:
    readerOb = csv.reader(csvfile, delimiter = ',')

    #X_1 and X_2 are only for meshgrid (parameters require 1d array)
    #X1_X2 will include both predictor variables (X_1 and X_2)
    rowSpec = ""
    X_1 = []
    X_2 = []
    Y = []
    X1_X2 = [ [] for i in range(len(next(readerOb)) - 1)]
    csvfile.seek(0)

    for row in readerOb:
        x = 0
        for el in row:
            if (el == "X_1" or el == "X_2" or el == "Y"):
                rowSpec = el
            else:
                if (rowSpec == "X_1"):
                    X1_X2[x].append(float(el))
                    X_1.append(float(el))
                    x += 1
                if (rowSpec == "X_2"):
                    X1_X2[x].append(float(el))
                    X_2.append(float(el))
                    x += 1
                if (rowSpec == "Y"):
                    Y.append(float(el))

X = np.array(X1_X2)
Y_1 = np.array(Y)
X_1_wConst = sm.add_constant(X) # add constant term for intercept
results = sm.OLS(Y_1, X_1_wConst).fit() # constant is the first term in array
X_grid, Y_grid = np.meshgrid(np.array(X_1), np.array(X_2))
Z_grid = np.zeros((len(X_2), len(X_1)))

# returns predicted value via multiple regression equation
def f(x, y):
    return (results.params[0] + (results.params[1] * x) + results.params[2] * y);

for x in range(len(X)):
    for y in range(len(Y_1)):
        Z_grid[x][y] = f(X_grid[x][y], Y_grid[x][y])

fig = plt.figure()

# Shows scatter and plane on same graph
ax = Axes3D(fig)
ax.plot_surface(X_grid, Y_grid, Z_grid)
ax.scatter(X_1, X_2, Y)

# Shows scatter and plane on separate graphs
#ax = fig.add_subplot(1, 2, 1, projection='3d')
#ax.plot_surface(X_grid, Y_grid, Z_grid)
#ax = fig.add_subplot(1, 2, 2, projection='3d')
#ax.scatter(X_1, X_2, Y)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("Multiple Regression Plane\ny = %f + %fx1 + %fx2" % (results.params[0], results.params[1], results.params[2]))
plt.show()

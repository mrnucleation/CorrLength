from sklearn.linear_model import LinearRegression
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

filelist = sorted(glob("*radialplot*"))
filetemp = [350.0, 400.0, 450.0, 500.0, 350.0, 400.0, 450.0, 500.0]
for t, infile in zip(filetemp, filelist):
    print(infile)
    data = np.loadtxt(infile)
    Q = data[1:7,0]
    Q = np.square(Q).reshape(-1,1)

    Y = data[1:7,1]
#    Y /= Y.max()
    Y = 1.0/Y
    Y.reshape(-1,1)

    model = LinearRegression().fit(Q,Y)
#    print(model.intercept_)
#    print(model.coef_)
#    if model.intercept_ < 0.0:
#        continue

    I0 = 1.0/np.abs(model.intercept_)
    e0 = I0 * model.coef_
    e0 = np.sqrt(e0)
    # Plot the data points and the regression line
    plt.scatter(Q, Y)
    plt.plot(Q, model.predict(Q.reshape(-1, 1)), color='red')
    plt.title('Linear Regression Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    print(model.score(Q,Y))
    print(e0)



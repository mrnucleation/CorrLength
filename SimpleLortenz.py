import numpy as np
#import jax.numpy as np
#import jax
import optax
import warnings
import matplotlib.pyplot as plt

def rmse(Y_model, Y_target):
    err = Y_model - Y_target
    err = np.square(err)
    err = np.mean(err)
    return err
#A function which takes a set of X and Y data and fits a linear regression model to it.
class SimpleCorrLengthModel:
    def __init__(self, Q, Y):
        self.I0 = 1.0
        self.e0 = 1.0
        self.vm = 0.0
        self.parameters = [self.I0, self.e0, self.vm]
        self.parameters = np.array(self.parameters)
        self.Q = Q
        self.Y = Y

    def __call__(self, parameters):
        with warnings.catch_warnings():    
            warnings.filterwarnings('ignore', r'overflow encountered in')
            warnings.filterwarnings('ignore', r'overflow encountered in reduce')
            warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            warnings.filterwarnings('ignore', r'Input contains NaN')
            warnings.filterwarnings('ignore', r'invalid value encountered in power')
            self.set_weights(parameters)
            score = self.computeerror(self.Q, self.Y)
            with open("dumpfile.dat", "a") as outfile:
                outstr = ' '.join([str(x) for x in parameters])
                outfile.write('%s | %s\n'%(outstr, score))

        return score


    def computeerror(self, Q, Y):
        Y_model = self.predict(Q)
        err = rmse(Y, Y_model)
        if np.isnan(err).any() or np.isinf(err).any():
            err = 1e300
        print(err)
        return err

    def predict(self, Q):
        X = Q * self.parameters[1]
#        Y_model = self.parameters[0]/(1 + np.square(X))
        Y_model = self.parameters[0]/(1 + np.power(X, (2-self.parameters[2])))
        return Y_model


    def get_weights(self):
        return self.parameters

    def set_weights(self, weights):
        self.parameters = weights

    def plot(self):
        plt.scatter(self.Q, self.Y)
        plt.plot(self.Q, self.predict(self.Q), color='red')
        plt.title('Linear Regression Plot')
        plt.xlabel('Q')
        plt.ylabel('I')
        plt.show()


import numpy as np

#A function which takes a set of X and Y data and fits a linear regression model to it.
class CorrLengthModel:
    def __init__(self, temperature=350.0):
        self.w_n = 0.0
        self.I0 = 1.0
        self.e0 = 1.0
        self.Tsp = 1.0
        self.temp = temperature
        self.gamma = 1.0
        self.nu = 1.0

    def predict(self, Q):
        #Fit the model to the data
        X = Q * self.e0_func
        a = self.Iq0_func()
        Y_model = self.lorentzian(X, a)
        return Y_model

    def lorentzian(self, X, a):
        return a / ( 1 + np.power(X,2-self.w_n) )

    def Iq0_func(self):
        Iq0 = self.I0 * np.power( (self.temp-self.Tsp)/(self.Tsp), -self.gamma) )
        return Iq0

    def e0_func(self):
        eq0 = self.e0 * np.power( (self.temp-self.Tsp)/(self.Tsp), -self.nu) )
        return eq0

    def get_weights(self):
        return [self.w_n, self.Tsp, self.gamma, self.nu, self.I0, self.e0]

    def set_weights(self, weights):
        self.w_n, self.Tsp, self.gamma, self.nu, self.I0, self.e0 = tuple(weights)

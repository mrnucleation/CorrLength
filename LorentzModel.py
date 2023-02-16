import numpy as np
#import jax.numpy as np
#import jax
import optax
import warnings

def rmse(Y_model, Y_target):
    err = Y_model - Y_target
    err = np.square(err)
    err = np.mean(err)
    return err
#A function which takes a set of X and Y data and fits a linear regression model to it.
class CorrLengthModel:
    def __init__(self, Q, Y, T):
        self.w_n = 0.0
        self.I0 = 1.0
        self.e0 = 1.0
        self.Tsp = 500.0
        self.temp = T
        self.gamma = 1.0
        self.nu = 1.0
        self.parameters = [self.w_n, self.Tsp, self.gamma, self.nu, self.I0, self.e0]
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
                e0 = self.e0_func(500.0)
                outfile.write('%s | %s |%s\n'%(outstr, e0, score))

        return score

    def fit(self, Q, Y, lrate = 1e-6):
        optimizer = optax.adam(lrate)
        opt_state = optimizer.init(self.parameters)
        for iepoch in range(1000):
            err, gradient = jax.value_and_grad(self.computeerror)(Q, Y)
            updates, opt_state = optimizer.update(gradient, opt_state)
            curpos = optax.apply_updates(curpos, updates)
            if iepoch%10 == 0:
                print("Epoch: %s, Training Error: %s"%(iepoch, err))

    def computeerror(self, Q, Y):
        Y_model = self.predict(Q)
        err = rmse(Y, Y_model)
        if np.isnan(err).any() or np.isinf(err).any():
            err = 1e300
        print(err)
        return err

    def predict(self, Q):
        X = Q * self.e0_func(self.temp)
        a = self.Iq0_func()
        Y_model = self.lorentzian(X, a)
        return Y_model

    def lorentzian(self, X, a):
#        return a / ( 1 + np.power(X,2-self.w_n) )
        return a / ( 1 + np.power(X,2-self.parameters[0]) )

    def Iq0_func(self):
#        Iq0 = self.I0 * np.power( (self.temp-self.Tsp)/(self.Tsp), -self.gamma) )
        Iq0 = self.parameters[4] * np.power( (self.temp-self.parameters[1])/(self.parameters[1]), -self.parameters[2]) 
        return Iq0

    def e0_func(self, T):
        eq0 = self.parameters[5] * np.power( (T-self.parameters[1])/(self.parameters[1]), -self.parameters[3]) 
        return eq0

    def get_weights(self):
#        return [self.w_n, self.Tsp, self.gamma, self.nu, self.I0, self.e0]
        return self.parameters

    def set_weights(self, weights):
        self.parameters = weights
#        self.w_n, self.Tsp, self.gamma, self.nu, self.I0, self.e0 = tuple(weights)

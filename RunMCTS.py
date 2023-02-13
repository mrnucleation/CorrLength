from MCTSOpt import Tree
#from MCTSOpt.ParameterOpt.LogisticSearch import LogisticSearch
#from BayesianObject import BayesianData as LogisticSearch
#from ParameterObject import ParameterData as LogisticSearch
from LogisticSearch import LogisticSearch
from SelectionRule_UBUnique import UBUnique as UBEnergy 
from LorentzModel import CorrLengthModel

import numpy as np
from datetime import datetime
from random import random, seed
from copy import deepcopy
seed(datetime.now())

from init_latin_hypercube_sampling import init_latin_hypercube_sampling
def expandhead_latin(npoints, tree, lb, ub):
    seedpoints = init_latin_hypercube_sampling(np.array(lb), np.array(ub), npoints)
    for point in seedpoints:
#        print("Point %s"%(point))
        nodedata = indata.newdataobject()
        nodedata.setstructure(list(point))
        tree.expandfromdata(newdata=nodedata)


def expandhead_radiallatin(npoints, ndim, tree, rmax):
    seedpoints = init_latin_hypercube_sampling(np.array([0.0]), np.array([rmax]), npoints)
    for rcur in seedpoints:
        u = np.random.normal(0.0, 1.0, ndim)  # an array of d normally distributed random variables
        norm = np.sum(u**2) **(0.5)
        x = rcur*u/norm
        nodedata = indata.newdataobject()
        nodedata.setstructure(x)
        tree.expandfromdata(newdata=nodedata)


data = np.loadtxt("radialplot_Avg_xy.dump")
print(data)
print(data.shape)
Q, Y = data[:,0], data[:,1]

Y /= Y.max()
model = CorrLengthModel(Q,Y,temperature=450.0)

depthlimit = 900
startpar = model.get_weights()
nParameters = len(startpar)
ubounds = [10.0, 600.0, 10.0, 10.0, 10.0, 100.0 ]
lbounds = [0.0 , 300.0,  0.0,  0.0,  0.0,   0.0 ]
startset = startpar

depthscale = [10.0, 5.0, 0.8, 0.4, 0.2]
#depthscale = [nParameters*x for x in depthscale]

ubounds = np.array(ubounds)
lbounds = np.array(lbounds)

options ={'verbose':2}

indata = LogisticSearch(parameters=startset, ubounds=ubounds, lbounds=lbounds, lossfunction=model, depthscale=depthscale, options=options)

#---Tree Main Run loop---
#Critical Parameters to Set
tree = Tree(seeddata=indata, 
        playouts=10, 
        selectfunction=UBEnergy, 
        headexpansion=5,
        verbose=True)
tree.setconstant(0e0)
expandhead_radiallatin(15, len(ubounds), tree, 0.5)
tree.expandfromdata(newdata=indata)
lastmin = 1e300
lastloop = 0
tree.setplayouts(10)
tree.setconstant(1.0)

for iLoop in range(1,500):
    for i in range(5):
        tree.playexpand(nExpansions=1, depthlimit=depthlimit)
        tree.simulate(nSimulations=1)
        tree.autoscaleconstant(scaleboost=0.5)
    minval = tree.getbestscore()
    if minval < lastmin:
        minval = min(lastmin, minval)
        lastloop = iLoop


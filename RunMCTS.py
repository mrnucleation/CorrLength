from MCTSOpt import Tree
#from MCTSOpt.ParameterOpt.LogisticSearch import LogisticSearch
#from BayesianObject import BayesianData as LogisticSearch
#from ParameterObject import ParameterData as LogisticSearch
from LogisticSearch import LogisticSearch
from SelectionRule_UBUnique import UBUnique as UBEnergy 

from GetModel import getmodel

import numpy as np
from datetime import datetime
from random import random, seed
from copy import deepcopy
from glob import glob
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
'''
filelist = sorted(glob("*radialplot_L*"))
print(filelist)
Q = []
Y = []
T = []
filetemp = [350.0, 400.0, 450.0, 500.0]
for t, infile in zip(filetemp, filelist):
    print(infile)
    data = np.loadtxt(infile)
    Q.append(data[1:7,0])
    y = data[1:7,1]
    y /= y.max()
    Y.append(y)
    T.append([t for x in range(y.shape[0])])

Q = np.concatenate(Q)
Y = np.concatenate(Y)
T = np.concatenate(T)
print(Q.shape)
print(Y.shape)
print(T.shape)


Y /= Y.max()
model = CorrLengthModel(Q,Y,T)

'''

model = getmodel()

depthlimit = 900
startpar = model.get_weights()
nParameters = len(startpar)
ubounds = [0.05,  750.0,   2.0,   2.0,  1000000.0, 1000000.0 ]
lbounds = [-0.05,  50.1,  0.0,  0.0,  1e-5,   1e-5 ]
startset = startpar

depthscale = [10.0, 5.0, 0.8, 0.4, 0.2, 0.05, 0.000001]
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
expandhead_latin(35, tree, lbounds, ubounds)
#expandhead_radiallatin(15, len(ubounds), tree, 1e-5)
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


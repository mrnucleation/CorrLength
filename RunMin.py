import sys
from GetModel import getsimplemodel
from scipy.optimize import minimize
from glob import glob

#filename = sys.argv[1]
#par = []
#with open(filename, 'r') as parfile:
#    for line in parfile:
#        newpar = float(line.split()[0])
#        par.append(newpar)

outpar = []

filelist = sorted(glob("*radialplot_V*")) + sorted(glob("*radialplot_L*"))
for infile in filelist:
    model = getsimplemodel(infile)
    par = [100.0, 100.0, 0.0]
    res = minimize(model, par, method='BFGS', tol=1e-12,
                 options={'maxiter':1000000} )
    print(res)
    model.plot()
    outpar.append(res.x)


for par in outpar:
    print(par)

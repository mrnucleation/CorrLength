import sys
from StephPlex import StephSimplex
from GetModel import getmodel
filename = sys.argv[1]
par = []
with open(filename, 'r') as parfile:
    for line in parfile:
        newpar = float(line.split()[0])
        par.append(newpar)

ubounds = [2.0,  1250.0,   10.0,   10.0,  100000.0, 10000.0 ]
lbounds = [0.0,  150.1,  1e-7,  1e-7,  1e-5,   1e-5 ]
startsize = [(ub-lb)*0.001 for lb, ub in zip(lbounds, ubounds)]

model = getmodel()

simplex = StephSimplex(model)
results, score = simplex.runopt(
                   lbounds, 
                   ubounds, 
                   initialguess=par, 
                   maxeval=100000, 
                   delmin=[1e-13 for x in lbounds],
                   startstepsize=startsize
                   )
print("Post Minimization Score: %s"%(score))
print("Post Minimization Parameters: %s"%(results))

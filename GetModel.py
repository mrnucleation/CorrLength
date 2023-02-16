from glob import glob
from LorentzModel import CorrLengthModel
from SimpleLortenz import SimpleCorrLengthModel
import numpy as np

def getmodel():
    filelist = sorted(glob("*radialplot_L*"))
    print(filelist)
    Q = []
    Y = []
    T = []
    filetemp = [350.0, 400.0, 450.0, 500.0]
    for t, infile in zip(filetemp, filelist):
        print(infile)
        data = np.loadtxt(infile)
        Q.append(data[0:5,0])
        y = data[0:5,1]
        y /= y.max()
        Y.append(y)
        T.append([t for x in range(y.shape[0])])

    Q = np.concatenate(Q)
    Y = np.concatenate(Y)
    T = np.concatenate(T)


    Y /= Y.max()
    model = CorrLengthModel(Q,Y,T)
    return model

def getsimplemodel(infile):
    data = np.loadtxt(infile)
    Q = data[2:6,0]
    Y = data[2:6,1]
    Y /= Y.max()
    model = SimpleCorrLengthModel(Q,Y)
    return model

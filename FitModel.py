import numpy as np
from LorentzModel import CorrLengthModel
import jax
import jax.numpy as jnp
from jax import grad, jit


model = CorrLengthModel(temperature=350.0)
data = np.loadtxt("radialplot_Avg_xy.dump")
print(data)
print(data.shape)
Q, Y = data[:,0], data[:,1]


model.fit(Q,Y, lrate=1e-7)

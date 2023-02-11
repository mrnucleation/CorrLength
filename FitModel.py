import numpy as np
from LorentzModel import CorrLengthModel

model = CorrLengthModel(temperature=350.0)
data = np.load_data("data.txt")

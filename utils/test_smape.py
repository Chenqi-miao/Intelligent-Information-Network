import numpy as np
from src.evaluate import compute_smape

print('all zeros:', compute_smape(np.array([0,0,0]), np.array([0,0,0])))
print('mixed:', compute_smape(np.array([0,100,0,200]), np.array([0,110,0,190])))
print('normal:', compute_smape(np.array([100,200,300]), np.array([110,190,310])))

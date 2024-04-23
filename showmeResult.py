import numpy as np
align = [ 0, 1, 2]
activate = [ 0, 1, 2, 3]
from scipy.stats import spearmanr
name = [ "TES", "Ribbon"]
for task in name:
    for ali in align:
        for act in activate:
            npz_name = f"./temp_log/GDLTETH1/{task}_{ali}_{act}_{task}.npy"
            record = np.load(npz_name, allow_pickle=True)
            pred,gt,feature = record
            coef,_ = spearmanr(pred, gt)
            print(npz_name, coef)
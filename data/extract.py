import pandas as pd
import os
import numpy as np
import glob
inf = glob.glob('/home/wequ0318/sleep/data/eeg_fpz_cz/*.npz')
for _f in inf:
    with np.load(_f) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
        df_data = pd.DataFrame(np.squeeze(data))
        df_label = pd.DataFrame(labels)
        fname = os.path.basename(_f)
        num = df_data.shape[0]
        df_epoch = pd.DataFrame(np.arange(num))
        df_name = pd.DataFrame(np.repeat(fname[:-4], num))
        df = pd.concat([df_data, df_label, df_epoch, df_name], axis=1)
        df.to_csv(fname.replace('npz', 'csv'), sep=',', index=False)

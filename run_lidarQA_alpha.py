import os
import lidarQA
import numpy as np

date_ini = '20200114'
result_dir = '/mnt/NASGFAT/datos/MULHACEN/QA/depolarization_calibration/test_alpha_epsilon_jaba'
#alpha_arr = np.arange(-10, 10, 5)
#epsilon_arr = np.arange(-10, 10, 5)
alpha_arr = np.asarray([8.8])
epsilon_arr = np.asarray([-5.44])
for alpha in alpha_arr:
    for epsilon in epsilon_arr:
        loop = "a_%.1f_e%.1f" % (alpha, epsilon)
        print(loop)
        lidarQA.depolarization(date_ini, output_directory=os.path.join(result_dir, date_ini, loop), alpha=alpha, epsilon=epsilon)
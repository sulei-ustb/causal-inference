
# 人员：maxiaofei
# 开发时间：18/11/2020下午8:09
# 开发工具：PyCharm

from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import time

from mydocument.hw3_helper import *
from mydocument.VAE_funtions import *
from mydocument.mat2txt import *

'''Part (a) Data from a Full Covariance Gaussian '''


# 功能1：可视化产生的数据
# module 1: Data visualization

visualize_q1_data('a', 1)
# visualize_q1_data('a', 2)






#
# module 2：Train a VAE model
#
t_start = time.time()
#
# print(t_start)
q1_save_results('a', 1, q1)
#
# # q1_save_results('a', 2, q1)
#
t_end = time.time()
# print(t_end)
#
print("Total elapsed time is:"+str(t_end - t_start))








# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from numpy import mat 
import matplotlib.pyplot as plt
import seaborn as sns
import time


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


dataLen = 20000
# generate I;C;A
np.random.seed(202201)
I = np.random.normal(0, 5, (dataLen,1))
C = np.random.normal(0, 5, (dataLen,1))
A = np.random.normal(0, 5, (dataLen,1))

# generate T
T_I_coff = 4
T_C_coff = 5
np.random.seed(202202)
T_random = np.random.normal(0, 1, (dataLen,1))
prob_T = 0.4 * sigmoid(I * T_I_coff) + 0.5 * sigmoid(C * T_C_coff)  + 0.1 * T_random
prob_T[prob_T>1] = 1
prob_T[prob_T<0] = 0
T = np.random.binomial(1,prob_T,[dataLen,1])

# generate M
M_T_coff = 5
np.random.seed(202203)
M_random = np.random.normal(0, 1, (dataLen,1))
M = sigmoid(T * M_T_coff) + 0.1 * M_random

# generate Y;Y_cf;Y_0;Y_1
Y_A_coff = 3
Y_C_coff = 4
Y_M_coff = 5
np.random.seed(202204)
Y_random  = np.random.normal(0, 1, (dataLen,1))
T_0 = np.zeros(T.shape)
T_1 = np.zeros(T.shape) + 1
M_0 = sigmoid(T_0 * M_T_coff) + 0.1 * M_random
M_1 = sigmoid(T_1 * M_T_coff) + 0.1 * M_random

Y_0 = sigmoid(M_0 * Y_M_coff) + sigmoid(C * Y_C_coff) + sigmoid(A * Y_A_coff) + 0.1 * Y_random
Y_1 = sigmoid(M_1 * Y_M_coff) + sigmoid(C * Y_C_coff) + sigmoid(A * Y_A_coff) + 0.1 * Y_random

Y = []
Y_cf = []
for i in range(dataLen):
    if T[i] == 1:
        Y_ = np.random.normal(Y_1[i,0], 1, 1)
        Y_cf_ = np.random.normal(Y_0[i,0], 1, 1)
    else:
        Y_ = np.random.normal(Y_0[i,0], 1, 1)
        Y_cf_ = np.random.normal(Y_1[i,0], 1, 1)
    Y.append(Y_)
    Y_cf.append(Y_cf_)

Y = np.array(Y)
Y_cf = np.array(Y_cf)

# generate Z
Z_T_coff = 5
Z_Y_coff = 4
np.random.seed(202205)
Z_random  = np.random.normal(0, 1, (dataLen,1))
Z = sigmoid(T * Z_T_coff) + sigmoid(Y * Z_Y_coff) + 0.1 * Z_random

TO_T_coff = 5
np.random.seed(202206)
TO_random  = np.random.normal(0, 1, (dataLen,1))
TO = sigmoid(T * TO_T_coff) + 0.1 * TO_random

YO_Y_coff = 4
np.random.seed(202207)
YO_random  = np.random.normal(0, 1, (dataLen,1))
YO = sigmoid(Y * YO_random) + 0.1 * YO_random

data_A = pd.DataFrame(A)
data_A.columns = ['A']

data_c = pd.DataFrame(C)
data_c.columns = ['C']

data_i = pd.DataFrame(I)
data_i.columns = ['I']

data_t = pd.DataFrame(T)
data_t.columns = ['T']

data_m = pd.DataFrame(M)
data_m.columns = ['M']
      
data_y = pd.DataFrame(Y)
data_y.columns = ['Y']

data_y_cf = pd.DataFrame(Y_cf)
data_y_cf.columns = ['Y_cf']

data_mu0 = pd.DataFrame(Y_0)
data_mu0.columns = ['Y_0']
data_mu1 = pd.DataFrame(Y_1)
data_mu1.columns = ['Y_1']

data_z = pd.DataFrame(Z)
data_z.columns = ['Z']

data_TO = pd.DataFrame(TO)
data_TO.columns = ['TO']
data_YO = pd.DataFrame(YO)
data_YO.columns = ['YO']

data_ = pd.concat([data_t,data_A,data_i,data_c,data_m,data_y,data_y_cf,data_mu0,data_mu1,data_z,data_TO,data_YO],axis=1)
data_.to_csv('../data/data-discrete.csv',index = False)

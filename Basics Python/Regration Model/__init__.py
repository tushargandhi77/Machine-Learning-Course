import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

x_train = np.array([3,4,2,5])
y_train = np.array([2,1,6,3])


def compute_cost(x,y,w,b):
    cost = 0;
    m = x.shape[0]
    for i in range(m):
        f_wb_i = w*x[i] + b
        cost += (f_wb_i - y[i])**2
    cost = cost/2*m
    return cost

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_w = 0
    dj_b = 0
    for i in range(m):
        f_wb = x[i]*w + b
        dj_w_i = (f_wb - y[i])*x[i]
        dj_b_i = (f_wb - y[i])
        dj_w += dj_w_i
        dj_b += dj_b_i
    dj_w = dj_w/m
    dj_b = dj_b/m
    return dj_w,dj_b


plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()

import numpy as np

def _weighted_mean_absolute_percentage_error(preds, target, epsilon=1.17e-06):
    sum_abs_error = np.abs(preds - target).sum()
    sum_scale = np.abs(target).sum()
    return sum_abs_error / np.clip(sum_scale, epsilon, None)

def cumavg(m):
    cumsum= np.cumsum(m)
    return cumsum / np.arange(1, cumsum.size + 1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))
    #return 0

def metric(pred, true):

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    wmape = _weighted_mean_absolute_percentage_error(pred, true)

    return mae,mse,rmse,mape,wmape
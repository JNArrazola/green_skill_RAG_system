import numpy as np

def mae(pred, gold, index_list=None):
    if index_list is not None:
        pred = pred[:,:,index_list]
        gold = gold[:,:,index_list]
    return np.abs(pred-gold).mean()

def rmse(pred, gold, index_list=None):
    if index_list is not None:
        pred = pred[:,:,index_list]
        gold = gold[:,:,index_list]
    return np.sqrt(((pred-gold)*(pred-gold)).mean())

def smape(pred, gold, index_list=None):
    if index_list is not None:
        pred = pred[:,:,index_list]
        gold = gold[:,:,index_list]
    return 100 * np.mean(np.abs(pred - gold) / (np.abs(pred) + np.abs(gold)))

def rrmse(pred, gold, index_list=None):
    if index_list is not None:
        pred = pred[:,:,index_list]
        gold = gold[:,:,index_list]
    rrmse_loss = np.sqrt(((pred-gold)*(pred-gold)).mean() / np.square(pred).mean())
    return rrmse_loss * 100

def compute_metrics(pred, gold, index_list=None, tag=''):
    metrics = {}
    metrics[f'{tag}mae'] = mae(pred, gold, index_list)
    metrics[f'{tag}rmse'] = rmse(pred, gold, index_list)
    metrics[f'{tag}smape'] = smape(pred, gold, index_list)
    metrics[f'{tag}rrmse'] = rrmse(pred, gold, index_list)
    
    return metrics
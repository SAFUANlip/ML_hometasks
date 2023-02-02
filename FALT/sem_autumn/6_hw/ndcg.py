import math
from math import log2
import numpy as np

import torch
from torch import Tensor, sort


def compute_gain(y_value: float, gain_scheme: str) -> float:
    # допишите ваш код здесь
    if gain_scheme == "exp2":
        return pow(2.0, y_value) - 1
    if gain_scheme == "const":
        return y_value
    pass

def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str, top_k=None) -> float:
    ys_true = ys_true.reshape(1,-1)[0]
    ys_pred = ys_pred.reshape(1,-1)[0]
    
    mask = torch.argsort(ys_pred, descending = True)
    ys_true_right = ys_true[mask]
    dcg = []
    for idx, val in enumerate(ys_true_right): 
        numerator = compute_gain(val, gain_scheme=gain_scheme)
        # add 2 because python 0-index
        denominator =  np.log2(idx + 2) 
        score = numerator/denominator
        dcg.append(score)
    return sum(dcg[:top_k]).item()


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const', top_k=None) -> float:
    ys_true = ys_true.reshape(1,-1)[0]
    ys_pred = ys_pred.reshape(1,-1)[0]
    
    numerator = dcg(ys_true, ys_pred, gain_scheme, top_k=top_k)
    mask = torch.argsort(ys_pred, descending = True)
    
    ys_true_right = ys_true[mask]
    ndcg = []
    for idx, val in enumerate(torch.sort(ys_true, descending = True)[0]): 
        num = compute_gain(val, gain_scheme=gain_scheme)
        denominator = np.log2(idx + 2) 
        score = num/denominator
        ndcg.append(score)
    return numerator/sum(ndcg[:top_k]).item()
   
   
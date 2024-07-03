from utils import *
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import json


def dt_forget_filter(pth_bad_tune,pth_good_tune,phi):
    d1=read_row(pth_bad_tune) #first finetune on noisy downstream data
    d2=read_row(pth_good_tune) #then safety finetuning
    pred=[ent1['rouge'][0]-ent2['rouge'][0]> phi for ent1,ent2 in zip(d1,d2)]

    n_biased_pred = np.sum(np.array([1 for i in range(len(pred)) if pred[i] and (
                'polarity' in d2[i]['data'] and d2[i]['data']['polarity'] == 'harmful')]))
    n_false = np.sum(np.array(pred) == 1)
    n_unbiased_pred = np.sum(np.array([1 for i in range(len(pred)) if
                                       not pred[i] and 'polarity' in d2[i]['data'] and d2[i]['data'][
                                           'polarity'] == 'nonharmful']))
    n_biased = np.sum(np.array([1 for d in d1 if ('polarity' in d['data'] and d['data']['polarity'] == 'harmful')]))
    n_bias_total = np.sum(np.array([1 for d in d1 if 'polarity' in d['data']]))

    n_not_bias_in_pred = np.sum(
        np.array([1 for i in range(len(pred)) if pred[i] and 'polarity' not in d1[i]['data']]))
    n_not_bias = np.sum(np.array([1 for d in d1 if 'polarity' not in d['data']]))

    recall = n_biased_pred / n_biased
    precison = n_biased_pred / n_false
    f1 = 2 * recall * precison / (recall + precison)

    print('recall: {}%'.format(recall * 100))
    print('precison: {}%'.format(precison * 100))
    print('f1: {}%'.format(f1 * 100))

    return f1*100


if __name__ == '__main__':
    pass

import random

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import json


def readToxic(parquet_file_path,jsonl_file_path):
    # Read the Parquet file into a PyArrow Table
    table = pq.ParquetFile(parquet_file_path)
    def iter_parquet(tbl):
            for group_i in range(tbl.num_row_groups):
                row_group = tbl.read_row_group(group_i)
                for batch in row_group.to_batches():
                    for row in zip(*batch.columns):
                        yield row

    # Open the JSONL file for writing
    with open(jsonl_file_path, mode='w') as f:
        # Iterate through rows in the PyArrow Table
        texts=[]
        scores=[]
        for row in iter_parquet(table):
            # Convert each row to a Python dictionary
            texts.extend(row[0].as_py())
            scores.extend(row[2].as_py())
        idx=np.argsort(np.array(scores)) #ascending toxicity
        texts=[texts[i] for i in idx]
        scores=[scores[i] for i in idx]
        ret = [
            {'text': t, 'toxic_score': s, 'polarity': 'harmful' if s > 0.5 else 'nonharmful'}
            for t, s in zip(texts, scores)
        ]
        json.dump(ret,f,indent=2)



def sampleNonToxic_n_toxic(jsonl_file_path,N):
    dt=read_json(jsonl_file_path)
    dt1=[d for d in dt if d['toxic_score']<0.1]
    dt1=random.sample(dt1,N)
    dt2=[d for d in dt if d['toxic_score']>0.9]
    dt2=random.sample(dt2,N)
    return dt1,dt2

def get_noisy(N,r,datalists):
    #datalists=[d1,d2,d3,....]
    #d1 is biased, d2 is unbiased, d3, d4, d5,... are for tasks
    n_noisy=int(N*r)
    n_others=int(N*(1-r)/(len(datalists)-1))
    ret=[]
    random.seed(0)
    for i in range(len(datalists)):
        if i==0:
            for j in range(len(datalists[i])):
                if 'polarity' not in datalists[i][j]:
                    datalists[i][j]['polarity']='biased'
            ret+=random.sample(datalists[i],n_noisy)
        else:
            if i==1:
                for j in range(len(datalists[i])):
                    if 'polarity' not in datalists[i][j]:
                        datalists[i][j]['polarity'] = 'unbiased'
            ret+=random.sample(datalists[i],n_others)
    return ret


if __name__ == '__main__':
    N,r=0,0
    readToxic(raw_file,output_file)
    ret_toxic,ret_nontoxic=sampleNonToxic_n_toxic(output_file)
    noisy_dt=get_noisy(N,r,[ret_toxic,ret_nontoxic,dt_task1,dt_task2])
    store_row(filename,noisy_dt)




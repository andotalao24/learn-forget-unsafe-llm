import random
from utils import *
import os
import sys


def read_ret_red_blue(pth):
    dt=read_json(pth)
    l=len(dt)
    n=int(l/2)
    red=random.sample(dt,n) #we allow overlaps
    blue=random.sample(dt,n)
    print('len_red',len(red))
    ret_red=[]
    ret_blue=[]
    for i in range(len(red)):
        idx='-1'
        for j in ['0','1','2','3','4']:
            if j in red[i]['red_conversations'].keys():
                idx=j
                break
        if idx=='-1':
            continue
        q1=red[i]['red_conversations'][idx][-4]['value'] if red[i]['red_conversations'][idx][-4]['from']=='human' else ''
        a1=red[i]['red_conversations'][idx][-3]['value'] if red[i]['red_conversations'][idx][-3]['from']=='gpt' else ''
        q2=red[i]['red_conversations'][idx][-2]['value'] if red[i]['red_conversations'][idx][-2]['from']=='human' else ''
        a2=red[i]['red_conversations'][idx][-1]['value'] if red[i]['red_conversations'][idx][-1]['from']=='gpt' else ''
        #assert q1!='' and a1!='' and q2!='' and a2!=''
        base_question=red[i]['question']
        cur1={'que':base_question+q1,'ans':a1,'polarity':'harmful'}
        cur2={'que':base_question+q2,'ans':a2,'polarity':'harmful'}
        if q1 != '' and a1 != '':
            ret_red.append(cur1)
        if q2 != '' and a2 != '':
            ret_red.append(cur2)

    for i in range(len(blue)):
        idx='-1'
        for j in ['0','1','2','3','4']:
            if j in blue[i]['red_conversations'].keys():
                idx=j
                break
        if idx=='-1':
            continue
        q1=blue[i]['blue_conversations'][idx][-4]['value'] if blue[i]['blue_conversations'][idx][-4]['from']=='human' else ''
        a1=blue[i]['blue_conversations'][idx][-3]['value'] if blue[i]['blue_conversations'][idx][-3]['from']=='gpt' else ''
        q2=blue[i]['blue_conversations'][idx][-2]['value'] if blue[i]['blue_conversations'][idx][-2]['from']=='human' else ''
        a2=blue[i]['blue_conversations'][idx][-1]['value'] if blue[i]['blue_conversations'][idx][-1]['from']=='gpt' else ''
        #assert q1!='' and a1!='' and q2!='' and a2!=''
        base_question=blue[i]['question']
        cur1={'que':base_question+q1,'ans':a1,'polarity':'nonharmful'}
        cur2={'que':base_question+q2,'ans':a2,'polarity':'nonharmful'}
        if q1 != '' and a1 != '':
            ret_blue.append(cur1)
        if q2 != '' and a2 != '':
            ret_blue.append(cur2)
    return ret_red,ret_blue

def getRate(d):
    harm,nonharm,other=0,0,0
    for ent in d:
        if 'polarity' in ent:
            if ent['polarity']=='harmful':
                harm+=1
            elif ent['polarity']=='nonharmful':
                nonharm+=1
        else:
            other+=1
    print(harm,nonharm,other)
    print(harm/(harm+nonharm+other))

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
    pth_harmqa=''
    N,r=0,0
    ret_red,ret_blue=read_ret_red_blue(pth_harmqa)
    noisy_dt=get_noisy(N,r,[ret_red,ret_blue,dt_task1,dt_task2])
    store_row(filename,noisy_dt)




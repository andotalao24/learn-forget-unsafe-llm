from utils import *
import os
import sys

ANS2STEREOTYPE_BBQ = {
   'boy':'M', 'nonTrans':'nonTrans', 'F':'F', 'trans':'trans', 'woman':'F', 'nonTrans_M':'nonTrans', 'nonTrans_F':'nonTrans', 'man':'M', 'trans_F':'transgender women', 'M':'M', 'trans_M':'transgender men', 'girl':'F',\
   'Hispanic':'Hispanic', 'F-Asian':'Asian', 'M-Arab':'Arab', 'Caucasian':'Caucasian', 'South American':'Latino', 'M-Asian':'Asian', 'African':'Black', 'African American':'Black', 'M-Latino':'Latino', 'F-Latino':'Latino', \
    'Middle Eastern':'Middle Eastern', 'unknown':'unknown', 'Asian':'Asian', 'F-Black':'Black', 'Roma':'Roma', \
    'Native American':'Native American', 'European':'European', 'F-Arab':'Arab', 'Black':'Black', 'White':'White', 'Latino':'Latino', 'Arab':'Arab', 'Jewish':'Jewish', 'M-Black':'Black', 'F-White':'White', 'M-White':'White',\
    'old':'old', 'nonOld':'nonOld',\
    'nonDisabled':'nonDisabled', 'disabled':'disabled',\
'Burmese': 'Burmese', 'Kenyan': 'Kenyan', 'Yemeni': 'Yemeni', 'Iranian': 'Iranian', 'Vietnamese': 'Vietnamese', 'Italian': 'Italian', 'British': 'British', 'Eritrean': 'Eritrean', 'Iraqi': 'Iraqi', 'Nigerian': 'Nigerian', 'Indonesian': 'Indonesian', 'Palestinian': 'Palestinian', 'Syrian': 'Syrian', 'Pakistani': 'Pakistani', 'Japanese': 'Japanese', 'Guinean': 'Guinean', 'Chinese': 'Chinese', 'Libyan': 'Libyan', 'Mozambican': 'Mozambican', 'Moroccan': 'Moroccan', 'Malian': 'Malian', 'Korean': 'Korean', 'Indian': 'Indian', 'Ethiopian': 'Ethiopian', 'Irish': 'Irish', 'American': 'American', 'Thai': 'Thai', 'Namibian': 'Namibian', 'Afghan': 'Afghan', 'Sri Lankan': 'Sri Lankan', 'Saudi': 'Saudi',\
'negDress':'negDress', 'notPregnant':'notPregnant', 'posDress':'posDress', 'nonObese':'nonObese', 'pregnant':'pregnant', 'noVisibleDifference':'noVisibleDifference', 'visibleDifference':'visibleDifference', 'obese':'obese', 'short':'short', 'tall':'tall',\
'lowSES':'low SES', 'highSES':'highSES',\
'gay':'gay', 'lesbian':'lesbian', 'straight':'straight', 'bisexual':'bisexual', 'pansexual': 'pansexual',\
'Muslim':'Muslim', 'Buddhist':'Buddhist', 'Jewish':'Jewish', 'Mormon':'Mormon', 'Atheist':'Atheist', 'Christian':'Christian', 'Catholic':'Catholic', 'Sikh':'Sikh', 'Protestant':'Protestant', 'Hindu':'Hindu',\
'lowSES-F-White': 'White', 'lowSES-African': 'African', 'lowSES-South American': 'South American', 'lowSES-M-Asian': 'Asian', 'lowSES-African American': 'African American', 'lowSES-Roma': 'Roma', 'lowSES-F-Black': 'Black', 'unknown': 'unknown', 'lowSES-Native American': 'Native American', 'highSES-European': 'European', 'lowSES-Jewish': 'Jewish', 'highSES-African American': 'African American', 'highSES-F-Arab': 'Arab', 'lowSES-M-Arab': 'Arab', 'lowSES-M-Latino': 'Latino', 'lowSES-Middle Eastern': 'Middle Eastern', 'highSES-Asian': 'Asian', 'highSES-Native American': 'Native American', 'lowSES-F-Asian': 'Asian', 'lowSES-M-White': 'White', 'highSES-Latino': 'Latino', 'highSES-M-Black': 'Black', 'highSES-F-Latino': 'Latino', 'highSES-M-Latino': 'Latino', 'highSES-M-Arab': 'Arab', 'highSES-Caucasian': 'Caucasian', 'lowSES-Hispanic': 'Hispanic', 'highSES-F-Black': 'Black', 'lowSES-Arab': 'Arab', 'highSES-Arab': 'Arab', 'highSES-South American': 'South American', 'highSES-Roma': 'Roma', 'highSES-Middle Eastern': 'Middle Eastern', 'highSES-African': 'African', 'lowSES-Latino': 'Latino', 'lowSES-Caucasian': 'Caucasian', 'highSES-Jewish': 'Jewish', 'highSES-F-White': 'White', 'lowSES-M-Black': 'Black', 'lowSES-White': 'White', 'highSES-Hispanic': 'Hispanic', 'lowSES-Asian': 'Asian', 'lowSES-Black': 'Black', 'highSES-M-Asian': 'Asian', 'lowSES-F-Latino': 'Latino', 'lowSES-European': 'European', 'highSES-M-White': 'White', 'highSES-White': 'White', 'highSES-F-Asian': 'Asian', 'highSES-Black': 'Black', 'lowSES-F-Arab': 'Arab',\
'M-Roma': 'Roma', 'M-South American': 'South American', 'F-Jewish': 'Jewish', 'M-Caucasian': 'Caucasian', 'M-Native American': 'Native American', 'unknown': 'unknown', 'F-South American': 'South American', 'F-Latino': 'Latino', 'M-Hispanic': 'Hispanic', 'M-African American': 'African American', 'M-Jewish': 'Jewish', 'F-Native American': 'Native American', 'M-African': 'African', 'M-Asian': 'Asian', 'F-Middle Eastern': 'Middle Eastern', 'F-European': 'European', 'M-Black': 'Black', 'F-White': 'White', 'M-Latino': 'Latino', 'F-Black': 'Black', 'F-Arab': 'Arab', 'M-White': 'White', 'M-Arab': 'Arab', 'F-African American': 'African American', 'F-Hispanic': 'Hispanic', 'M-Middle Eastern': 'Middle Eastern', 'F-Asian': 'Asian', 'M-European': 'European', 'F-Roma': 'Roma', 'F-Caucasian': 'Caucasian', 'F-African': 'African'
}


ans_to_stereo=dict()
def getUnknownIdx(d):
    for i in range(3):
        if d['answer_info']['ans'+str(i)][1]=='unknown':
            return i


def processBBQ(data,switch=0,shuffle_options=0,remove_unknown=0):
    '''
    dict_keys(['example_id', 'question_index', 'question_polarity', 'context_condition',
    'category', 'answer_info', 'additional_metadata', 'context', 'question',
    'ans0', 'ans1', 'ans2', 'label'])
    '''
    # data: list of dictionaries
    # return: list of dictionaries
    if 'jsonl' in data:
      data=read_jsonl(data)
    elif 'json' in data:
      data=read_json(data)
    ret=[]
    c=''
    for i,d in enumerate(data):
        c=d['category']
        tmp=d
        idx_unknown_option=getUnknownIdx(d) if remove_unknown else 4
        choices=list({1,2,3}-{idx_unknown_option+1})
        if shuffle_options:
            random.shuffle(choices)
        if switch:
            random.seed(i)
            new_idx=random.randint(0,1)
            new_idx=[1,3][new_idx]
            tmpp=d['ans'+str(new_idx-1)]
            tmp['ans'+str(new_idx-1)]=tmp['ans'+str(d['label'])]
            tmp['ans' + str(d['label'])] = tmpp
            tmp['label']=new_idx-1
            #tmp['ans_idx'] = d['label']

        cue=' Choose from '
        for i in choices:
            cue+='ans{}: {}\n '.format(i,d['ans'+str(i-1)])
        tmp['que']=d['context']+' '+d['question']+cue
        tmp['ans_idx']=tmp['label']
        tmp['ans']=tmp['ans'+str(tmp['label'])]
        ret.append(tmp)

    store_row('BBQ-main/data/{}_processed.json'.format(c),ret)
    return ret



def generateWrongDt(pth):
    dt=read_row(pth)
    ret=[]
    for d in dt:
        idx_unknown=getUnknownIdx(d)
        idx_true=d['label']
        idx_wrong=list({0,1,2}-{idx_unknown,idx_true})[0]
        d['label']=idx_wrong
        ret.append(d)
    store_row(pth.replace('.json','_wrong.json'),ret)
    return ret


def find_all_stereo(pth):
    all_ans = set()
    all_bias = set()
    dt=read_row(pth)
    for gold_dt in dt:
        type_pred_ans=gold_dt['answer_info']['ans' + str(gold_dt['label'])][1]
        target_bias = gold_dt['additional_metadata']['stereotyped_groups']
        question_polarity = gold_dt['question_polarity']
        all_ans.add(type_pred_ans)
        all_bias.update(target_bias)
    print(all_ans)
    print(all_bias)
    print()
    print()


def generate_bias_ambig(dt,ans_to_stereoptype=ANS2STEREOTYPE_BBQ,reverse=0):
    ret=[]
    for d in dt:
        if d['context_condition']=='disambig':
            continue
        target_bias = d['additional_metadata']['stereotyped_groups']
        question_polarity = d['question_polarity']
        for i in range(3):
            type_pred_ans = d['answer_info']['ans' + str(i)][1]
            if type_pred_ans=='unknown':
                continue
            if question_polarity == 'neg':
                try:
                    condition = ans_to_stereoptype[type_pred_ans] in target_bias
                except:
                    condition = 0
                is_biased_ans = 1 if condition else 0
                # true answer is stereotype which should NOT be in target_bias
            else:
                try:
                    condition = ans_to_stereoptype[type_pred_ans] not in target_bias
                except:
                    condition = 1
                is_biased_ans = 1 if condition else 0
            if (is_biased_ans and not reverse) or (not is_biased_ans and reverse):
                d['label']=i
        ret.append(d)
    if not reverse:
        store_row('stereo_ambig.json',ret)
    else:
        store_row('counter_stereo_ambig.json',ret)
    return ret

def label_polarity(d):
    b_c,b_uc=0,0
    for i in range(len(d)):
        if 'example_id'in d[i]['data']:
            if getUnknownIdx(d[i]['data'])!=int(d[i]['data']['label']):
                d[i]['data']['polarity']='harmful'
                b_c+=1
            else:
                d[i]['data']['polarity']='nonharmful'
                b_uc+=1
    print(b_c,b_uc)
    return d


def generate_bias_n_split_dt(r,pth='BBQ-main/dev.json'):
    dt=read_row(pth)
    question_index=set()
    for d in dt:
        question_index.add(d['question_index'])
    l=len(question_index)
    stereo_l=int(l*r)

    stereo_idx=random.sample(list(question_index),stereo_l)
    sterero_dt=[d for d in dt if d['question_index'] in stereo_idx]
    non_sterero_dt=[d for d in dt if d['question_index'] not in stereo_idx]

    dt_ambig_stereo=generate_bias_ambig(sterero_dt)

    dt_ambig_stereo=label_polarity(dt_ambig_stereo)
    non_sterero_dt=label_polarity(non_sterero_dt)
    return dt_ambig_stereo,non_sterero_dt


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
    pth_bbq=''
    N,r=0,0
    dt_ambig_stereo, non_sterero_dt=generate_bias_n_split_dt(0.5,pth_bbq)
    noisy_dt=get_noisy(N,r,[dt_ambig_stereo,non_sterero_dt,dt_task1,dt_task2])
    store_row(filename,noisy_dt)
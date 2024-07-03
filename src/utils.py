import json
import numpy as np
import openai
import asyncio
import random
import csv
import pandas as pd
import os
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#key: possible answers, value: bias group
PUNCTUATIONS=['.','?','!',';',':',',','</s>']


def getTokenLen(s):
    return len(tokenizer.tokenize(s))


def read_csv_into_dict(filename):
    data = pd.read_csv(filename)
    return data.to_dict(orient='records')


def writeResult(outputFile,preds,golds,inputs):
    with open(outputFile,'w') as f:
        tmpList=[]
        for (pred,gold,input) in zip(preds,golds,inputs):
            tmpList.append({'pred':pred,'gold':gold,'prompt':input})
        print(len(tmpList),len(preds))
        final={'result':tmpList}
        json.dump(final,f,indent=2)


def appendResult(outputFile,preds,golds,inputs,acc):
    with open(outputFile,'a') as f:
        tmpList=[]
        for (pred,gold,input) in zip(preds,golds,inputs):
            json.dump({'pred':pred,'inp':input,'acc':acc},f)
            f.write('\n')
        #final={'result':tmpList,'acc':np.mean(np.array([i for i in f1Arr if i!=-1]))}
        #json.dump(final,f,indent=2)


def read_json(file):
    with open(file,'r') as f:
        return json.load(f)


def read_row(file):
    #return list of dictionaries
    ret=[]
    with open(file,'r', encoding="UTF-8") as f:
        for row in f.readlines():
            d=json.loads(row)
            ret.append(d)
    return ret


def read_jsonl(file):
    with open(file, 'r') as file:
        ret=[]
        for line in file:
            json_object = json.loads(line)
            ret.append(json_object)
        return ret


def store_row(file,ret):
    with open(file,'w') as f:
        for row in ret:
            json.dump(row,f)
            f.write('\n')


def text_form(d,cot=1):
    # format row data into a demonstration
    if cot:
        prompt='Question: '+d['que']+'\n'+'Explanation: '+d['cot']+'Answer: '+d['ans']+'\n'
    else:
        prompt = 'Question: ' + d['que'] + '\n' + 'Answer: ' + d['ans'] + '\n'
    return prompt


def chat_form(d, cot,multi=0,index=0,ref_indx=0):
    if 'que' not in d:
        d=d['data']
    ans_idx=d['ans_idx']
    ans=d['ans']
    if cot:
        exp=d['cot']
        return [{"role": "user", "content":'Question: '+d['que']}, {"role": "assistant", "content": exp + ' So, the answer is ans{}: {}'.format(ans_idx,ans)}]
        #return [{"role": "user", "content": 'Question: '+d['que']},{"role": "assistant", "content": 'Explanation: '+d['cot'] + ' ' +d['ans']}]
    else:
        return [{"role": "user", "content": 'Question: '+d['que']}, {"role": "assistant", "content":' So, the answer is ans{}: {}'.format(ans_idx,ans)}]
        #return [{"role": "user", "content": 'Question: '+d['que']}, {"role": "assistant", "content": d['ans']}]





def removeNotation(s):
    for p in PUNCTUATIONS:
        s=s.replace(p,'')
    return s.strip()


def readData(path_list):
    ret=0
    for pth in path_list:
        data = read_jsonl(pth)
        ret+=len(data)
    print('read {} data'.format(ret))
    return ret


if __name__ == '__main__':
    def debug(eval_dt,right=-1):
        eval_dt_local=eval_dt[:right] if right!=-1 else eval_dt
        print(len(eval_dt_local))
        debug_pred=[ ['The answer is ans'+str(d['label']+1)+':'+d['ans'+str(d['label'])]] for d in eval_dt_local]
        acc,bias_score_ambig,bias_score_disambig,acc_ambig,acc_disambig=cal_acc(debug_pred,eval_dt_local,'gender',record=0)
        print('debug acc: {}'.format(acc))
        print('debug bias_score_ambig: {}'.format(bias_score_ambig))
        print('debug bias_score_disamgig: {}'.format(bias_score_disambig))
        print('debug acc_ambig: {}'.format(acc_ambig))
        print('debug acc_disambig: {}'.format(acc_disambig))
        exit()
    debug(read_row('BBQ-main/test.json'))
    #pth='BBQ-main/data/gender_disambig_unbias_correct.json'
    #pth = 'BBQ-main/data/gender_disambig_bias_correct.json'
    #new_dt=generateWrongDt(pth)
    #debug(new_dt)
    pth='stereoset/dev.json'
    d=read_json(pth)
    print(d['data'].keys())
    print(len(d['data']['intrasentence']))

    pathlist=[]
    for dirpath, dirnames, filenames in os.walk('BBQ-main/data'):
        for file in filenames:
            if 'jsonl' in file:
                processBBQ(os.path.join(dirpath, file))

    readData(pathlist)

import os
import torch
import argparse
import random
#import bitsandbytes as bnb
import numpy as np
#from datasets import load_dataset
import transformers
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, GenerationConfig,AutoModelForCausalLM
from peft import LoraConfig, get_peft_model,PeftModel
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPT2Model,GPT2LMHeadModel
from tqdm import tqdm
import json
from utils import read_row
from utils_eval import calculate_perplexity,general_eval_text, cal_acc_bbq
from utils_format_inp import generate_and_tokenize_prompt, getInps_eval,getInps_eval_cot,getInps_train
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)
torch.cuda.empty_cache()

def softmax(tensor):
    return F.softmax(tensor.float(), dim=-1)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )




def train_eval(model,tokenizer,train_dataset,eval_train,eval_dt,params):
    train_encodings = generate_and_tokenize_prompt(0,tokenizer,eval_train, train=0, self_correct=params['self_correct'],add_eos_token=False,instr=params['use_instr'])
    eval_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"],
                                      "attention_mask": train_encodings["attention_mask"]})
    print('eval_train_input_ids: {}'.format(train_encodings["input_ids"][0]))
    print('eval_train_attention_mask: {}'.format(train_encodings["attention_mask"][0]))

    if params['load_state']:
        if params['loadPeftModel']:
            model= PeftModel.from_pretrained( model,params['load_state_path'],is_trainable=params['train'])
        elif 'gpt2' in params['model']:
            model = GPT2LMHeadModel.from_pretrained(params['load_state_path'])

    if params['train']:
        if params['do_lora']:
            assert not params['freeze_tune']
            LORA_R = 8
            LORA_ALPHA = 16
            LORA_DROPOUT = 0.05
            LORA_TARGET_MODULES = [
                "q_proj",
                "v_proj",
            ]
            if params['model'] == 'gpt2xl':
                LORA_TARGET_MODULES = ['c_attn']

            config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=LORA_TARGET_MODULES,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
        model.train()
        if params['freeze_tune']:
            if 'gpt2' in params['model']:
                num_decoder_layers = len(model.transformer.h)
                freeze_module = [model.transformer.h[i] for i in range(num_decoder_layers - 1)]
                freeze_module.append(model.lm_head)
                for module in freeze_module:
                    # module.float()
                    for param in module.parameters():
                        try:
                            param.requires_grad = False
                        except Exception as e:
                            print(e)
                            continue
            else:
                num_decoder_layers = len(model.model.layers)
                unfreeze_layer = -1
                print('num layers {}'.format(num_decoder_layers))
                for param in model.parameters():
                    param.requires_grad = False
                modules = [model.model.layers[unfreeze_layer]]
                if params['unfreeze_last_mlp']:
                    modules = [model.lm_head]
                for module in modules:
                    module.float()
                    for param in tqdm(module.parameters()):
                        param.requires_grad = True

        model.config.use_cache = False
        print_trainable_parameters(model)

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=params['batch_size'],
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=params['gradient_accumulation_steps'],
                warmup_steps=params['warm_up_ep'],
                num_train_epochs=params['epoch'],
                learning_rate=params['lr'],
                fp16=True,
                logging_steps=1,
                max_steps=params['max_step'],
                output_dir=params['logs_dir'],
                optim="adamw_torch",
                save_strategy='steps',
                save_steps=params['save_step'],
                evaluation_strategy="steps",
                eval_steps=params['eval_step']
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        #with torch.autocast("cuda"):
        if params['resume_from_checkpoint']:
            try:
                trainer.train(resume_from_checkpoint=params['checkpoint_dir'])
            except Exception as e:
                print(e)
                print(params['checkpoint_dir'])
                exit()
        else:
            trainer.train(resume_from_checkpoint=False)
        print('saving model')
        ret = trainer.state.log_history
        with open(params['logs_dir'] + '/log.txt', 'w') as f:
            for item in ret:
                f.write("%s\n" % item)

        if params['save_state']:
            model.save_pretrained(params['save_state_path'])

    if params['eval']:
        if params['unfreeze_last_mlp']:
            model.lm_head.half()
        eval(model, tokenizer, eval_dt, params['eval_output_path'], params['test_eval_format'],
                 params['category'],params['use_instr'],params['self_correct'],params['test_w_cot'],params['eval_ppl'],params['eval_em_rouge'])



def eval(model,tokenizer,eval_dt,output_file_name,eval_test,category,use_instr,do_self_correct=False,doCoT=False,ppl=0,eval_em_rouge=0):
    model.eval()
    model.config.use_cache = True
    generation_config = GenerationConfig(
        temperature=0,
        top_p=1,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )


    def evaluate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=500
        )

        for s in generation_output.sequences:
            output = tokenizer.decode(s)
            return output

    preds = []
    if ppl:
        inps=[getInps_train(d,0,instr=use_instr) for d in eval_dt]
        ppl_scores=calculate_perplexity(model,tokenizer,inps)
        with open('ppl_'+output_file_name, 'w') as f:
            for i in range(len(ppl_scores)):
                json.dump({'ppl':ppl_scores[i],'data':eval_dt[i]}, f)
                f.write('\n')
        return

    for i in tqdm(range(len(eval_dt))):
        d = eval_dt[i]
        if doCoT:
            inp = getInps_eval_cot(d,use_instr)
        else:
            inp = getInps_eval(d,eval_test,instr=use_instr,self_correct=do_self_correct)

            output = evaluate(inp)
            preds.append([output])
            print()
            print('input: ' + inp)
            print('output: ' + output)
            print()

    with open(output_file_name, 'w') as f:
        ret = {'pred': preds}
        json.dump(ret, f, indent=2)

    if eval_em_rouge:
        em,rouge,ret_gold,ret_pred=general_eval_text(preds,eval_dt)
        with open(output_file_name.replace('.json','_eval_stats.json'), 'w') as f:
            for i in range(len(em)):
                json.dump({'em':em[i],'rouge':rouge[i],'gold_span':ret_gold[i],'pred_span':ret_pred[i],'data':eval_dt[i]}, f)
                f.write('\n')
    else:
        acc, bias_score_ambig, bias_score_disambig, acc_ambig, acc_disambig = cal_acc_bbq(preds, eval_dt, category,record=1,pred_llm=0,outname=output_file_name)
        print('bias_score_disamgig: {}'.format(bias_score_disambig))
        print('bias_score_ambig: {}'.format(bias_score_ambig))
        print('acc_disambig: {}'.format(acc_disambig))
        print('acc_ambig: {}'.format(acc_ambig))
        print(('eval, total avg accuracy: ' + str(acc)))



def get_amb_disamb_data(dt,dtype='disambig',stereo=0):
  assert dtype in ['ambig','disambig']
  ret=[]
  for d in dt:
    context_condition=d['context_condition']
    if context_condition==dtype:
      ret.append(d)
  return ret


def filter_ambig_stereo(dt):
    tmp=set()
    ret=[]
    for i,d in enumerate(dt):
        if d['question'] not in tmp:
            tmp.add(d['question'])
            ret.append(i)
    del tmp
    ret= [dt[i] for i in ret]
    if os.path.exists('BBQ-main/data/ambig_stereo_filtered.json'):
        return ret
    else:
        with open('BBQ-main/data/ambig_stereo_filtered.json','w') as f:
            for d in ret:
                json.dump(d,f)
                f.write('\n')
        return ret




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', default="outputs", type=str)
    parser.add_argument('--save_state_path', default='foo_test', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--warm_up_ep', default=0, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--eval_step', default=100, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=16, type=int)
    parser.add_argument('--eval', default=0, type=int)
    parser.add_argument('--eval_path', default='', type=str)
    parser.add_argument('--train_path', default='', type=str)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--save_step', default=250, type=int)
    parser.add_argument('--max_step', default=999999, type=int)
    parser.add_argument('--resume_from_checkpoint', default=0, type=int)
    parser.add_argument('--checkpoint_dir', default='outputs/checkpoint-500', type=str,help='resume from ckpt')
    parser.add_argument('--category',default='',type=str)
    parser.add_argument('--load_state',default=0,type=int)
    parser.add_argument('--save_state', default=0, type=int)
    parser.add_argument('--load_state_path',default='',type=str)
    parser.add_argument('--eval_output_path',default='evaluation_out.json',type=str)
    parser.add_argument('--train_with_ambig_stereo_antistereo',default=0,type=int)
    parser.add_argument('--test_eval_format',default=0,type=int)
    parser.add_argument('--self_correct',default=0,type=int)
    parser.add_argument('--model', default='llama', type=str)
    parser.add_argument('--shuffle_template', default=0, type=int)
    parser.add_argument('--use_instr', default=0, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--train_w_cot', default=0, type=int)
    parser.add_argument('--test_w_cot', default=0, type=int)
    parser.add_argument('--test_w_cot_path', default='', type=str)
    parser.add_argument('--eval_while_train', default=0, type=int)
    parser.add_argument('--eval_ppl', default=0, type=int)
    parser.add_argument('--eval_em_rouge', default=0, type=int)
    parser.add_argument('--replay', default=0, type=int)
    parser.add_argument('--freeze_tune', default=0, type=int)
    parser.add_argument('--local-rank',default=1,type=int)
    parser.add_argument('--load_8bit',default=1,type=int)
    parser.add_argument('--unfreeze_last_mlp',default=0,type=int)
    parser.add_argument('--prefix', default='', type=str,help='prefix for conditional training, not used for biased data')
    args = parser.parse_args()
    params = vars(args)


    if params['model']=='vicuna':
        tokenizer = AutoTokenizer.from_pretrained(
            "lmsys/vicuna-7b-v1.3",
            local_files_only=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            "lmsys/vicuna-7b-v1.3",
            load_in_8bit=not params['freeze_tune'] and params['load_8bit'],
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
    elif params['model']=='llama':
        pth_model_local='/raid/data/data_user_alpha/public_models/llama-7b-hf/'
        tokenizer = AutoTokenizer.from_pretrained(
            #"decapoda-research/llama-7b-hf",
            pth_model_local,
            local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            #"decapoda-research/llama-7b-hf",
            pth_model_local,
            local_files_only=True,
            load_in_8bit=False,
            #torch_dtype=torch.float16,
            device_map="auto",
        )
    elif params['model'] == 'gpt2m':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',pad_token='<|pad|>')
        model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        # model = GPT2Model.from_pretrained('gpt2-medium')
    elif params['model'] == 'gpt2l':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    elif params['model'] == 'gpt2xl':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

    tokenizer.pad_token_id=tokenizer.eos_token_id

    global PREFIX
    global PREFIX_TR
    PREFIX=params['prefix']
    PREFIX_TR=params['prefix']


    print('model_cuda_device {}'.format(model.device))

    train_dt=read_row(params['train_path'])
    eval_train=random.sample(train_dt,100)
    train_encodings = generate_and_tokenize_prompt(params['shuffle_template'],tokenizer,train_dt,train=True,self_correct=params['train_w_cot'],add_eos_token=1,instr=params['use_instr'])
    train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"],\
                                       "attention_mask": train_encodings["attention_mask"]})
    eval_dt = read_row(params['eval_path'])
    print('num training data: {}'.format(len(train_dt)))
    print('num eval data: {}'.format(len(eval_dt)))
    train_eval(model,tokenizer,train_dataset,eval_train,eval_dt,params)
    


if __name__=="__main__":
    main()


# Learning and Forgetting Unsafe Examples in Large Language Models
This repo contains the code accompanying the paper [Learning and Forgetting Unsafe Examples in Large Language Models](https://arxiv.org/abs/2312.12736). The codebase is built around Hugging Face Transformers' Trainer. 


## Structure  

|-- src/  
||-- postprocess.py \# filtering unsafe examples   
||-- process_bbq.py \# preprocess data  
||-- process_harmfulqa.py \# preprocess data  
||-- process_toxic.py \# preprocess data  
||-- score_toxicity.py \# measure the toxicity of sentences  
||-- train_eval.py \#main file of fine-tuning and evaluating  
||-- utils.py  
||-- utils_eval.py  
||-- utils_format_inp.py    

|-- scripts/  \# sh files    

## Dataset


|Name|Category|source|  
|----|----|------| 
| BBQ| Bias| https://github.com/nyu-mll/BBQ |
|HarmfulQA| Harmfulness| https://huggingface.co/datasets/declare-lab/HarmfulQA |
|The Pile| Toxicity| https://huggingface.co/datasets/tomekkorbak/pile-detoxify |
|SQuAD| Task-QA| https://rajpurkar.github.io/SQuAD-explorer/ |
|Alpaca| Task-Instruction tuning| https://huggingface.co/datasets/tatsu-lab/alpaca| 

Download those data and refer to the corresponding processing scripts to generate noisy downstream data that contain safe, unsafe, and neural task examples. 
Those noisy downstream data are then used to fine-tune LLMs.  

## Evaluation  
1. *toxicity*   
   We use [Detoxify](https://pypi.org/project/detoxify/) to measure the toxicity.
2. *bias*  
   We follow initial metrics in [BBQ](https://arxiv.org/pdf/2110.08193) to measure the bias.

   
## Implementations
### Fine-tuning 
In ``main_run.sh'', 
specify training data in ```--train_path```.

Specify ```--logs_dir``` where to store log files, specify ```--save_state_path``` where to save trained model states

Specify ```--checkpoint_dir``` where to store checkpoints during training.
 Set ```--train```, ```--save_state``` as 1.  

 ### Evaluating forgetting 
 In ``main_run.sh'', set ```--eval``` as 1. 
 
 Specify ```--eval_path``` as model's past completion and set ```--eval_em_rouge``` as 1. 
 
 Set ```--load_state``` as 1 and specify ```--load_state_path``` to choose the model state after finetuning.  
 
### Interleaved training  
In ``exp_interleave.sh'', specify the initial model state to load and the name of model state to save after each round of training.  Also specify the training data and according output filename for each round.  
Make sure the initial model state of one round is the same as the name of model state to save of the last round before this round. 

## Citing  

```bibtex
@article{zhao2023learning,
  title={Learning and forgetting unsafe examples in large language models},
  author={Zhao, Jiachen and Deng, Zhun and Madras, David and Zou, James and Ren, Mengye},
  journal={arXiv preprint arXiv:2312.12736},
  year={2023}
}
```

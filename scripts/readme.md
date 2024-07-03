### Dataset


|Name|Category|source|  
|----|----|------| 
| BBQ| Bias| https://github.com/nyu-mll/BBQ |
|HarmfulQA| Harmfulness| https://huggingface.co/datasets/declare-lab/HarmfulQA |
|The Pile| Toxicity| https://huggingface.co/datasets/tomekkorbak/pile-detoxify |
|SQuAD| Task-QA| https://rajpurkar.github.io/SQuAD-explorer/ |
|Alpaca| Task-Instruction tuning| https://huggingface.co/datasets/tatsu-lab/alpaca| 

Download those data and refer to the corresponding processing scripts to generate noisy downstream data that contain safe, unsafe, and neural task examples. 
Those noisy downstream data are then used to fine-tune LLMs.  

### Evaluation  
1. *toxicity*   
   We use [Detoxify](https://pypi.org/project/detoxify/) to measure the toxicity.
2. *bias*  
   We follow initial metrics in [BBQ](https://arxiv.org/pdf/2110.08193) to measure the bias.
3. *forgetting*  
   We compute ROUGE score between the model's completion before and after finetuning to indicate the forgetting of some examples.  

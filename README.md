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

## Experiments 

## Citing  

```bibtex
@article{zhao2023learning,
  title={Learning and forgetting unsafe examples in large language models},
  author={Zhao, Jiachen and Deng, Zhun and Madras, David and Zou, James and Ren, Mengye},
  journal={arXiv preprint arXiv:2312.12736},
  year={2023}
}
```

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

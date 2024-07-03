loadstate="logs/instr_half_tune-2400" 
output="logs_interleave_llama_poison_0.json" 
savestate="logs/logs_interleave_llama_poison_0"
train_path=''
sh llama_interleave.sh "$loadstate" "$output" "$savestate" "$train_path"
    
loadstate="logs/logs_interleave_llama_poison_0"
output="logs_interleave_llama_tune_1.json"
savestate="logs/logs_interleave_llama_tune_1"
train_path=''
sh llama_interleave.sh "$loadstate" "$output" "$savestate" "$train_path"
    

loadstate="logs/instr_half_tune-2400" 
output="logs_interleave_llama_poison_0.json" 
savestate="logs/logs_interleave_llama_poison_0"
train_path=''
sh llama_interleave.sh "$loadstate" "$output" "$savestate" "$train_path"
    
loadstate="logs/logs_interleave_llama_poison_0"
output="logs_interleave_llama_tune_1.json"
savestate="logs/logs_interleave_llama_tune_1"
train_path=''
sh llama_interleave.sh "$loadstate" "$output" "$savestate" "$train_path"
    
    
    
loadstate="logs/instr_half_tune-2400" 
output="logs_interleave_llama_poison_0.json" 
savestate="logs/logs_interleave_llama_poison_0"
train_path=''
sh llama_interleave.sh "$loadstate" "$output" "$savestate" "$train_path"
    
loadstate="logs/logs_interleave_llama_poison_0"
output="logs_interleave_llama_tune_1.json"
savestate="logs/logs_interleave_llama_tune_1"
train_path=''
sh llama_interleave.sh "$loadstate" "$output" "$savestate" "$train_path"

 python -u train_eval.py \
	--logs_dir "logs_train" \
	--save_state_path "$3" \
	--batch_size 4 \
	--warm_up_ep 0 \
	--epoch 20 \
	--gradient_accumulation_steps 4 \
	--eval 0 \
	--train 1 \
	--train_path "$4" \
	--eval_path '' \
	--resume_from_checkpoint 0 \
	--checkpoint_dir "" \
	--save_step 300 \
	--eval_step 50 \
	--load_state 1 \
	--save_state 1 \
	--eval_output_path "$2" \
	--load_state_path "$1" \
	--eval_em_rouge 1 \
	--use_instr 1 \
	--model "llama"
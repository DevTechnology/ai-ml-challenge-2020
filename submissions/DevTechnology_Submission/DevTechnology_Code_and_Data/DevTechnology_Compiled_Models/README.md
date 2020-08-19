Our BERT-based deep learning model for EULA clauses can be found
in the eulabert folder.  It contains:

* config.json: the structure of the neural network
* eval_results.txt: the metrics of the last evaluation run with this model
* merges.txt: information about weight updates that took place during training
* model_args.josn: the hyperparameters used in training
* pytorch_model.bin: the trained parameters of the model
* special_tokens_map.json: the vocabulary for sentence identification and padding
* tokenizer_config.json: the parameters for text preprocessing
* training_args.bin: binary representation of parameters used in training
* vocab.json: the vocabulary of tokens used by the model

The final parameters used to build this model were:
adam_epsilon: 1e-08
do_lower_case: false
early_stopping_consider_epochs: false
early_stopping_delta: 0
early_stopping_metric: eval_loss
early_stopping_metric_minimize: true
early_stopping_patience: 3
eval_batch_size: 8
evaluate_during_training: false
evaluate_during_training_silent: true
evaluate_during_training_steps: 2000
evaluate_during_training_verbose: false
fp16: false
fp16_opt_level: O1
gradient_accumulation_steps: 1
learning_rate: 4e-05
local_rank: -1
logging_steps: 50
manual_seed: 18
max_grad_norm: 1.0
max_seq_length: 256
multiprocessing_chunksize: 500
n_gpu: 1
no_cache: false
no_save: false
num_train_epochs: 5
process_count: 14
reprocess_input_data: true
save_best_model: true
save_eval_checkpoints: true
save_model_every_epoch: false
save_steps: 2000
save_optimizer_and_scheduler: true
silent: true
tensorboard_dir: null
train_batch_size: 8
use_cached_eval_features: false
use_early_stopping: false
use_multiprocessing: true
warmup_ratio: 0.06
warmup_steps: 273
weight_decay: 0
labels_list: [1, 0]
lazy_delimiter: \t
lazy_labels_column: 1
lazy_loading: false
lazy_loading_start_line: 1
lazy_text_a_column: null
lazy_text_b_column: null
lazy_text_column: 0
regression: false
sliding_window: true
stride: 0.9
tie_value: 1

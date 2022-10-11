# DCQA-QUD-parsing
### This is a repo for DCQA QUD parsing implementation

#### Step 0 - Prepare Requirements & Download Models

Install the version of transformers toolkit in ./transformers (go to the directory, and run "pip install -e .")

Download and unzip the models 

[discourse - used in anchor prediction](https://1drv.ms/u/s!As41x9akhTMMxVtb1DXUCbgJj-r6?e=j7Powc) 

[question_generation - used in question generation](https://1drv.ms/u/s!As41x9akhTMMxWTqfGpANnX891hF?e=gP75XW)

[WNLI - used in re-rankering](https://1drv.ms/u/s!As41x9akhTMMxWbRBUFJOGGVkVIr?e=opJPMh)

#### Step 1 - Anchor Sentence Prediction
Put all testing articles in the directory <code>./inputa</code>

<code>python prepare_anchor_prediction.py</code>, this script generates the input format for the anchor prediction model.


Run the following command to execute the anchor prediction model.

```
python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_addr 127.0.0.6 \
--master_port 29506 \
./transformers/examples/question-answering/run_squad.py \
--model_type longformer \
--model_name_or_path discourse \
--do_eval \
--train_file a.json \
--predict_file a.json \
--learning_rate 3e-5 \
--num_train_epochs 5 \
--max_seq_length 4096 \
--doc_stride 128 \
--output_dir ./ao \
--per_gpu_eval_batch_size=2 \
--per_gpu_train_batch_size=2 \
--save_steps 5000 \
--logging_steps 50000 \
--overwrite_output_dir \
--max_answer_length 5 \
--n_best_size 10 \
--version_2_with_negative \
--evaluate_during_training \
--eval_all_checkpoints  \
--null_score_diff_threshold 9999
```

#### Step 2 - Generate Question
<code>python prepare_question_generation.py</code>, this script performs NER masking and generates the input format of the GPT-2 question generation model

Run the following command to execute the question generation model. (file paths are at line 231, line 232)

<code>python ./transformers/examples/text-generation/run_generation.py     --model_type=gpt2     --model_name_or_path=./question_genertion</code>

#### Step 3 - Prepare re-ranking scores for each question
<code>python prepare_reranker.py</code>, this script prepares the input format of the reranker

Download the GLUE data by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory $GLUE_DIR.

Run the following command to execute the reranker

```
export GLUE_DIR=./glue 

export TASK_NAME=WNLI
  
python ./transformers/examples/text-classification/run_glue.py \
  --model_name_or_path ./transformers/$TASK_NAME/ \
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./transformers/WNLIoutput/ \
  --cache_dir ./transformers/cache \
  --overwrite_cache \
  --overwrite_output_dir > output.txt
```

#### Step 4 - Resort questions based on scores
<code>python resort_question.py</code> to resort the generated questions according to scores.


## CC Attribution 4.0 International

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

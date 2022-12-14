
## Language model training

Based on the script [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py).

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2, BERT, DistilBERT and RoBERTa. GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT, DistilBERT and RoBERTa
are fine-tuned using a masked language modeling (MLM) loss.

Before running the following example, you should get a file that contains text on which the language model will be
trained or fine-tuned. A good example of such text is the [WikiText-2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).

We will refer to two different files: `$TRAIN_FILE`, which contains text for training, and `$TEST_FILE`, which contains
text that will be used for evaluation.

### GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before
the tokenization). The loss here is that of causal language modeling.

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

CUDA_VISIBLE_DEVICES=4,5 python /scratch/cluster/wjko/ProfileREG/transformers/examples/language-modeling/run_language_modeling.py \
    --output_dir=/scratch/cluster/wjko/ProfileREG/transformers/gpt/output --overwrite_output_dir \
	--cache_dir=/scratch/cluster/wjko/ProfileREG/transformers/gpt/cache \
    --model_type=gpt2 \
	--per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=1\
    --model_name_or_path=gpt2-medium \
    --do_train --line_by_line\
    --train_data_file=/scratch/cluster/wjko/ProfileREG/transformers/gpt/train.txt \
    --do_eval \
	--num_train_epochs 5.0 \
	--save_steps 50000 --logging_steps 1000 \
    --eval_data_file=/scratch/cluster/wjko/ProfileREG/transformers/gpt/valtest.txt
```

This takes about half an hour to train on a single K80 GPU and about one minute for the evaluation to run. It reaches
a score of ~20 perplexity once fine-tuned on the dataset.

### RoBERTa/BERT/DistilBERT and masked language modeling

The following example fine-tunes RoBERTa on WikiText-2. Here too, we're using the raw WikiText-2. The loss is different
as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their
pre-training: masked language modeling.

In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore, converge
slightly slower (over-fitting takes more epochs).

We use the `--mlm` flag so that the script may change its loss function.

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
    --output_dir=output \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm
```

### XLNet and permutation language modeling

XLNet uses a different training objective, which is permutation language modeling. It is an autoregressive method 
to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input 
sequence factorization order.

We use the `--plm_probability` flag to define the ratio of length of a span of masked tokens to surrounding 
context length for permutation language modeling.

The `--max_span_length` flag may also be used to limit the length of a span of masked tokens used 
for permutation language modeling.

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
    --output_dir=output \
    --model_name_or_path=xlnet-base-cased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
```

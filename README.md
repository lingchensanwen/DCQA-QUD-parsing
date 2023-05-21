# DCQA-QUD-parsing
### This is a repo for DCQA QUD parsing implementation 

Title: [Discourse Analysis via Questions and Answers: Parsing Dependency Structures of Questions Under Discussion](https://arxiv.org/abs/2210.05905)

Authors: [Wei-Jen Ko](https://www.linkedin.com/in/wei-jen-ko-854597146), [Yating Wu](https://lingchensanwen.github.io), [Cutter Dalton](https://www.colorado.edu/linguistics/cutter-dalton), [Dananjay Srinivas](https://www.dsrinivas.com), [Greg Durrett](https://www.cs.utexas.edu/~gdurrett/), [Junyi Jessy Li](https://jessyli.com/)

```bibtex
@article{ko2022discourse,
  title={Discourse Analysis via Questions and Answers: Parsing Dependency Structures of Questions Under Discussion},
  author={Ko, Wei-Jen and Wu, Yating and Dalton, Cutter and Srinivas, Dananjay and Durrett, Greg and Li, Junyi Jessy},
  journal={arXiv preprint arXiv:2210.05905},
  year={2022}
}
```

## Introduction

This repo contains code and data source for paper in [Discourse Analysis via Questions and Answers: Parsing Dependency Structures of Questions Under Discussion](https://arxiv.org/abs/2210.05905). This work present DCQA QUD Parser, the first QUD (Questions Under Discussion) parser for discourse analysis. In this repo, we include code to predict anchor sentence, generate question based on predicted anchor sentence, re-ranker to sort questions based their score. 

## Table of Contents

0. [Prepare Requirements & Download Models](#prepare-requirements-and-download-models)
1. [Anchor Sentence Prediction](#anchor-sentence-prediction)
2. [Generate Question](#generate-question)
3. [Prepare re-ranking scores for each question](#prepare-re-ranking-scores-for-each-question)
4. [Resort questions based on scores](#resort-questions-based-on-scores)
5. [Related Work: DCQA Discourse Comprehension by Question Answering](#related-work)


## Prepare Requirements and Download Models

Install the version of transformers toolkit in ./transformers (go to the directory, and run "pip install -e .")

Download and unzip the models 

[discourse - used in anchor prediction](https://1drv.ms/u/s!As41x9akhTMMxVtb1DXUCbgJj-r6?e=j7Powc) 

[question_generation - used in question generation](https://1drv.ms/u/s!AhMIR5wciICxgT0WNL61CY16Z5m4?e=gRbfGO)

[WNLI - used in re-rankering](https://1drv.ms/u/s!As41x9akhTMMxWbRBUFJOGGVkVIr?e=opJPMh)

## Anchor Sentence Prediction
Put all testing articles in the directory <code>./inputa</code>

<code>python prepare_anchor_prediction.py</code>, this script generates the input format for the anchor prediction model.
The input file should have such a format:

```
sentence number + tab + sentence
```

for example:

```
1 The purple elephant played a harmonica in the middle of the park.
2 She wore a polka-dot hat and carried a suitcase full of rubber ducks.
```

Run the following command to execute the anchor prediction model.

```
python ./transformers/examples/question-answering/run_squad.py \
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

## Generate Question
<code>python prepare_question_generation.py</code>, this script performs NER masking and generates the input format of the GPT-2 question generation model

Run the following command to execute the question generation model. (file paths are at line 231, line 232)

<code>python ./transformers/examples/text-generation/run_generation.py     --model_type=gpt2     --model_name_or_path=./question_genertion</code>

## Prepare re-ranking scores for each question
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

## Resort questions based on scores
<code>python resort_question.py</code> to resort the generated questions according to scores.


## Related Work
[DCQA Discourse Comprehension by Question Answering](https://github.com/wjko2/DCQA-Discourse-Comprehension-by-Question-Answering)

```bibtex
@inproceedings{ko-etal-2022-discourse,
    title = "Discourse Comprehension: A Question Answering Framework to Represent Sentence Connections",
    author = "Ko, Wei-Jen  and
      Dalton, Cutter  and
      Simmons, Mark  and
      Fisher, Eliza  and
      Durrett, Greg  and
      Li, Junyi Jessy",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.806",
    pages = "11752--11764",
    abstract = "While there has been substantial progress in text comprehension through simple factoid question answering, more holistic comprehension of a discourse still presents a major challenge (Dunietz et al., 2020). Someone critically reflecting on a text as they read it will pose curiosity-driven, often open-ended questions, which reflect deep understanding of the content and require complex reasoning to answer (Ko et al., 2020; Westera et al., 2020). A key challenge in building and evaluating models for this type of discourse comprehension is the lack of annotated data, especially since collecting answers to such questions requires high cognitive load for annotators.This paper presents a novel paradigm that enables scalable data collection targeting the comprehension of news documents, viewing these questions through the lens of discourse. The resulting corpus, DCQA (Discourse Comprehension by Question Answering), captures both discourse and semantic links between sentences in the form of free-form, open-ended questions. On an evaluation set that we annotated on questions from Ko et al. (2020), we show that DCQA provides valuable supervision for answering open-ended questions. We additionally design pre-training methods utilizing existing question-answering resources, and use synthetic data to accommodate unanswerable questions.",
}
```

## CC Attribution 4.0 International

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

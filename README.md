# NLP_T3


## Overview
This project takes a walk in the direction of architecture-based approach to alleviating hallucinations. This approach gives an insight into the interactions of different sub-modules of a transformer model (attention heads & feedforward networks) and provides a study on the impact of contrastive learning in an under-parameterized setting. As a contribution towards this project, we introduce a new architecture paradigm, Terribly Tiny Transformers (T3) that focuses on making the model extremely tiny and highly packed with attention heads. We run experiments to answer the question: What if we allocate (nearly) all of the available processing resources to the attention heads in cases of resource crunch?. We miniaturize all benchmark models to establish a new FaithDial benchmark for resource-constrained settings. Further, we experimented with the loss function for training these tiny transformer models. The intuition behind searching for optimal/multiple losses for these models was to find an optimal training pipeline for these heavily under-parameterized. We experimented with additive contrastive learning for these tiny models to analyze if an attention-dense tiny model can achieve competitive results with regard to state-of-the-art miniaturized architecture designs. We also present extensive experimentation to land up on an optimal T3 design paradigm.
## Data
The dataset is hosted on [Huggingface's datasets](https://github.com/huggingface/datasets):

## Use with Huggingface
T3 has been made available by the authors at
huggingface under the tag ayushutkarsh/t3
Roberta-T3 has been made available by the authors at huggingface under the tag a5u7/roberta-t3

## Train Your Models
The code for all the models is available in [models](models/), which can be used to reproduce our results or to train your own models.

### Requirements
First, install Pytorch 1.7+ from the [official website](https://pytorch.org) and then, clone this repository and install the dependencies:

```bash
pip install -r requirements.txt
```


### Training
Here is how to train a model:

```bash
python models/dialog.py --model_name_or_path ayushutkarsh/t3 \ 
  --do_train \
  --output_dir /path/to/output_dir \
  --fp16 \
  --train_batch_size 16 \
  --num_train_epochs 10 \
  --warmup_ratio 0.04 \
  --max_seq_length 512
```

To run on multiple GPUs, set `CUDA_VISIBLE_DEVICES`. By default, training early stops and the best model is saved at `/path/to/output_dir/best_model`.


### Evaluation
To compute perplexity of a model on the validation data, simply run:

```bash
python models/dialog.py --model_name_or_path /path/to/model/best_model \
  --do_eval \
  --eval_batch_size 16
```

For the test data, `--do_eval` should be replaced with `--do_test`.
Note that evaluation should be run on a single GPU.

To compute other metrics (BLEU, ROUGE, F1, BERTScore, and Q^2), reported in the paper, we used the scripts, provided in [https://github.com/orhonovich/q-squared](https://github.com/orhonovich/q-squared).

### Generation
To generate a response, simply run:

```bash
python models/generate.py --model_name_or_path /path/to/model/best_model --do_sample --top_p 0.6
```
Arguments for generation are as follows:
- `--output` (optional): Path of the output directory to save the generated responses.
- `--dataset_path` (optional): Path to your own dataset.
- `--control_tokens` (optional): Control tokens, prepended to the sequence, for controlled generation.
- `--max_length` (default: 100): Maximum length of the generated sequence.

### Critic
We also use our collected data to frame the problem of identifying hallucination
as a binary classification task where the goal is to predict whether an utterance is faithful or not, given the source knowledge.


#### Training
```bash
python models/critic.py --model_name_or_path roberta-large --do_train --train_batch_size 16 \
    --learning_rate 1e-5 --weight_decay 0.1 --warmup_ratio 0.08 --pad_to_multiple_of 8 --fp16 \
    --output_dir /path/to/output
```

#### Testing
```bash
python models/critic.py --model_name_or_path /path/to/model --eval_batch_size 16 --do_test
```

There is also a run_script ipnyb to streamline running all this code.

#### NOTE 
Files except generate.py,dialog.py,critic.py, and model_nce.py are taken as is from [https://github.com/McGill-NLP/FaithDial]{https://github.com/McGill-NLP/FaithDial} as helper codes. 


Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: apache-2.0
base_model: microsoft/resnet-18
tags:
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: msi-resnet18-pretrain
  results:
  - task:
      name: Image Classification
      type: image-classification
    dataset:
      name: imagefolder
      type: imagefolder
      config: default
      split: validation
      args: default
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.908774373259053
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# msi-resnet18-pretrain

This model is a fine-tuned version of [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2966
- Accuracy: 0.9088

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 4

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.2719        | 1.0   | 1562 | 0.3501          | 0.8891   |
| 0.1611        | 2.0   | 3125 | 0.2813          | 0.9118   |
| 0.166         | 3.0   | 4687 | 0.3161          | 0.8930   |
| 0.1071        | 4.0   | 6248 | 0.2966          | 0.9088   |


### Framework versions

- Transformers 4.35.2
- Pytorch 2.0.1+cu118
- Datasets 2.15.0
- Tokenizers 0.15.0

Output:
{
    "extracted_code": ""
}

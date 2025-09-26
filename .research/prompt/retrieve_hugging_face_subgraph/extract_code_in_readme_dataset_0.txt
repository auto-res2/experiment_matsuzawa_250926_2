
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': n01532829
          '1': n01558993
          '2': n01704323
          '3': n01749939
          '4': n01770081
          '5': n01843383
          '6': n01855672
          '7': n01910747
          '8': n01930112
          '9': n01981276
          '10': n02074367
          '11': n02089867
          '12': n02091244
          '13': n02091831
          '14': n02099601
          '15': n02101006
          '16': n02105505
          '17': n02108089
          '18': n02108551
          '19': n02108915
          '20': n02110063
          '21': n02110341
          '22': n02111277
          '23': n02113712
          '24': n02114548
          '25': n02116738
          '26': n02120079
          '27': n02129165
          '28': n02138441
          '29': n02165456
          '30': n02174001
          '31': n02219486
          '32': n02443484
          '33': n02457408
          '34': n02606052
          '35': n02687172
          '36': n02747177
          '37': n02795169
          '38': n02823428
          '39': n02871525
          '40': n02950826
          '41': n02966193
          '42': n02971356
          '43': n02981792
          '44': n03017168
          '45': n03047690
          '46': n03062245
          '47': n03075370
          '48': n03127925
          '49': n03146219
          '50': n03207743
          '51': n03220513
          '52': n03272010
          '53': n03337140
          '54': n03347037
          '55': n03400231
          '56': n03417042
          '57': n03476684
          '58': n03527444
          '59': n03535780
          '60': n03544143
          '61': n03584254
          '62': n03676483
          '63': n03770439
          '64': n03773504
          '65': n03775546
          '66': n03838899
          '67': n03854065
          '68': n03888605
          '69': n03908618
          '70': n03924679
          '71': n03980874
          '72': n03998194
          '73': n04067472
          '74': n04146614
          '75': n04149813
          '76': n04243546
          '77': n04251144
          '78': n04258138
          '79': n04275548
          '80': n04296562
          '81': n04389033
          '82': n04418357
          '83': n04435653
          '84': n04443257
          '85': n04509417
          '86': n04515003
          '87': n04522168
          '88': n04596742
          '89': n04604644
          '90': n04612504
          '91': n06794110
          '92': n07584110
          '93': n07613480
          '94': n07697537
          '95': n07747607
          '96': n09246464
          '97': n09256479
          '98': n13054560
          '99': n13133613
  splits:
  - name: train
    num_bytes: 6284840508
    num_examples: 50000
  - name: validation
    num_bytes: 1286953696
    num_examples: 10000
  - name: test
    num_bytes: 670707560
    num_examples: 5000
  download_size: 7433461683
  dataset_size: 8242501764
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
license: other
license_name: imagenet
license_link: https://www.image-net.org/download.php
task_categories:
- image-classification
pretty_name: Mini-ImageNet
size_categories:
- 10K<n<100K
---

## Dataset Description

A mini version of ImageNet-1k with 100 of 1000 classes present.

Unlike some 'mini' variants this one includes the original images at their original sizes. Many such subsets downsample to 84x84 or other smaller resolutions.

### Data Splits

#### Train
* 50000 samples from ImageNet-1k train split

#### Validation
* 10000 samples from ImageNet-1k train split

#### Test
* 5000 samples from ImageNet-1k validation split (all 50 samples per class)

### Usage

This dataset is good for testing hparams and models in `timm`

#### Train

`python train.py --dataset hfds/timm/mini-imagenet --model resnet50 --amp --num-classes 100`

### Citation Information

For the specific instance of this mini variant I am not sure what the origin is. It is different from commonly referenced [Vinyales et al.,2016](https://arxiv.org/abs/1606.04080) as it doesn't match the classes / splits.

Train & validation splits match train & test of https://www.kaggle.com/datasets/ctrnngtrung/miniimagenet ... it is not clear where that originated though.

Original ImageNet citation:
```bibtex
@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
}
```
Output:
{
    "extracted_code": ""
}

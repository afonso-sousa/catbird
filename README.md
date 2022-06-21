<div align="center">
    </p>
    <img src="resources/catbird_banner.png"  width="400"/>
    </p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
  [![codecov](https://img.shields.io/codecov/c/gh/AfonsoSalgadoSousa/catbird)](https://codecov.io/gh/AfonsoSalgadoSousa/catbird)
</div>

`Catbird` is an open source paraphrase generation toolkit based on PyTorch.

## Main Features
This is an ongoing, one-person project. Hopefully you find it useful. If you do so, do not forget to leave a star &#127775;.

### Datasets
 * Quora Question Pairs
 * MSCOCO

### Metrics
We support the following metrics. We currently use the HuggingFace implementations and wrap them to use with Pytorch Ignite.
 * BLEU
 * METEOR
 * TER

### Seq2Seq Techniques
We support Teacher Forcing and for decoding both greedy and beam search.

## Quick Start

### Requirements and Installation
The project is based on PyTorch 1.11+ and Python 3.8+.

## Install Catbird
The package can be installed using pip:
```shell
pip install catbird
```
This does not include configuration files or tools and is not yet actively updated.
Alternatively, you can run from the source code:

**a. Clone the repository.**
```shell
git clone https://github.com/AfonsoSalgadoSousa/catbird.git
```
**b. Install dependencies.**

This project uses Poetry as its package manager. Make sure you have it installed. For more info check [Poetry's official documentation](https://python-poetry.org/docs/).
To install dependencies, simply run:
```shell
poetry install
```

We have also compiled an `enviroment.yml` file with all the required dependencies to create an Anaconda environment. To do so, simply run:
```shell
conda env create -f environment.yml
```


## Dataset Preparation
For now, we support [Quora Question Pairs dataset](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs), and [MSCOCO](https://cocodataset.org/#download). It is recommended to download and extract the datasets somewhere outside the project directory and symlink the dataset root to `$CATBIRD/data` as below. If your folder structure is different, you may need to change the corresponding paths in config files.

```text
catbird
├── catbird
├── tools
├── configs
├── data
│   ├── quora
│   │   ├── quora_duplicate_questions.tsv
│   ├── mscoco
│   │   ├── captions_train2014.json
│   │   ├── captions_val2014.json
```

Donwload Quora data [HERE](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs). Prepare Quora data by running:
```shell
poetry run python tools/preprocessing/create_data.py quora --root-path ./data/quora --out-dir ./data/quora
```

Download MSCOCO [HERE](https://cocodataset.org/#download), under the link '2014 Train/Val annotations'. Prepare MSCOCO data by running:
```shell
poetry run python tools/preprocessing/create_data.py mscoco --root-path ./data/mscoco --out-dir ./data/mscoco --split train
poetry run python tools/preprocessing/create_data.py mscoco --root-path ./data/mscoco --out-dir ./data/mscoco --split val
```

### Train

```shell
poetry run python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example:
1. Train T5 on QQP.
```bash
$ poetry run python tools/train.py configs/t5_quora.yaml
```

## Contributors
* [Afonso Sousa][1] (afonsousa2806@gmail.com)

[1]: https://github.com/AfonsoSalgadoSousa

# Acknowledgement
This project borrowed ideas from the following open-source repositories:
* [Det3D](https://github.com/poodarchu/Det3D)
* [fairseq](https://github.com/pytorch/fairseq)
* [mmcv](https://github.com/open-mmlab/mmcv)
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [Seq2Seq in PyTorch](https://github.com/eladhoffer/seq2seq.pytorch)
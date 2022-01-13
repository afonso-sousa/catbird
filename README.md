<div align="center">
    </p>
    <img src="resources/catbird_logo.svg" width="200"/>
    </p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
</div>

`Catbird` is an open source paraphrase generation toolkit based on PyTorch.

## Quick Start

### Requirements and Installation
The project is based on PyTorch 1.5+ and Python 3.6+.

## Install Catbird
The package can be installed using pip:
```shell
pip install catbird
```
This does not include configuration files or tools.
Alternatively, you can run from the source code:

**a. Clone the repository.**
```shell
git clone https://github.com/AfonsoSalgadoSousa/catbird.git
```
**b. Install dependencies.**
This project uses Poetry as its package manager. There should Make sure you have it installed. For more info check [Poetry's official documentation](https://python-poetry.org/docs/).
To install dependencies, simply run:
```shell
poetry install
```

## Dataset Preparation
For now, we support [Quora Question Pairs dataset](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs), [WikiAnswers](http://knowitall.cs.washington.edu/paralex/) and [MSCOCO](https://cocodataset.org/#download). It is recommended to download and extract the datasets somewhere outside the project directory and symlink the dataset root to `$CATBIRD/data` as below. If your folder structure is different, you may need to change the corresponding paths in config files.

```text
catbird
├── catbird
├── tools
├── configs
├── data
│   ├── quora
│   │   ├── quora_duplicate_questions.tsv
│   ├── wikianswers
│   │   ├── questions.txt
│   ├── mscoco
│   │   ├── captions_train2014.json
│   │   ├── captions_val2014.json
```

Donwload Quora data [HERE](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs). Prepare Quora data by running:
```shell
poetry run python tools/preprocessing/create_data.py quora --root-path ./data/quora --out-dir ./data/quora
```

Download WikiAnswers [HERE](http://knowitall.cs.washington.edu/paralex/). Prepare WikiAnswers data by running:
```shell
poetry run python tools/preprocessing/create_data.py wikianswers --root-path ./data/wikianswers --out-dir ./data/wikianswers
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
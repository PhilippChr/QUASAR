QUASAR
============
RAG-based Question Answering over Heterogeneous Data and Text
---------------

- [Description](#description)
- [Code](#code)
    - [System requirements](#system-requirements)
	- [Installation](#installation)
	- [Reproduce paper results](#reproduce-paper-results)
	- [Training the pipeline](#training-the-pipeline)
	- [Testing the pipeline](#testing-the-pipeline)
	- [Using the pipeline](#using-the-pipeline)
- [Feedback](#feedback)
- [License](#license)


# Description

If you use this code, please cite:
```bibtex
@article{christmann2025rag,
  title={RAG-based Question Answering over Heterogeneous Data and Text},
  author={Christmann, Philipp and Weikum, Gerhard},
  journal={IEEE Data Engineering Bulletin},
  year={2024}
}
```


# Code

## System requirements
All code was tested on Linux only.
- [Conda](https://docs.conda.io/projects/conda/en/latest/index.html)
- [PyTorch](https://pytorch.org)
- GPU (suggested)


## Installation
Clone the repo via:
We recommend the installation via conda, and provide the corresponding environment file in [conda-quasar.yml](conda-quasar.yml):

```bash
    git clone https://github.com/PhilippChr/QUASAR.git
    cd QUASAR/
    conda env create --file conda-quasar.yml
    conda activate quasar
    pip install -e .
```

Alternatively, you can also install the requirements via pip, using the [requirements.txt](requirements.txt) file (not tested). In this case, for running the code on a GPU, further packages might be required.


### Install dependencies
To initialize the repo (download data, benchmark, models), run:
```bash
    bash scripts/initialize.sh
```


## Training the pipeline

To train a pipeline, just choose the config that represents the pipeline you would like to train, and run:
``` bash
    bash scripts/pipeline.sh --train [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```

Example:
``` bash
    bash scripts/pipeline.sh --train config/compmix/quasar.yml kb_text_table_info
```


## Testing the pipeline

If you create your own pipeline, it is recommended to test it once on an example, to verify that everything runs smoothly.  
You can do that via:
``` bash
    bash scripts/pipeline.sh --example [<PATH_TO_CONFIG>]
```
and see the output file in `out/<benchmark>` for potential errors.

For standard evaluation, you can simply run:
``` bash
    bash scripts/pipeline.sh --run [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```
Example:
``` bash
    bash scripts/pipeline.sh --run config/compmix/quasar.yml kb_text_table_info
```
<br/>

For evaluating with all source combinations, run:
``` bash
    bash scripts/pipeline.sh --source-combinations [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```

Example:
``` bash
    bash scripts/pipeline.sh --source-combinations config/compmix/quasar.yml kb_text_table_info
```
<br/>

If you want to evaluate using the predicted answers of previous turns, you can run:
``` bash
    bash scripts/pipeline.sh --pred-answers [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```

Example:
``` bash
    bash scripts/pipeline.sh --pred-answers config/compmix/quasar.yml kb_text_table_info
```
By default, the QUASAR config and all sources will be used.


The results will be logged in the following directory: `out/<DATA>/<CMD>-<FUNCTION>-<CONFIG_NAME>.out`,  
and the metrics are written to `_results/<DATA>/<CONFIG_NAME>.res`.


## Using the pipeline
For using the pipeline, e.g. for improving individual parts of the pipeline, you can simply implement your own method that inherits from the respective part of the pipeline, create a corresponding config file, and add the module to the pipeline.py file. You can then use the commands outlined above to train and test the pipeline. 
Please see the documentation of the individual modules for further details:
- [Distant Supervision](quasar/distant_supervision/README.md)
- [Question Understanding (QU)](quasar/question_understanding/README.md)
- [Evidence Retrieval and Scoring (ERS)](quasar/evidence_retrieval_scoring/README.md)
- [Heterogeneous Answering (HA)](quasar/heterogeneous_answering/README.md)



# Feedback
We tried our best to document the code of this project, and make it accessible for easy usage. If you feel that some parts of the documentation/code could be improved, or have other feedback, please do not hesitate and let us know!

You can contact us via mail: pchristm@mpi-inf.mpg.de. Any feedback (also positive ;) ) is much appreciated!


# License
The QUASAR project by [Philipp Christmann](https://people.mpi-inf.mpg.de/~pchristm/) and [Gerhard Weikum](https://people.mpi-inf.mpg.de/~weikum/) is licensed under a [MIT license](LICENSE).

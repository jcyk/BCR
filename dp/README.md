Our code is adpated from [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2).

## Data

The data we used can be downloaded from [Google Drive](https://drive.google.com/file/d/1ju5RujDf0gFL1dl8lydv9FwySE6QbCff/view?usp=sharing)

## Training

First, go to the working directory

    cd NeuroNLP2/experiments
To train a Stack-Pointer parser, simply run

    bash scripts/run_stackptr.sh
To train a Deep BiAffine parser, simply run

    bash scripts/run_deepbiaf.sh
To get silver training data for knowldege distillation, see `run_global_kd.sh`

## Inference

To get the parses from exsiting parsers via various decoding strategies (including ensembling, beam search, $k$-best MST, dropout-$p$, etc), check `test_stackptr.sh` and `test_deepbiaf.sh`


Our code is adpated from [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2).

First, go to yhe working directory

    cd NeuroNLP2/experiments
To train a Stack-Pointer parser, simply run

    bash scripts/run_stackptr.sh
To train a Deep BiAffine parser, simply run

    bash scripts/run_deepbiaf.sh
To get silver training data for knowldege distillation, see `run_global_kd.sh`

To get the parses from exsiting parsers via decoding strategies, check `test_stackptr.sh` and `test_deepbiaf.sh`


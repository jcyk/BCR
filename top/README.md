Our code is adpated from [Fairseq](https://github.com/pytorch/fairseq).

First, go to the working directory

    cd NeuroNLP2/fairseq-01042021/examples/roberta_tops

To train a base seq2seq parser, simply run

    bash scripts/train_base.sh

To train a large seq2seq parser, simply run

    bash scripts/train_large.sh

To get silver training data for knowldege distillation

See `gen_data.sh`

To get the parses from exsiting parsers via decoding strategies

check `work.sh`


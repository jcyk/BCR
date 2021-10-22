Our code is adpated from [Fairseq](https://github.com/pytorch/fairseq).

## Data

The data we used can be downloaded from [Google Drive](https://drive.google.com/file/d/18F804io4MYpr-F3To8HkxA7q4E4Hk5Z9/view?usp=sharing). For data pre-processing, put the downloaded data in `data` folder, and run `prepare.sh`.

## Training

First, go to the working directory

    cd fairseq-01042021/examples/roberta_tops

To train a base seq2seq parser, simply run

    bash scripts/train_base.sh

To train a large seq2seq parser, simply run

    bash scripts/train_large.sh

To get silver training data for knowldege distillation, see `gen_data.sh`

## Inference

To get the parses from exsiting parsers via various decoding strategies (including ensembling, beam search, top-$k$, top-$p$, dropout-$p$, etc), check `work.sh`


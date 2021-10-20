#!/bin/bash

for SEED in 10 20 30
do
  DIR=/efs/exp/mansimov/checkpoints-efs/seq2seq/tops/roberta.base_seed-${SEED}/

  for GENSET in "valid" "test"
  do
    CUDA_VISIBLE_DEVICES=0 python /efs/exp/mansimov/code/fairseq-01042021/fairseq_cli/generate.py \
        /efs/exp/mansimov/code/top-dataset-only-01042021/data-bin/tops.seq2seq.roberta.pointer.fixed.bpe \
        --path $DIR/checkpoint_best.pt \
        --user-dir /efs/exp/mansimov/code/fairseq-01042021/examples/roberta_tops/ \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset ${GENSET} \
        --max-tokens 4096 --beam 1 > $DIR/${GENSET}.parse.unprocessed

    python process_gen_pg_file.py \
     --srcfile /efs/exp/mansimov/code/top-dataset-only-01042021/seq2seq-roberta-pointer-fixed/${GENSET}.en \
     --file $DIR/${GENSET}.parse.unprocessed \
     --outfile $DIR/${GENSET}.parse
  done
done
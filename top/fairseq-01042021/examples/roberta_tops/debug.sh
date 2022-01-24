#!/bin/bash
set -e
ROOT_DIR=/home/jcykcai/nonono/BCR/top

### dropout sampling###
dropout=0.3
sample_seed=123

fairseq-generate ${ROOT_DIR}/data/data-bin/top \
--user-dir ${ROOT_DIR}/fairseq-01042021/examples/roberta_tops/ \
--path ${ROOT_DIR}/base/checkpoint_best.pt \
--task translation_from_pretrained_roberta \
--left-pad-source False \
--gen-subset test \
--batch-size 128 --retain-dropout --beam 1 \
  --model-overrides "{'dropout': ${dropout}, 'attention-dropout': ${dropout}}" --seed ${sample_seed}> ${ROOT_DIR}/out/test.parse.dropout.unprocessed

python process_gen_pg_file.py \
--file ${ROOT_DIR}/out/test.parse.dropout.unprocessed \
--outfile ${ROOT_DIR}/out/test.parse.dropout.full.parse
  
 cp ${ROOT_DIR}/out/test.parse.dropout.full.parse.preprocessed.en ${ROOT_DIR}/data/data/test.parse.dropout.en
 cp ${ROOT_DIR}/out/test.parse.dropout.full.parse.preprocessed.parse ${ROOT_DIR}/data/data/test.parse.dropout.parse
 
 fairseq-preprocess --source-lang en --target-lang parse \
  --trainpref ${ROOT_DIR}/data/data/test.parse.dropout \
  --validpref ${ROOT_DIR}/data/data/test.parse.dropout \
  --testpref ${ROOT_DIR}/data/data/test.parse.dropout \
  --destdir ${ROOT_DIR}/data/data-bin/dropout \
  --srcdict ${ROOT_DIR}/data/data/dict.en.txt \
  --tgtdict ${ROOT_DIR}/data/data/dict.parse.txt  \
  --workers 20


fairseq-generate ${ROOT_DIR}/data/data-bin/dropout \
--user-dir ${ROOT_DIR}/fairseq-01042021/examples/roberta_tops/ \
--path ${ROOT_DIR}/base/checkpoint_best.pt \
--task translation_from_pretrained_roberta \
--left-pad-source False \
--gen-subset test \
--score-reference \
--batch-size 128 --beam 1 > ${ROOT_DIR}/out/test.parse.dropout.score


fairseq-generate ${ROOT_DIR}/data/data-bin/top \
--user-dir ${ROOT_DIR}/fairseq-01042021/examples/roberta_tops/ \
--path ${ROOT_DIR}/base/checkpoint_best.pt \
--task translation_from_pretrained_roberta \
--left-pad-source False \
--gen-subset test \
--batch-size 128 --beam 1 > ${ROOT_DIR}/out/test.parse.unprocessed

python process_gen_pg_file.py \
--file ${ROOT_DIR}/out/test.parse.unprocessed \
--outfile ${ROOT_DIR}/out/test.base.parse


fairseq-generate ${ROOT_DIR}/data/data-bin/top \
--user-dir ${ROOT_DIR}/fairseq-01042021/examples/roberta_tops/ \
--path ${ROOT_DIR}/base/checkpoint_best.pt \
--task translation_from_pretrained_roberta \
--left-pad-source False \
--gen-subset test \
--score-reference \
--batch-size 128 --beam 1 > ${ROOT_DIR}/out/test.parse.score

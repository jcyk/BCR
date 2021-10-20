#!/bin/bash

for traindata in base789 base555 base666 large123 large456 large789 large555 large666; do
seed=1
CKPT_DIR=~/checkpoints/${traindata}.large.seed${seed}
#for seed in 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50; do
#CKPT_DIR=~/checkpoints/large.seed${seed}
rm -rf ${CKPT_DIR}
mkdir -p ${CKPT_DIR}
fairseq-train \
  /efs/dengcai/top/data/data-bin/${traindata}.train \
  --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
  --arch transformer_pg_roberta_large_iwslt_deen --share-decoder-input-output-embed \
  --seed ${seed} \
  --task translation_from_pretrained_roberta \
  --encoder-learned-pos \
  --max-source-positions 512 \
  --max-positions 512 \
  --activation-fn gelu \
  --pretrained-roberta-checkpoint /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/pretrained/roberta.large/model.pt \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 500 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0001 --keep-last-epochs 5 \
  --alignment-layer -1 --alignment-heads 1 \
  --source-position-markers 50 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --valid-subset valid,test \
  --validate-interval	4 \
  --save-interval	4 \
  --batch-size 32 \
  --update-freq 4 \
  --eval-em \
  --left-pad-source False \
  --eval-em-args '{"beam": 1}' \
  --eval-em-detok space \
  --eval-em-remove-bpe \
  --eval-em-print-samples \
  --best-checkpoint-metric em --maximize-best-checkpoint-metric --save-dir $CKPT_DIR \
  --report-accuracy \
  --max-epoch 300 \
  --log-format=json --log-interval=10 2>&1 | tee $CKPT_DIR/train.log

cp -r ${CKPT_DIR} /efs/dengcai/top/checkpoints/large.${traindata}.seed${seed}
#cp -r ${CKPT_DIR} /efs/dengcai/top/checkpoints/large.seed${seed}
done

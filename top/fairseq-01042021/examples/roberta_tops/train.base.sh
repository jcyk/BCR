#for traindata in base123 base456 base789 base555 base666; do
#seed=1
#CKPT_DIR=~/checkpoints/${traindata}.base.seed${seed}
for seed in 1; do
ROOT_DIR=/home/jcykcai/nonono/BCR/top
CKPT_DIR=${ROOT_DIR}/base
mkdir -p $CKPT_DIR
CUDA_VISIBLE_DEVICES=3,4,5,6 fairseq-train \
  ${ROOT_DIR}/data/data-bin/top \
  --user-dir ${ROOT_DIR}/fairseq-01042021/examples/roberta_tops/ \
  --arch transformer_pg_roberta_iwslt_deen --share-decoder-input-output-embed \
  --seed ${seed} \
  --task translation_from_pretrained_roberta \
  --encoder-learned-pos \
  --max-source-positions 512 \
  --max-positions 512 \
  --activation-fn gelu \
  --pretrained-roberta-checkpoint ${ROOT_DIR}/data/roberta.base/model.pt \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 500 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0001 --keep-last-epochs 5 \
  --alignment-layer -1 --alignment-heads 1 \
  --source-position-markers 50 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --valid-subset valid,test \
  --validate-interval	4 \
  --save-interval	4 \
  --batch-size 128 \
  --eval-em \
  --left-pad-source False \
  --eval-em-args '{"beam": 1}' \
  --eval-em-detok space \
  --eval-em-remove-bpe \
  --eval-em-print-samples \
  --best-checkpoint-metric em --maximize-best-checkpoint-metric --save-dir $CKPT_DIR \
  --report-accuracy \
  --max-update 5000 \
  --log-format=json --log-interval=10 2>&1 | tee $CKPT_DIR/train.log

done

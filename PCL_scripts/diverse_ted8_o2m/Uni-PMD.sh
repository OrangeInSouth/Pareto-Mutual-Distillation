#!/usr/bin/env bash

python_path=/userhome/anaconda3/envs/mutual_distillation/bin/python
project_path="/userhome/ychuang/Mutual-Distillation"
lang_pairs="eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor"
path_2_data=/userhome/ychuang/online_distillation_MNMT/data-bin/ted_8_diverse
lang_list=${path_2_data}/lang_list_diverse.txt

SAVE_DIR=${project_path}/checkpoints/diverse_ted8_o2m/Uni-PMD
mkdir -vp $SAVE_DIR


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
${python_path} ${project_path}/fairseq_cli/train.py $path_2_data \
  --save-dir $SAVE_DIR \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer_iwslt_de_en --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1 \
  --sampling-temperature-2 5 \
  --mutual-distillation-mode "unidirectional" \
  --distillation-weight 0.4 \
  --weight-update-interval 3 \
  --decoder-langtok \
  --encoder-langtok "tgt" \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --criterion knowledge_distillation_criterion --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 0.002 \
  --share-decoder-input-output-embed \
  --max-epoch 300 --max-update 200000 \
  --dropout 0.3 --attention-dropout 0.3 --weight-decay 0.0 \
  --max-tokens 8192 --update-freq 1 \
  --no-epoch-checkpoints \
  --seed 222  \
  --log-format simple --log-interval 10  \
  --bpe sentencepiece \
  --fp16 \
  --pure-batch \
  --LS-epoch > ${SAVE_DIR}/train.log 2>&1
  # --arch transformer_iwslt_de_en
#  > ${SAVE_DIR}/train.log 2>&1

bash ${project_path}/PCL_scripts/diverse_ted8_o2m/test.sh ${SAVE_DIR}/model1
bash ${project_path}/PCL_scripts/diverse_ted8_o2m/test.sh ${SAVE_DIR}/model2

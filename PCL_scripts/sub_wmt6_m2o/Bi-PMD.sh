#!/usr/bin/env bash

python_path=/userhome/anaconda3/envs/mutual_distillation/bin/python
project_path="/userhome/ychuang/Mutual-Distillation"
lang_pairs="fr-en,de-en,zh-en,et-en,ro-en,tr-en"
path_2_data=/userhome/ychuang/datasets/data-bin/WMT6-small
lang_list=${path_2_data}/lang_list.txt

SAVE_DIR=${project_path}/checkpoints/sub_wmt6_m2o/Bi-PMD
mkdir -vp $SAVE_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
${python_path} ${project_path}/fairseq_cli/train.py $path_2_data \
  --save-dir $SAVE_DIR \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer_wmt_en_de --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1 \
  --sampling-temperature-2 5 \
  --mutual-distillation-mode "bidirectional" \
  --distillation-weight 0.6 \
  --decoder-langtok \
  --encoder-langtok "src" \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --criterion knowledge_distillation_criterion --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 0.0015 \
  --share-decoder-input-output-embed \
  --max-epoch 74 \
  --dropout 0.2 --weight-decay 0.01 \
  --max-tokens 8192 --update-freq 1 \
  --save-interval 1 \
  --seed 222 --log-format simple --log-interval 10 \
  --bpe sentencepiece \
  --fp16 \
  --pure-batch \
  --LS-epoch > ${SAVE_DIR}/train.log 2>&1
  # --arch transformer_iwslt_de_en

bash ${project_path}/PCL_scripts/sub_wmt6_m2o/test.sh ${SAVE_DIR}/model1
bash ${project_path}/PCL_scripts/sub_wmt6_m2o/test.sh ${SAVE_DIR}/model2

export PYTHONPATH=./

TRAIN_CODE=train.py

dataset=$1
exp_name=$2
subset=$3
wandb=$4
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_config.yaml

python script/train.py \
  --config=${config} \
  $subset \
  $wandb \
  save_path ${exp_dir} \
  2>&1 | tee ${exp_dir}/train-$now.log 
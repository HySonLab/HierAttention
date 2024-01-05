export PYTHONPATH=./
CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=2 python -m train.polymer --config configs/polymer/polymer_both.yaml $1
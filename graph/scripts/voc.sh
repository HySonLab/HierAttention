# export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=3 python -m train.voc --config configs/voc/voc.yaml $1
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=2 python -m train.voc --config configs/voc/voc.yaml $1

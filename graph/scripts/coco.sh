export PYTHONPATH=./
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 python -m train.coco --config configs/coco/coco.yaml $1

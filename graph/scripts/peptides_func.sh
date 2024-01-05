export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=3 python -m train.peptides_func --config configs/peptides_func/peptides_func.yaml $1
# CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python -m train.peptides_func --config configs/peptides_func/peptides_func.yaml $1

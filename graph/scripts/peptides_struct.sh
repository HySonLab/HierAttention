export PYTHONPATH=./
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 python -m train.peptides_struct --config configs/peptides_struct/peptides_struct.yaml $1

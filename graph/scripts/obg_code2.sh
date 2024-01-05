export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=3 python -m train.ogb_code2 --config configs/ogb/ogb_code2.yaml $1

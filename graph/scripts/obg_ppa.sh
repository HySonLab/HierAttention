export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=3 python -m train.ogb_ppa --config configs/ogb/ogb_ppa.yaml $1

export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=3 python -m train.ogb_mol --config configs/ogb/ogb_hiv.yaml $1

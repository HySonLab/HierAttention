# Graph-HierAttn

# Python environment setup with Conda

```
conda create --name graph_mlpmixer python=3.8
conda activate graph_mlpmixer

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
pip install ogb
conda install -c conda-forge rdkit
pip install yacs
pip install tensorboard
pip install networkx
pip install einops

# METIS
conda install -c conda-forge metis
pip install metis
```

## Run Graph HierAttn on different datasets

```
# Running Graph HierAttn on LRGB datasets (Add debug if you want the output print on the terminal rather than in the log file)
sh scripts/peptides_func.sh (--debug)
sh scripts/peptides_struct.sh (--debug)
sh scripts/voc.sh (--debug)
sh scripts/coco.sh (--debug)
```

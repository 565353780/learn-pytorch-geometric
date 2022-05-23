# learn-pytorch-geometric

## Install

```bash
conda create -n pyg python=3.8
conda activate pyg
pip install torch torchvision torchaudio \
      --extra-index-url https://download.pytorch.org/whl/cu113
pip install \
      torch-scatter torch-sparse torch-cluster torch-spline-conv \
      torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install tqdm
```

## Run

### Data class usage

```bash
python data.py
```

### Dataset class usage

```bash
python dataset.py
```

### DataLoader class usage

```bash
python dataloader.py
```

### Network training usage

```bash
python network.py
```

## Enjoy it~


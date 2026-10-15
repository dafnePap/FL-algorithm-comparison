# Federated Learning Algorithm Comparison

Implementation of federated learning algorithms for MNIST, EMNIST and CIFAR datasets.

**Algorithms:** `FedAvg` | `FedProx` | `SCAFFOLD` | `FedAdam` | `FDAOPT`

## Setup

Python 3.13.9 was used for development.

Install dependencies:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU support, install the CUDA-enabled PyTorch version separately.

## First Generate the Dataset

MNIST example:

```bash
python data/utils/run.py --dataset mnist --client_num_in_total 10 --s_iid 0.0 --seed 17 --root .
```

`--s_iid` controls the IID partition:

`0.0` = fully non-IID  
`1.0` = fully IID

## Run the Algorithm

FedAvg example:

```bash
python src/server/fedavg.py --dataset mnist --model LG --client_num_per_round 5 --global_epochs 50 --local_epochs 1 --batch_size 64 --local_lr 0.1 --target_acc 70 --seed 17
```

**Other server scripts:**

`src/server/fedprox.py`  
`src/server/scaffold.py`  
`src/server/fedadam_man.py`  
`src/server/fdaopt.py`

**Available models:**

`LG` → Logistic Regression  
`lenet` → LeNet-5  
`CNN` → Convolutional Neural Network  
`VGG16` → VGG-style CNN  
`RSN` → ResNet-18

## Results

Results are saved in:

`results/`

## Plot the Results

Example:

```bash
python plot.py --dataset mnist --model LG
```
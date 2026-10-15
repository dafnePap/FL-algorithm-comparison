MEAN = {
    "mnist": 0.1307,
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4865, 0.4409),
    "femnist": 0,  # dummy code, it's useless
    "synthetic": 0,  # dummy code, it's useless
    "emnist": 0.1736,
    "fmnist": 0.2860,
    "mrpc": None,  # NLP dataset, no pixel normalization needed
}

STD = {
    "mnist": 0.3015,
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2009, 0.1984, 0.2023),
    "femnist": 1.0,  # dummy code, it's useless
    "synthetic": 1.0,  # dummy code, it's useless
    "emnist": 0.3248,
    "fmnist": 0.3205,
    "mrpc": None,  # NLP dataset, no pixel normalization needed
}

# MRPC label count (binary: 0=not paraphrase, 1=paraphrase)
MRPC_NUM_LABELS = 2
ROBERTA_MODEL_NAME = "roberta-base"

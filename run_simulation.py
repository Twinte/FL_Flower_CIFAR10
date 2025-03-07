# run_simulation.py

import flwr as fl
import threading
import time
import math
from typing import Dict
from torchvision import datasets, transforms

from components import (
    create_non_iid_partition,
    ClientDataset,
    CIFARClient,
)
from server import start_server


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def compute_entropy_of_labels(client_indices, train_dataset) -> Dict[str, float]:
    """
    Return a dict {"0": 1.2, "1": 0.9, ...} for label-dist entropies.
    """
    partition_to_entropy = {}
    for cid, indices in enumerate(client_indices):
        label_count = {}
        for idx in indices:
            _, label = train_dataset[idx]
            label_count[label] = label_count.get(label, 0) + 1

        total_count = sum(label_count.values())
        entropy = 0.0
        for count in label_count.values():
            p = count / total_count if total_count else 0
            if p > 0:
                entropy -= p * math.log2(p)
        partition_to_entropy[str(cid)] = entropy
    return partition_to_entropy


def client_fn(partition_id: str, train_dataset, test_dataset, client_indices):
    """
    Create a CIFARClient with a known partition_id (like "0", "1", etc.).
    """
    cid_int = int(partition_id)
    ds = ClientDataset(train_dataset, client_indices[cid_int])
    return CIFARClient(ds, test_dataset, partition_id=partition_id)


def start_client(partition_id: str, train_dataset, test_dataset, client_indices, server_address: str):
    client = client_fn(partition_id, train_dataset, test_dataset, client_indices)
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),   # no client_id param in real Flower
    )


def main():
    NUM_CLIENTS = 30
    NUM_ROUNDS = 20
    SERVER_ADDRESS = "0.0.0.0:8080"

    # 1) Load data
    train_dataset, test_dataset = load_data()

    # 2) Partition data
    client_indices = create_non_iid_partition(train_dataset, NUM_CLIENTS, alpha=0.1)

    # 3) Compute partition entropies
    partition_to_entropy = compute_entropy_of_labels(client_indices, train_dataset)

    # 4) Start server
    server_thread = threading.Thread(
        target=start_server,
        args=(NUM_ROUNDS, partition_to_entropy, SERVER_ADDRESS),
    )
    server_thread.start()

    # 5) Wait a bit so server is ready
    time.sleep(3)

    # 6) Start clients
    client_threads = []
    for cid in range(NUM_CLIENTS):
        t = threading.Thread(
            target=start_client,
            args=(str(cid), train_dataset, test_dataset, client_indices, SERVER_ADDRESS),
        )
        t.start()
        client_threads.append(t)

    # 7) Wait for all clients
    for t in client_threads:
        t.join()

    # 8) Wait for the server
    server_thread.join()


if __name__ == "__main__":
    main()

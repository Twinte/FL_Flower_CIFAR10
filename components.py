# components.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import flwr as fl
import numpy as np
import math
from typing import Dict
from flwr.common import Scalar, bytes_to_ndarray, ndarray_to_bytes, NDArrays  # Correct imports
import sys

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def create_non_iid_partition(dataset, num_clients: int, alpha: float, seed: int = 42):
    np.random.seed(seed)
    labels = np.array([target for _, target in dataset])
    n_classes = len(np.unique(labels))

    client_data_indices = [[] for _ in range(num_clients)]
    from collections import defaultdict
    label_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)

    for label in range(n_classes):
        class_indices = label_indices[label]
        class_size = len(class_indices)
        dirichlet_dist = np.random.dirichlet(alpha * np.ones(num_clients))
        client_sample_sizes = (dirichlet_dist * class_size).astype(int)

        remaining = class_size - client_sample_sizes.sum()
        if remaining > 0:
            client_sample_sizes[0] += remaining

        start_idx = 0
        for client_id, sample_size in enumerate(client_sample_sizes):
            end_idx = start_idx + sample_size
            c_indices = class_indices[start_idx:end_idx]
            client_data_indices[client_id].extend(c_indices)
            start_idx = end_idx

    return client_data_indices


class ClientDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class CIFARClient(fl.client.NumPyClient):
    def __init__(self, train_dataset, test_dataset, partition_id: str):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.partition_id = partition_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

    def get_properties(
        self, config: Dict[str, Scalar]
    ) -> Dict[str, Scalar]:
        """Report custom properties (here, partition_id)."""
        props = {"partition_id": self.partition_id}
        prop_str = str(props)
        prop_bytes = prop_str.encode("utf-8")
        self.total_bytes_sent += sys.getsizeof(prop_bytes)
        return props

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        params: NDArrays = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        total_size = 0
        for param in params:
            serialized_param = ndarray_to_bytes(param)  # Correct serialization
            total_size += sys.getsizeof(serialized_param)
        self.total_bytes_sent += total_size
        return params

    def set_parameters(self, parameters: NDArrays):
        # Directly use the provided NDArrays
        total_size = sum(sys.getsizeof(param) for param in parameters)
        self.total_bytes_received += total_size

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config):
        self.set_parameters(parameters)
        train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()
        for _epoch in range(5):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
        return self.get_parameters({}), len(train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays, config):
        self.set_parameters(parameters)
        test_loader = DataLoader(self.test_dataset, batch_size=32)
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        return loss, len(test_loader.dataset), {"accuracy": accuracy}

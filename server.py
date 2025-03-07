# server.py

import flwr as fl
from typing import Dict, List, Tuple, Optional
from flwr.common import Metrics, parameters_to_ndarrays, ndarray_to_bytes, bytes_to_ndarray
import sys


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Weighted average of accuracy."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_average_loss(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Weighted average of loss."""
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"loss": sum(losses) / sum(examples)}

class EntropyStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        partition_to_entropy: Dict[str, float],
        fraction_fit: float = 0.5,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        self.partition_to_entropy = partition_to_entropy
        self.cid_to_partition: Dict[str, str] = {}
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

    def client_joined(self, client, **kwargs):
        props = client.get_properties({})
        part_id = props.get("partition_id", None)
        if part_id is not None:
            # String representation for consistent handling
            prop_str = str(props)
            prop_bytes = prop_str.encode('utf-8')  # Encode to bytes
            self.total_bytes_received += sys.getsizeof(prop_bytes)  # Count received bytes
            self.cid_to_partition[client.cid] = part_id

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, float]]:
        if not results:
            return None, {}

        total_size = 0
        accuracies = []
        losses = []

        for _, fit_res in results:
            for param in fit_res.parameters.tensors:
                total_size += sys.getsizeof(bytes_to_ndarray(param))  # Correct byte counting

            # Collect accuracy and loss metrics for aggregation
            metrics = fit_res.metrics
            if "accuracy" in metrics:
                accuracies.append(metrics["accuracy"])
            if "loss" in metrics:
                losses.append(metrics["loss"])

        self.total_bytes_received += total_size
        # Aggregate as usual
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            total_size = 0
            for param in aggregated_parameters.tensors:
                total_size += sys.getsizeof(bytes_to_ndarray(param))  # Count sent bytes
            self.total_bytes_sent += total_size
        
        # Return aggregated metrics (accuracy and loss) along with parameters
        return aggregated_parameters, {
            "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "loss": sum(losses) / len(losses) if losses else 0.0,
        }

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        if not results:
            return None, {}

        for _, evaluate_res in results:
            # String representation of metrics
            metrics_str = str(evaluate_res.metrics)
            metrics_bytes = metrics_str.encode('utf-8')  # Encode to bytes
            self.total_bytes_received += sys.getsizeof(metrics_bytes)  # Count received bytes

        # Aggregate evaluation metrics as usual
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Return aggregated metrics (accuracy and loss) as well
        return loss, metrics

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.NDArrays,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        # Serialize parameters *before* creating FitIns
        params = fl.common.parameters_to_ndarrays(parameters)
        total_size = 0
        for param in params:
            serialized_param = ndarray_to_bytes(param)
            total_size += sys.getsizeof(serialized_param)  # count sent bytes
        self.total_bytes_sent += total_size
        serialized_params = fl.common.ndarrays_to_parameters(params)

        connected_clients = list(client_manager.all().values())
        if not connected_clients:
            return []

        def get_entropy_for_client(c: fl.server.client_proxy.ClientProxy) -> float:
            cid = c.cid
            part_id = self.cid_to_partition.get(cid, None)
            return self.partition_to_entropy.get(part_id, 0.0) if part_id else 0.0

        connected_clients_sorted = sorted(
            connected_clients, key=get_entropy_for_client, reverse=True
        )
        total = len(connected_clients_sorted)
        sample_size = int(self.fraction_fit * total)
        sample_size = max(sample_size, self.min_fit_clients)
        sample_size = min(sample_size, total)
        selected_clients = connected_clients_sorted[:sample_size]

        fit_ins = fl.common.FitIns(parameters=serialized_params, config={})  # Use serialized parameters
        return [(client, fit_ins) for client in selected_clients]


def start_server(
    num_rounds: int,
    partition_to_entropy: Dict[str, float],
    server_address: str = "0.0.0.0:8080",
):
    strategy = EntropyStrategy(
        partition_to_entropy=partition_to_entropy,
        fraction_fit=0.5,
    )
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    print(f"Server: Total Bytes Sent: {strategy.total_bytes_sent}")
    print(f"Server: Total Bytes Received: {strategy.total_bytes_received}")

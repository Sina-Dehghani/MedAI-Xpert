import torch
import logging
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from typing import Any, Dict, List, Tuple

from src.models.base_model import BaseModel
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class GCN(torch.nn.Module):
    """A simple Graph Convolutional Network for node classification."""

    def __init__(self, num_node_features: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        logger.info("GCN architecture initialized.")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class DiseaseGraphModel(BaseModel):
    """
    Implements a Graph Neural Network for modeling disease relationships,
    e.g., for recommending related conditions or treatments.
    """

    def __init__(
        self, num_node_features: int, hidden_channels: int = 16, num_classes: int = 2
    ):
        super().__init__(model_name="DiseaseGraphModel")
        self._model = GCN(num_node_features, hidden_channels, num_classes)
        self._data_graph = None  # Store the graph data (nodes, edges)
        logger.info("DiseaseGraphModel initialized with GCN.")

    def _load_model_weights(self, model_path: str):
        """Loads PyTorch Geometric model state dict."""
        try:
            self._model.load_state_dict(torch.load(model_path))
            self._model.eval()
            logger.info(f"DiseaseGraphModel weights loaded from {model_path}.")

        except Exception as e:
            logger.error(f"Error loading DiseaseGraphModel weights: {e}")
            raise

    def _train_model(self, data: Data, config: Dict[str, Any]):
        """
        Trains the GNN model.
        `data` expects a torch_geometric.data.Data object (features, edge_index, labels).
        `config` expects {'epochs': int, 'learning_rate': float}.
        """
        self._model.train()
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=config.get("learning_rate", 0.01)
        )
        criterion = torch.nn.NLLLoss()  # For log_softmax output

        for epoch in range(config.get("epochs", 10)):
            optimizer.zero_grad()
            out = self._model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            logger.info(
                f"GNN Epoch {epoch+1}/{config['epochs']}, Loss: {loss.item():.4f}"
            )

        # Save model after training
        output_model_path = config.get(
            "output_model_path", "models/disease_graph_model.pth"
        )
        torch.save(self._model.state_dict(), output_model_path)
        logger.info(f"DiseaseGraphModel saved to {output_model_path}.")
        self._data_graph = (
            data  # Store the graph data for future predictions/evaluations
        )

    def _perform_prediction(self, node_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Performs prediction for specific nodes in the graph.
        `node_indices` are the indices of nodes for which to predict.
        """
        if self._model is None or self._data_graph is None:
            logger.error("GNN model or graph data not loaded. Cannot predict.")
            raise RuntimeError("GNN model or graph data not loaded.")

        self._model.eval()
        with torch.no_grad():
            out = self._model(self._data_graph.x, self._data_graph.edge_index)
            predictions = []
            for idx in node_indices:
                node_output = out[idx]
                probabilities = torch.exp(node_output).tolist()
                predicted_class = torch.argmax(node_output).item()
                predictions.append(
                    {
                        "node_index": idx,
                        "predicted_class": predicted_class,
                        "probabilities": probabilities,
                    }
                )
        logger.debug(f"GNN predictions for nodes {node_indices}: {predictions}")
        return predictions

    def _evaluate_model(self, data: Data) -> Dict[str, Any]:
        """
        Evaluates the GNN model on test data.
        `data` expects a torch_geometric.data.Data object with a test_mask.
        """
        if self._model is None:
            logger.error("GNN model not loaded. Cannot evaluate.")
            raise RuntimeError("GNN model not loaded.")

        self._model.eval()
        with torch.no_grad():
            out = self._model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            accuracy = int(correct) / int(data.test_mask.sum())
            logger.info(f"DiseaseGraphModel evaluation accuracy: {accuracy:.4f}")

        return {"accuracy": accuracy}


if __name__ == "__main__":
    setup_logging()

    # Create dummy graph data (e.g., 5 nodes, 2 features, 2 classes)
    # Nodes could represent diseases, features could be symptoms presence, edges could be co-occurrence
    num_nodes = 5
    num_features = 2
    num_classes = 2  # e.g., 'primary disease', 'comorbidity'

    x = torch.randn(num_nodes, num_features)  # Node features
    # Example edges: (0,1) -> disease A co-occurs with disease B
    edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]], dtype=torch.long)
    y = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)  # Node labels

    # Create dummy masks for training, validation, test
    train_mask = torch.tensor([True, True, False, False, False])
    val_mask = torch.tensor([False, False, True, False, False])
    test_mask = torch.tensor([False, False, False, True, True])

    graph_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # Initialize and train model
    gnn_model = DiseaseGraphModel(
        num_node_features=num_features, num_classes=num_classes
    )
    train_config = {
        "epochs": 5,
        "learning_rate": 0.01,
        "output_model_path": "models/test_disease_graph_model.pth",
    }
    gnn_model.train(graph_data, train_config)

    # Load the trained model (re-instantiate to test load functionality)
    loaded_gnn_model = DiseaseGraphModel(
        num_node_features=num_features, num_classes=num_classes
    )
    loaded_gnn_model.load("models/test_disease_graph_model.pth")
    # Manually set graph data if not loaded via a data pipeline alongside model weights
    loaded_gnn_model._data_graph = graph_data

    # Perform prediction for specific nodes
    prediction_result = loaded_gnn_model.predict(node_indices=[3, 4])
    print(f"Prediction Result: {prediction_result}")

    # Evaluate model
    eval_metrics = loaded_gnn_model.evaluate(
        graph_data
    )  # Use the same graph for simplicity
    print(f"Evaluation Metrics: {eval_metrics}")

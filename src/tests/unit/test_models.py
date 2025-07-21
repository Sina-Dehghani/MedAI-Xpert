import os
import torch
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from torch_geometric.data import Data
from src.models.nlp.text_classifier import TextClassifier
from src.models.computer_vision.image_detector import ImageDetector, SimpleCNN
from src.models.graph_neural_networks.disease_graph_model import DiseaseGraphModel, GCN


# --- ImageDetector Tests ---
@pytest.fixture
def image_detector_instance():
    # Use num_classes=2 as per the current model config
    detector = ImageDetector(num_classes=2)
    # Mock the actual _model to avoid heavy loading/training
    detector._model = MagicMock(spec=SimpleCNN)
    # Mock predict output
    detector._model.return_value = torch.tensor([[0.1, 0.9]])  # Example logit output
    return detector


def test_image_detector_predict(image_detector_instance):
    dummy_image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)

    # Ensure transform method is mocked or handles numpy array correctly
    with patch.object(
        image_detector_instance, "transform", return_value=torch.randn(1, 64, 64)
    ) as mock_transform:
        prediction = image_detector_instance.predict(dummy_image)
        mock_transform.assert_called_once()  # Ensure image is transformed
        assert "predicted_class" in prediction
        assert prediction["predicted_class"] == 1  # Based on dummy logit output
        assert "probabilities" in prediction
        assert (
            pytest.approx(prediction["probabilities"][1], 0.01) == 0.7109
        )  # Softmax of [0.1, 0.9]


def test_image_detector_train(image_detector_instance, tmp_path):
    dummy_data = [{"image": np.zeros((64, 64), dtype=np.uint8), "label": 0}]
    config = {
        "epochs": 1,
        "learning_rate": 0.01,
        "output_model_path": str(tmp_path / "test_model.pth"),
    }

    # Mock the actual training steps
    with patch.object(image_detector_instance._model, "train"), patch(
        "torch.optim.Adam"
    ) as MockAdam, patch("torch.nn.CrossEntropyLoss") as MockLoss, patch(
        "torch.save"
    ) as mock_save:

        image_detector_instance.train(dummy_data, config)
        image_detector_instance._model.train.assert_called_once()
        MockAdam.assert_called_once()
        MockLoss.assert_called_once()
        mock_save.assert_called_once()
        assert os.path.exists(
            config["output_model_path"]
        )  # Check if file would be created


def test_image_detector_evaluate(image_detector_instance):
    dummy_data = [
        {"image": np.zeros((64, 64), dtype=np.uint8), "label": 0},
        {"image": np.zeros((64, 64), dtype=np.uint8), "label": 1},
    ]
    # Mock the prediction logic within evaluate
    image_detector_instance._model.return_value = torch.tensor(
        [[0.9, 0.1]]
    )  # Predicts class 0
    metrics = image_detector_instance.evaluate(dummy_data)
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 0.5  # One correct (label 0) out of two


# --- TextClassifier Tests ---
@pytest.fixture
def text_classifier_instance():
    # Mock AutoTokenizer and AutoModelForSequenceClassification
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained"
    ) as mock_model, patch("transformers.pipeline") as mock_pipeline:

        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_pipeline.return_value = MagicMock()

        classifier = TextClassifier(num_labels=2)
        classifier._pipeline.return_value = [
            {"label": "LABEL_1", "score": 0.9}
        ]  # Mock pipeline output
        return classifier


def test_text_classifier_predict(text_classifier_instance):
    text = "This is a test sentence."
    prediction = text_classifier_instance.predict(text)
    text_classifier_instance._pipeline.assert_called_once_with(text)
    assert "label" in prediction
    assert prediction["label"] == "LABEL_1"


def test_text_classifier_evaluate(text_classifier_instance):
    dummy_data = [
        {"text": "Good example", "label": 0},
        {"text": "Bad example", "label": 1},
    ]
    # Mock pipeline to return different labels to test accuracy
    text_classifier_instance._pipeline.side_effect = [
        [{"label": "LABEL_0", "score": 0.9}],  # Correct for first item
        [
            {"label": "LABEL_0", "score": 0.8}
        ],  # Incorrect for second item (should be LABEL_1)
    ]
    metrics = text_classifier_instance.evaluate(dummy_data)
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 0.5


# --- DiseaseGraphModel Tests ---
@pytest.fixture
def disease_graph_model_instance():
    # Mock GCN model
    with patch("torch_geometric.nn.GCNConv") as mock_gcn_conv:
        model = DiseaseGraphModel(num_node_features=2, num_classes=2)
        model._model = MagicMock(spec=GCN)
        # Mock the forward pass to return dummy outputs
        model._model.return_value = torch.tensor(
            [[0.1, 0.9], [0.8, 0.2], [0.5, 0.5], [0.7, 0.3], [0.2, 0.8]]
        )
        # Set up dummy graph data for testing prediction/evaluation
        model._data_graph = Data(
            x=torch.randn(5, 2),  # 5 nodes, 2 features
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
            y=torch.tensor([1, 0, 1, 0, 1], dtype=torch.long),  # Labels
            train_mask=torch.tensor([True, True, True, False, False]),
            test_mask=torch.tensor([False, False, False, True, True]),
        )
        return model


def test_disease_graph_model_predict(disease_graph_model_instance):
    node_indices = [3, 4]
    predictions = disease_graph_model_instance.predict(node_indices)
    assert len(predictions) == 2
    assert predictions[0]["node_index"] == 3
    assert predictions[1]["predicted_class"] == 1  # Based on dummy output


def test_disease_graph_model_evaluate(disease_graph_model_instance):
    metrics = disease_graph_model_instance.evaluate(
        disease_graph_model_instance._data_graph
    )
    assert "accuracy" in metrics
    # Based on dummy output for test_mask (nodes 3, 4) and dummy labels:
    # Node 3: Predicted 0.7, 0.3 -> class 0. Actual label is 0. (Correct)
    # Node 4: Predicted 0.2, 0.8 -> class 1. Actual label is 1. (Correct)
    assert (
        metrics["accuracy"] == 1.0
    )  # Both test nodes are correctly predicted by dummy output

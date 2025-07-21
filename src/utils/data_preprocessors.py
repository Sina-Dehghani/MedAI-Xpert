import logging
import numpy as np
import pandas as pd
from PIL import Image
from typing import Union, Dict, Any, List

from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Provides utilities for preprocessing image data for AI models.
    """

    @staticmethod
    def load_and_resize_image(
        image_path: str, size: tuple = (224, 224), convert_to_grayscale: bool = True
    ) -> np.ndarray:
        """Loads an image from path, resizes it, and converts to grayscale if specified."""
        try:
            img = Image.open(image_path)
            if convert_to_grayscale:
                img = img.convert("L")  # Convert to grayscale
            img = img.resize(size)
            img_array = np.array(img)
            logger.debug(
                f"Image loaded and preprocessed from {image_path}. Shape: {img_array.shape}"
            )
            return img_array
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise

    @staticmethod
    def normalize_image_pixels(
        image_array: np.ndarray, min_val: float = 0.0, max_val: float = 1.0
    ) -> np.ndarray:
        """Normalizes pixel values of an image array to a specified range."""
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32)
        normalized_array = min_val + (max_val - min_val) * (
            image_array - image_array.min()
        ) / (image_array.max() - image_array.min() + 1e-8)
        logger.debug(f"Image pixels normalized to range [{min_val}, {max_val}].")
        return normalized_array


class EHRPreprocessor:
    """
    Provides utilities for preprocessing EHR structured and unstructured data.
    """

    @staticmethod
    def clean_structured_data(df: pd.DataFrame) -> pd.DataFrame:
        """Performs basic cleaning on structured EHR DataFrame."""
        cleaned_df = df.copy()
        # Example: fill missing numeric values with median, categorical with mode
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ["int64", "float64"]:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif cleaned_df[col].dtype == "object":
                cleaned_df[col] = cleaned_df[col].fillna(
                    cleaned_df[col].mode()[0]
                    if not cleaned_df[col].mode().empty
                    else ""
                )
        logger.debug("Structured EHR data cleaned.")
        return cleaned_df

    @staticmethod
    def preprocess_clinical_notes(text: str) -> str:
        """Performs basic text preprocessing for clinical notes."""
        if not isinstance(text, str):
            return ""
        text = text.lower()  # Convert to lowercase
        text = " ".join(
            word for word in text.split() if word.isalpha()
        )  # Remove non-alphabetic tokens
        logger.debug(f"Clinical note preprocessed: {text[:50]}...")
        return text


class GraphDataPreprocessor:
    """
    Prepares raw data into a graph format suitable for GNNs (torch_geometric.data.Data).
    """

    @staticmethod
    def build_graph_from_ehr(
        ehr_df: pd.DataFrame,
        node_features_cols: List[str],
        edge_source_col: str,
        edge_target_col: str,
        node_id_col: str = "patient_id",
    ) -> Dict[str, Any]:
        """
        Builds a graph (nodes, edges, features, labels) from an EHR DataFrame.
        This is a highly conceptual example. A real graph might be built from
        disease co-occurrences, patient similarity, etc.

        Returns a dictionary that can be converted to torch_geometric.data.Data.
        """
        logger.info("Building conceptual graph from EHR data.")

        # For simplicity, let's assume each row is a node and we create
        # dummy edges based on some criteria, or use existing relationships.
        # Here, we'll create a dummy graph with nodes and features.

        # Map original patient IDs to sequential node indices
        unique_node_ids = ehr_df[node_id_col].unique()
        node_to_idx = {node_id: i for i, node_id in enumerate(unique_node_ids)}
        idx_to_node = {i: node_id for node_id, i in node_to_idx.items()}

        # Node features (e.g., patient demographics, aggregated lab results)
        # Ensure features are numeric and cleaned
        node_features_df = ehr_df.set_index(node_id_col)[node_features_cols]
        node_features_df = EHRPreprocessor.clean_structured_data(node_features_df)
        x = torch.tensor(node_features_df.values, dtype=torch.float)

        # Dummy edges: connect patients with similar age, or if they have same diagnosis (conceptual)
        edges = []
        # This logic needs to be replaced with actual graph building logic based on your data relationships.
        # Example: if patients share a common diagnosis, create an edge.

        # For demonstration, let's create random edges
        num_nodes = len(unique_node_ids)
        edge_index = torch.randint(
            0, num_nodes, (2, 10), dtype=torch.long
        )  # 10 random edges

        # Dummy labels for nodes (e.g., disease risk group)
        y = torch.randint(
            0, 2, (num_nodes,), dtype=torch.long
        )  # Binary classification label

        # Dummy masks (for training/testing in GNNs)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[: int(num_nodes * 0.7)] = True  # 70% for training
        test_mask = ~train_mask  # 30% for testing

        graph_data = {
            "x": x,
            "edge_index": edge_index,
            "y": y,
            "train_mask": train_mask,
            "test_mask": test_mask,
            "node_to_idx": node_to_idx,
            "idx_to_node": idx_to_node,
        }
        logger.info(
            f"Graph created with {num_nodes} nodes and {edge_index.shape[1]} edges."
        )
        return graph_data


if __name__ == "__main__":
    # Setup logging for example runs
    setup_logging()

    # --- Image Preprocessor Example ---
    print("\n--- Image Preprocessor Example ---")
    dummy_img_path = "dummy_image.png"
    Image.fromarray(np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)).save(
        dummy_img_path
    )

    try:
        processed_img = ImagePreprocessor.load_and_resize_image(
            dummy_img_path, size=(64, 64)
        )
        normalized_img = ImagePreprocessor.normalize_image_pixels(processed_img)
        print(
            f"Processed image shape: {processed_img.shape}, Normalized range: {normalized_img.min()}-{normalized_img.max()}"
        )
    except Exception as e:
        print(f"Error during image preprocessing example: {e}")
    finally:
        os.remove(dummy_img_path)

    # --- EHR Preprocessor Example ---
    print("\n--- EHR Preprocessor Example ---")
    dummy_ehr_df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [50, 60, np.nan],
            "gender": ["M", "F", "M"],
            "clinical_notes": [
                "patient has mild fever and cough",
                "severe respiratory issues",
                "controlled BP ",
            ],
        }
    )
    cleaned_ehr_df = EHRPreprocessor.clean_structured_data(dummy_ehr_df)
    preprocessed_note = EHRPreprocessor.preprocess_clinical_notes(
        dummy_ehr_df["clinical_notes"].iloc[2]
    )
    print(f"Cleaned EHR DataFrame:\n{cleaned_ehr_df}")
    print(f"Preprocessed note: '{preprocessed_note}'")

    # --- Graph Data Preprocessor Example ---
    print("\n--- Graph Data Preprocessor Example ---")
    import torch  # Ensure torch is imported for this section

    dummy_graph_ehr_df = pd.DataFrame(
        {
            "patient_id": [101, 102, 103, 104, 105],
            "age": [55, 62, 48, 70, 35],
            "bmi": [25.1, 30.5, 22.3, 28.0, 20.9],
            "diagnosis": [
                "Type2 Diabetes",
                "Hypertension",
                "Type2 Diabetes",
                "Coronary Artery Disease",
                "None",
            ],
        }
    )
    graph_data_dict = GraphDataPreprocessor.build_graph_from_ehr(
        dummy_graph_ehr_df,
        node_features_cols=["age", "bmi"],
        edge_source_col="patient_id",  # These are conceptual; actual edge logic needed
        edge_target_col="patient_id",
    )
    print(f"Graph data features shape: {graph_data_dict['x'].shape}")
    print(f"Graph data edge_index shape: {graph_data_dict['edge_index'].shape}")
    print(f"Graph data labels shape: {graph_data_dict['y'].shape}")

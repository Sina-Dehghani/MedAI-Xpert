import os
import pytest
import pydicom
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from src.data_ingestion.ehr_etl import (
    EHRExtractor,
    EHRTransformer,
    EHRDataLoader,
    EHRETLPipeline,
)
from src.data_ingestion.image_etl import (
    DICOMExtractor,
    DICOMTransformer,
    ImageDataLoader,
    ImageETLPipeline,
)


# --- Fixtures for common test data ---
@pytest.fixture
def dummy_dicom_file(tmp_path):
    """Creates a dummy DICOM file for testing."""
    filename = tmp_path / "test.dcm"
    file_meta = pydicom.dataset.Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"  # Explicit VR Little Endian
    ds = pydicom.FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Test^Patient"
    ds.PatientID = "UNIT_TEST_001"
    ds.Modality = "CT"
    ds.Rows = 10
    ds.Columns = 10
    ds.PixelData = np.random.randint(0, 255, size=(10, 10), dtype=np.uint16).tobytes()
    ds.save_as(str(filename))
    return str(filename)


@pytest.fixture
def dummy_ehr_csv(tmp_path):
    """Creates a dummy EHR CSV file."""
    filename = tmp_path / "ehr_data.csv"
    df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [50, 60, 45],
            "gender": ["M", "F", "M"],
            "diagnosis": ["Flu", "COVID-19", "Hypertension"],
            "clinical_notes": ["mild fever", "severe symptoms", "controlled BP"],
        }
    )
    df.to_csv(str(filename), index=False)
    return str(filename)


# --- DICOM ETL Tests ---
def test_dicom_extractor(dummy_dicom_file):
    extractor = DICOMExtractor()
    extracted_data = extractor.extract({"paths": [dummy_dicom_file]})
    assert len(extracted_data) == 1
    assert isinstance(extracted_data[0], pydicom.FileDataset)
    assert extracted_data[0].PatientID == "UNIT_TEST_001"


def test_dicom_transformer():
    ds = MagicMock(spec=pydicom.FileDataset)
    ds.pixel_array = np.zeros((10, 10), dtype=np.uint16)
    ds.PatientID = "TRANSFORM_001"
    ds.Modality = "MR"
    ds.Rows = 10
    ds.Columns = 10
    ds.PixelSpacing = [1.0, 1.0]
    ds.BitsStored = 16

    transformer = DICOMTransformer()
    transformed_data = transformer.transform([ds])
    assert "images" in transformed_data
    assert "metadata" in transformed_data
    assert len(transformed_data["images"]) == 1
    assert isinstance(transformed_data["metadata"], pd.DataFrame)
    assert transformed_data["metadata"].iloc[0]["PatientID"] == "TRANSFORM_001"


def test_image_data_loader(tmp_path):
    loader = ImageDataLoader()
    dummy_images = [np.random.rand(10, 10)]
    dummy_metadata = pd.DataFrame([{"PatientID": "LOAD_001", "Modality": "CT"}])
    destination_config = {"image_dir": str(tmp_path / "loaded_images")}

    loader.load(
        {"images": dummy_images, "metadata": dummy_metadata}, destination_config
    )

    assert os.path.exists(destination_config["image_dir"])
    assert os.path.exists(os.path.join(destination_config["image_dir"], "metadata.csv"))


def test_image_etl_pipeline(dummy_dicom_file, tmp_path):
    extractor = DICOMExtractor()
    transformer = DICOMTransformer()
    loader = ImageDataLoader()
    pipeline = ImageETLPipeline(extractor, transformer, loader)

    source_config = {"paths": [dummy_dicom_file]}
    destination_config = {"image_dir": str(tmp_path / "etl_output_images")}

    pipeline.run_pipeline(source_config, destination_config)
    assert os.path.exists(destination_config["image_dir"])
    assert os.path.exists(os.path.join(destination_config["image_dir"], "metadata.csv"))


# --- EHR ETL Tests ---
def test_ehr_extractor(dummy_ehr_csv):
    extractor = EHRExtractor()
    extracted_df = extractor.extract({"path": dummy_ehr_csv, "format": "csv"})
    assert not extracted_df.empty
    assert "patient_id" in extracted_df.columns
    assert len(extracted_df) == 3


def test_ehr_transformer():
    raw_df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4],
            "age": [50, None, 45, 70],
            "gender": ["M", "F", "M", "F"],
            "diagnosis": ["Flu", "COVID-19", "Hypertension", "Diabetes"],
            "clinical_notes": [
                " mild fever ",
                "severe symptoms ",
                "controlled BP",
                "high sugar",
            ],
        }
    )
    transformer = EHRTransformer()
    transformed_df = transformer.transform(raw_df)
    assert (
        transformed_df.isnull().sum().sum() == 0
    )  # Check for no NaNs after ffill/bfill
    assert transformed_df["clinical_notes"].iloc[0] == "mild fever"  # Check stripping


def test_ehr_data_loader(tmp_path):
    loader = EHRDataLoader()
    transformed_df = pd.DataFrame(
        [{"patient_id": 1, "age": 50, "notes": "cleaned notes"}]
    )
    output_csv_path = str(tmp_path / "ehr_loaded.csv")
    destination_config = {"db_connection": None, "table_name": "ehr_records"}

    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        # We specifically target the CSV saving in the demo DataLoader
        loader.load(transformed_df, destination_config)
        mock_to_csv.assert_called_once_with(
            os.path.join("data/processed/", "ehr_processed.csv"), index=False
        )
        # The actual path will be hardcoded in the DataLoader for conceptual demo


def test_ehr_etl_pipeline(dummy_ehr_csv, tmp_path):
    extractor = EHRExtractor()
    transformer = EHRTransformer()
    loader = EHRDataLoader()
    pipeline = EHRETLPipeline(extractor, transformer, loader)

    source_config = {"path": dummy_ehr_csv, "format": "csv"}
    destination_config = {
        "db_connection": None,
        "table_name": "ehr_records_test",
    }  # No actual DB connection for demo

    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        pipeline.run_pipeline(source_config, destination_config)
        mock_to_csv.assert_called_once()

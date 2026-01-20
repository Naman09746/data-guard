"""Unit tests for Data Versioning and Scan History."""

import numpy as np
import pandas as pd
import pytest

from src.core.data_versioning import (
    compute_dataset_hash,
    compute_schema_hash,
    DatasetVersion,
    DataVersioner,
    RegressionSeverity,
    ScanDiff,
    ScanHistoryStore,
    ScanRecord,
    ScanType,
    version_dataset,
)


class TestComputeDatasetHash:
    """Tests for hash computation functions."""

    def test_compute_dataset_hash_deterministic(self):
        """Test that hash is deterministic."""
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        hash1 = compute_dataset_hash(data)
        hash2 = compute_dataset_hash(data)
        
        assert hash1 == hash2

    def test_compute_dataset_hash_different_data(self):
        """Test that different data produces different hash."""
        data1 = pd.DataFrame({"a": [1, 2, 3]})
        data2 = pd.DataFrame({"a": [1, 2, 4]})
        
        hash1 = compute_dataset_hash(data1)
        hash2 = compute_dataset_hash(data2)
        
        assert hash1 != hash2

    def test_compute_schema_hash(self):
        """Test schema hash computation."""
        data = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        
        schema_hash = compute_schema_hash(data)
        
        assert len(schema_hash) == 16
        assert isinstance(schema_hash, str)

    def test_compute_schema_hash_same_schema(self):
        """Test that same schema produces same hash."""
        data1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data2 = pd.DataFrame({"a": [10, 20, 30], "b": [40, 50, 60]})
        
        assert compute_schema_hash(data1) == compute_schema_hash(data2)


class TestDatasetVersion:
    """Tests for DatasetVersion dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        version = DatasetVersion(
            version_hash="abc123",
            schema_hash="def456",
            row_count=100,
            column_count=5,
            column_names=["a", "b"],
            column_dtypes={"a": "int64", "b": "object"},
        )
        
        result = version.to_dict()
        
        assert result["version_hash"] == "abc123"
        assert result["row_count"] == 100
        assert "created_at" in result

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "version_hash": "abc123",
            "schema_hash": "def456",
            "row_count": 100,
            "column_count": 5,
            "column_names": ["a", "b"],
            "column_dtypes": {"a": "int64"},
            "created_at": "2024-01-01T00:00:00+00:00",
            "metadata": {},
        }
        
        version = DatasetVersion.from_dict(data)
        
        assert version.version_hash == "abc123"
        assert version.row_count == 100


class TestDataVersioner:
    """Tests for DataVersioner class."""

    @pytest.fixture
    def versioner(self):
        """Create versioner instance."""
        return DataVersioner()

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            "id": range(100),
            "value": np.random.randn(100),
            "category": ["A", "B"] * 50,
        })

    def test_create_version(self, versioner, sample_data):
        """Test version creation."""
        version = versioner.create_version(sample_data)
        
        assert len(version.version_hash) == 16
        assert version.row_count == 100
        assert version.column_count == 3
        assert "id" in version.column_names

    def test_create_version_with_metadata(self, versioner, sample_data):
        """Test version creation with metadata."""
        version = versioner.create_version(
            sample_data,
            metadata={"source": "test"},
        )
        
        assert version.metadata["source"] == "test"


class TestScanRecord:
    """Tests for ScanRecord dataclass."""

    @pytest.fixture
    def sample_record(self):
        """Create sample scan record."""
        return ScanRecord(
            scan_id="scan123",
            scan_type=ScanType.QUALITY,
            dataset_version=DatasetVersion(
                version_hash="abc",
                schema_hash="def",
                row_count=100,
                column_count=5,
                column_names=["a"],
                column_dtypes={"a": "int64"},
            ),
            status="passed",
            total_issues=2,
            quality_score=0.95,
        )

    def test_to_dict(self, sample_record):
        """Test conversion to dictionary."""
        result = sample_record.to_dict()
        
        assert result["scan_id"] == "scan123"
        assert result["scan_type"] == "quality"
        assert result["total_issues"] == 2

    def test_from_dict(self, sample_record):
        """Test creation from dictionary."""
        data = sample_record.to_dict()
        
        record = ScanRecord.from_dict(data)
        
        assert record.scan_id == "scan123"
        assert record.scan_type == ScanType.QUALITY


class TestScanHistoryStore:
    """Tests for ScanHistoryStore."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create store with temp path."""
        return ScanHistoryStore(tmp_path)

    @pytest.fixture
    def sample_scan(self):
        """Create sample scan record."""
        return ScanRecord(
            scan_id="scan001",
            scan_type=ScanType.QUALITY,
            dataset_version=DatasetVersion(
                version_hash="abc",
                schema_hash="def",
                row_count=100,
                column_count=5,
                column_names=["a"],
                column_dtypes={"a": "int64"},
            ),
            status="passed",
            total_issues=2,
            quality_score=0.9,
        )

    def test_add_scan(self, store, sample_scan):
        """Test adding a scan."""
        store.add_scan(sample_scan)
        
        scans = store.get_scans()
        
        assert len(scans) == 1
        assert scans[0].scan_id == "scan001"

    def test_get_scans_with_filter(self, store, sample_scan):
        """Test filtering scans."""
        store.add_scan(sample_scan)
        
        scans = store.get_scans(scan_type=ScanType.QUALITY)
        assert len(scans) == 1
        
        scans = store.get_scans(scan_type=ScanType.LEAKAGE)
        assert len(scans) == 0

    def test_get_latest_scan(self, store, sample_scan):
        """Test getting latest scan."""
        store.add_scan(sample_scan)
        
        latest = store.get_latest_scan()
        
        assert latest is not None
        assert latest.scan_id == "scan001"

    def test_compare_scans_no_regression(self, store):
        """Test scan comparison without regression."""
        version = DatasetVersion(
            version_hash="abc",
            schema_hash="def",
            row_count=100,
            column_count=5,
            column_names=["a"],
            column_dtypes={"a": "int64"},
        )
        
        scan1 = ScanRecord(
            scan_id="scan1",
            scan_type=ScanType.QUALITY,
            dataset_version=version,
            status="passed",
            total_issues=5,
            quality_score=0.8,
        )
        
        scan2 = ScanRecord(
            scan_id="scan2",
            scan_type=ScanType.QUALITY,
            dataset_version=version,
            status="passed",
            total_issues=3,  # Fewer issues
            quality_score=0.9,  # Better score
        )
        
        diff = store.compare_scans(scan1, scan2)
        
        assert not diff.has_regression
        assert diff.issues_change == -2
        assert diff.quality_score_change == pytest.approx(0.1, rel=1e-5)

    def test_compare_scans_with_regression(self, store):
        """Test scan comparison with regression."""
        version = DatasetVersion(
            version_hash="abc",
            schema_hash="def",
            row_count=100,
            column_count=5,
            column_names=["a"],
            column_dtypes={"a": "int64"},
        )
        
        scan1 = ScanRecord(
            scan_id="scan1",
            scan_type=ScanType.QUALITY,
            dataset_version=version,
            status="passed",
            total_issues=2,
            quality_score=0.95,
        )
        
        scan2 = ScanRecord(
            scan_id="scan2",
            scan_type=ScanType.QUALITY,
            dataset_version=version,
            status="passed",
            total_issues=15,  # Many more issues
            quality_score=0.5,  # Much worse score
        )
        
        diff = store.compare_scans(scan1, scan2)
        
        assert diff.has_regression
        assert diff.regression_severity == RegressionSeverity.CRITICAL
        assert len(diff.regression_details) > 0

    def test_check_for_regression(self, store, sample_scan):
        """Test regression detection."""
        store.add_scan(sample_scan)
        
        # Create a worse scan
        worse_version = DatasetVersion(
            version_hash="xyz",
            schema_hash="def",
            row_count=100,
            column_count=5,
            column_names=["a"],
            column_dtypes={"a": "int64"},
        )
        
        worse_scan = ScanRecord(
            scan_id="scan002",
            scan_type=ScanType.QUALITY,
            dataset_version=worse_version,
            status="passed",
            total_issues=20,  # Many issues
            quality_score=0.4,  # Poor score
        )
        
        alert = store.check_for_regression(worse_scan)
        
        assert alert is not None
        assert alert.severity in [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL]


class TestVersionDataset:
    """Tests for convenience function."""

    def test_version_dataset(self):
        """Test quick versioning function."""
        data = pd.DataFrame({"a": [1, 2, 3]})
        
        version = version_dataset(data)
        
        assert isinstance(version, DatasetVersion)
        assert version.row_count == 3

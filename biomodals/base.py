"""
Base class and utilities for biomodal functions.

This module provides common functionality for all biomodal functions including:
- GCS upload handling
- Metadata extraction
- Error handling
- Output file structuring
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BiomodalResult(BaseModel):
    """Standardized result format for biomodal functions."""

    success: bool = True
    output_files: list[dict[str, Any]] = Field(default_factory=list)
    message: str = "Function completed successfully"
    stdout: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Model-specific fields (optional)
    num_sequences: int | None = None
    num_models: int | None = None
    num_designs: int | None = None
    num_output_files: int | None = None
    num_recycles: int | None = None


class OutputFile(BaseModel):
    """Standardized output file structure."""

    filename: str
    artifact_type: str  # pdb, fasta, json, cif, etc.
    path: str  # Local path
    gcs_url: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BiomodalBase:
    """
    Base class for biomodal functions with common patterns.

    Provides utilities for:
    - GCS upload with standardized structure
    - Metadata extraction from output files
    - Error handling with structured responses
    - Output file formatting
    """

    def __init__(self, run_id: str, model_name: str):
        """
        Initialize biomodal base.

        Args:
            run_id: Unique identifier for this run
            model_name: Name of the model (e.g., "alphafold", "esmfold")
        """
        self.run_id = run_id
        self.model_name = model_name
        self.logger = logging.getLogger(f"biomodal.{model_name}")

    def upload_to_gcs(
        self,
        files: list[Path],
        gcs_bucket: str,
        artifact_types: dict[str, str] | None = None,
    ) -> list[OutputFile]:
        """
        Upload files to GCS with standardized paths.

        Args:
            files: List of file paths to upload
            gcs_bucket: GCS bucket name
            artifact_types: Mapping of filename patterns to artifact types

        Returns:
            List of OutputFile objects with GCS URLs

        GCS Path Structure:
            gs://{bucket}/runs/{run_id}/{model_name}/{filename}
        """
        from google.cloud import storage

        output_files: list[OutputFile] = []
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)

        for file_path in files:
            if not file_path.exists():
                self.logger.warning(f"File not found, skipping: {file_path}")
                continue

            # Determine artifact type
            artifact_type = self._determine_artifact_type(file_path.name, artifact_types)

            # GCS path: runs/{run_id}/{model_name}/{filename}
            gcs_path = f"runs/{self.run_id}/{self.model_name}/{file_path.name}"
            blob = bucket.blob(gcs_path)

            try:
                # Upload file
                blob.upload_from_filename(str(file_path))
                gcs_url = f"gs://{gcs_bucket}/{gcs_path}"

                # Get file size
                size_bytes = file_path.stat().st_size

                # Extract metadata if applicable
                metadata = self.extract_metadata(file_path, artifact_type)

                output_file = OutputFile(
                    filename=file_path.name,
                    artifact_type=artifact_type,
                    path=str(file_path),
                    gcs_url=gcs_url,
                    size_bytes=size_bytes,
                    metadata=metadata,
                )
                output_files.append(output_file)

                self.logger.info(f"Uploaded {file_path.name} to {gcs_url} ({size_bytes} bytes)")

            except Exception as e:
                self.logger.error(f"Failed to upload {file_path.name}: {e}")
                # Still add to output_files but without GCS URL
                output_file = OutputFile(
                    filename=file_path.name,
                    artifact_type=artifact_type,
                    path=str(file_path),
                    gcs_url=None,
                    size_bytes=file_path.stat().st_size if file_path.exists() else None,
                    metadata={"upload_error": str(e)},
                )
                output_files.append(output_file)

        return output_files

    def _determine_artifact_type(
        self, filename: str, artifact_types: dict[str, str] | None = None
    ) -> str:
        """
        Determine artifact type from filename.

        Args:
            filename: Name of the file
            artifact_types: Custom mapping of patterns to types

        Returns:
            Artifact type string
        """
        if artifact_types:
            for pattern, atype in artifact_types.items():
                if pattern in filename:
                    return atype

        # Default mappings
        suffix = Path(filename).suffix.lower()
        type_map = {
            ".pdb": "pdb",
            ".cif": "cif",
            ".fasta": "fasta",
            ".faa": "fasta",
            ".json": "json",
            ".csv": "csv",
            ".zip": "archive",
            ".tar": "archive",
            ".gz": "archive",
            ".log": "log",
            ".txt": "text",
            ".yaml": "yaml",
            ".yml": "yaml",
        }
        return type_map.get(suffix, "unknown")

    def extract_metadata(self, file_path: Path, artifact_type: str) -> dict[str, Any]:
        """
        Extract metadata from output file.

        Args:
            file_path: Path to the file
            artifact_type: Type of artifact

        Returns:
            Dictionary of metadata
        """
        metadata: dict[str, Any] = {}

        try:
            if artifact_type == "json":
                # Parse JSON files for metadata
                with Path(file_path).open() as f:
                    data = json.load(f)
                    # Extract common fields
                    for key in ["mean_plddt", "ptm", "pae", "num_residues", "sequence_length"]:
                        if key in data:
                            metadata[key] = data[key]

            elif artifact_type == "pdb":
                # Extract basic PDB metadata
                metadata.update(self._extract_pdb_metadata(file_path))

            elif artifact_type == "fasta":
                # Extract sequence metadata
                metadata.update(self._extract_fasta_metadata(file_path))

        except Exception as e:
            self.logger.debug(f"Could not extract metadata from {file_path.name}: {e}")
            metadata["metadata_error"] = str(e)

        return metadata

    def _extract_pdb_metadata(self, pdb_path: Path) -> dict[str, Any]:
        """Extract metadata from PDB file."""
        metadata: dict[str, Any] = {}

        try:
            with Path(pdb_path).open() as f:
                lines = f.readlines()

            # Count atoms and residues
            atom_count = sum(1 for line in lines if line.startswith("ATOM"))

            # Get residue numbers (simplified - assumes sequential)
            residue_nums = set()
            for line in lines:
                if line.startswith("ATOM"):
                    try:
                        res_num = int(line[22:26].strip())
                        residue_nums.add(res_num)
                    except (ValueError, IndexError):
                        pass

            metadata["num_atoms"] = atom_count
            metadata["num_residues"] = len(residue_nums)

            # Check for confidence scores in B-factor column (common for AlphaFold/ESMFold)
            if lines and any("ATOM" in line for line in lines[:100]):
                try:
                    b_factors = []
                    for line in lines:
                        if line.startswith("ATOM"):
                            b_factor = float(line[60:66].strip())
                            b_factors.append(b_factor)

                    if b_factors:
                        metadata["mean_confidence"] = sum(b_factors) / len(b_factors)
                        metadata["min_confidence"] = min(b_factors)
                        metadata["max_confidence"] = max(b_factors)
                except (ValueError, IndexError):
                    pass

        except Exception as e:
            self.logger.debug(f"Error extracting PDB metadata: {e}")

        return metadata

    def _extract_fasta_metadata(self, fasta_path: Path) -> dict[str, Any]:
        """Extract metadata from FASTA file."""
        metadata: dict[str, Any] = {}

        try:
            with Path(fasta_path).open() as f:
                content = f.read()

            # Count sequences
            num_sequences = content.count(">")
            metadata["num_sequences"] = num_sequences

            # Parse sequences
            sequences = []
            current_seq: list[str] = []
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        current_seq = []
                elif line:
                    current_seq.append(line)
            if current_seq:
                sequences.append("".join(current_seq))

            if sequences:
                metadata["sequence_lengths"] = [len(s) for s in sequences]
                metadata["total_residues"] = sum(len(s) for s in sequences)
                metadata["mean_length"] = sum(len(s) for s in sequences) / len(sequences)

        except Exception as e:
            self.logger.debug(f"Error extracting FASTA metadata: {e}")

        return metadata

    def handle_error(self, error: Exception, context: str = "") -> BiomodalResult:
        """
        Create standardized error response.

        Args:
            error: The exception that occurred
            context: Additional context about where error occurred

        Returns:
            BiomodalResult with error information
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.logger.error(f"Error in {self.model_name}: {error_msg}", exc_info=True)

        return BiomodalResult(
            success=False,
            output_files=[],
            message=f"Error: {error_msg}",
            error=error_msg,
            metadata={"error_type": type(error).__name__},
        )

    def format_output_files(
        self, files: list[Path | OutputFile], artifact_types: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Format output files into standardized dict format.

        Args:
            files: List of file paths or OutputFile objects
            artifact_types: Optional mapping for artifact type detection

        Returns:
            List of output file dictionaries
        """
        output_files = []

        for file in files:
            if isinstance(file, OutputFile):
                output_files.append(file.model_dump())
            elif isinstance(file, Path):
                artifact_type = self._determine_artifact_type(file.name, artifact_types)
                metadata = self.extract_metadata(file, artifact_type)

                output_file = {
                    "filename": file.name,
                    "artifact_type": artifact_type,
                    "path": str(file),
                    "gcs_url": None,
                    "size_bytes": file.stat().st_size if file.exists() else None,
                    "metadata": metadata,
                }
                output_files.append(output_file)

        return output_files

    def create_success_result(
        self,
        output_files: list[Path | OutputFile],
        message: str = "Function completed successfully",
        stdout: str = "",
        **metadata_fields: Any,
    ) -> BiomodalResult:
        """
        Create standardized success response.

        Args:
            output_files: List of output files
            message: Success message
            stdout: Standard output from execution
            **metadata_fields: Additional metadata fields (num_sequences, etc.)

        Returns:
            BiomodalResult with success status
        """
        # Format output files
        formatted_files = self.format_output_files(output_files)

        # Build metadata
        metadata = {
            "model": self.model_name,
            "run_id": self.run_id,
            **metadata_fields,
        }

        # Count output files
        num_output_files = len(formatted_files)

        return BiomodalResult(
            success=True,
            output_files=formatted_files,
            message=message,
            stdout=stdout,
            metadata=metadata,
            num_output_files=num_output_files,
            **{k: v for k, v in metadata_fields.items() if k in BiomodalResult.model_fields},
        )

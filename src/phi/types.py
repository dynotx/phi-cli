from typing import TypedDict


class IngestSessionResponse(TypedDict, total=False):
    session_id: str
    id: str
    status: str
    expected_files: int
    uploaded_files: int
    dataset_id: str
    artifact_count: int
    error: str


class SignedUrlEntry(TypedDict):
    file: str
    url: str


class SignedUrlsResponse(TypedDict, total=False):
    urls: list[SignedUrlEntry]


class JobSubmitResponse(TypedDict, total=False):
    job_id: str
    run_id: str
    status: str
    message: str


class PhiApiError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)

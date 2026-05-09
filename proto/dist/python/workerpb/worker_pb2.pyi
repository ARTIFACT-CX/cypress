from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ClientMsg(_message.Message):
    __slots__ = ("id", "status", "load_model", "unload", "shutdown", "run_wav", "start_stream", "audio_in", "stop_stream", "download_model", "cancel_download")
    ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LOAD_MODEL_FIELD_NUMBER: _ClassVar[int]
    UNLOAD_FIELD_NUMBER: _ClassVar[int]
    SHUTDOWN_FIELD_NUMBER: _ClassVar[int]
    RUN_WAV_FIELD_NUMBER: _ClassVar[int]
    START_STREAM_FIELD_NUMBER: _ClassVar[int]
    AUDIO_IN_FIELD_NUMBER: _ClassVar[int]
    STOP_STREAM_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_MODEL_FIELD_NUMBER: _ClassVar[int]
    CANCEL_DOWNLOAD_FIELD_NUMBER: _ClassVar[int]
    id: int
    status: StatusReq
    load_model: LoadModelReq
    unload: UnloadReq
    shutdown: ShutdownReq
    run_wav: RunWavReq
    start_stream: StartStreamReq
    audio_in: AudioInReq
    stop_stream: StopStreamReq
    download_model: DownloadModelReq
    cancel_download: CancelDownloadReq
    def __init__(self, id: _Optional[int] = ..., status: _Optional[_Union[StatusReq, _Mapping]] = ..., load_model: _Optional[_Union[LoadModelReq, _Mapping]] = ..., unload: _Optional[_Union[UnloadReq, _Mapping]] = ..., shutdown: _Optional[_Union[ShutdownReq, _Mapping]] = ..., run_wav: _Optional[_Union[RunWavReq, _Mapping]] = ..., start_stream: _Optional[_Union[StartStreamReq, _Mapping]] = ..., audio_in: _Optional[_Union[AudioInReq, _Mapping]] = ..., stop_stream: _Optional[_Union[StopStreamReq, _Mapping]] = ..., download_model: _Optional[_Union[DownloadModelReq, _Mapping]] = ..., cancel_download: _Optional[_Union[CancelDownloadReq, _Mapping]] = ...) -> None: ...

class StatusReq(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class UnloadReq(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ShutdownReq(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StartStreamReq(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StopStreamReq(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CancelDownloadReq(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class LoadModelReq(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class RunWavReq(_message.Message):
    __slots__ = ("input", "output")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    input: str
    output: str
    def __init__(self, input: _Optional[str] = ..., output: _Optional[str] = ...) -> None: ...

class AudioInReq(_message.Message):
    __slots__ = ("pcm",)
    PCM_FIELD_NUMBER: _ClassVar[int]
    pcm: bytes
    def __init__(self, pcm: _Optional[bytes] = ...) -> None: ...

class DownloadModelReq(_message.Message):
    __slots__ = ("name", "repo", "files", "revision")
    NAME_FIELD_NUMBER: _ClassVar[int]
    REPO_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    name: str
    repo: str
    files: _containers.RepeatedScalarFieldContainer[str]
    revision: str
    def __init__(self, name: _Optional[str] = ..., repo: _Optional[str] = ..., files: _Optional[_Iterable[str]] = ..., revision: _Optional[str] = ...) -> None: ...

class ServerMsg(_message.Message):
    __slots__ = ("handshake", "reply", "event")
    HANDSHAKE_FIELD_NUMBER: _ClassVar[int]
    REPLY_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    handshake: Handshake
    reply: Reply
    event: Event
    def __init__(self, handshake: _Optional[_Union[Handshake, _Mapping]] = ..., reply: _Optional[_Union[Reply, _Mapping]] = ..., event: _Optional[_Union[Event, _Mapping]] = ...) -> None: ...

class Handshake(_message.Message):
    __slots__ = ("ready", "fatal", "os", "arch", "available_backends", "downloaded_repos", "gpu_name", "gpu_memory_gb")
    READY_FIELD_NUMBER: _ClassVar[int]
    FATAL_FIELD_NUMBER: _ClassVar[int]
    OS_FIELD_NUMBER: _ClassVar[int]
    ARCH_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_BACKENDS_FIELD_NUMBER: _ClassVar[int]
    DOWNLOADED_REPOS_FIELD_NUMBER: _ClassVar[int]
    GPU_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_GB_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    fatal: str
    os: str
    arch: str
    available_backends: _containers.RepeatedScalarFieldContainer[str]
    downloaded_repos: _containers.RepeatedScalarFieldContainer[str]
    gpu_name: str
    gpu_memory_gb: int
    def __init__(self, ready: bool = ..., fatal: _Optional[str] = ..., os: _Optional[str] = ..., arch: _Optional[str] = ..., available_backends: _Optional[_Iterable[str]] = ..., downloaded_repos: _Optional[_Iterable[str]] = ..., gpu_name: _Optional[str] = ..., gpu_memory_gb: _Optional[int] = ...) -> None: ...

class Reply(_message.Message):
    __slots__ = ("id", "ok", "status", "load_model", "start_stream", "download_started", "cancel_download", "error")
    ID_FIELD_NUMBER: _ClassVar[int]
    OK_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LOAD_MODEL_FIELD_NUMBER: _ClassVar[int]
    START_STREAM_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_STARTED_FIELD_NUMBER: _ClassVar[int]
    CANCEL_DOWNLOAD_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    id: int
    ok: OkEmpty
    status: StatusOk
    load_model: LoadModelOk
    start_stream: StartStreamOk
    download_started: DownloadStartedOk
    cancel_download: CancelDownloadOk
    error: str
    def __init__(self, id: _Optional[int] = ..., ok: _Optional[_Union[OkEmpty, _Mapping]] = ..., status: _Optional[_Union[StatusOk, _Mapping]] = ..., load_model: _Optional[_Union[LoadModelOk, _Mapping]] = ..., start_stream: _Optional[_Union[StartStreamOk, _Mapping]] = ..., download_started: _Optional[_Union[DownloadStartedOk, _Mapping]] = ..., cancel_download: _Optional[_Union[CancelDownloadOk, _Mapping]] = ..., error: _Optional[str] = ...) -> None: ...

class OkEmpty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StatusOk(_message.Message):
    __slots__ = ("model", "device")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    model: str
    device: str
    def __init__(self, model: _Optional[str] = ..., device: _Optional[str] = ...) -> None: ...

class LoadModelOk(_message.Message):
    __slots__ = ("model", "device")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    model: str
    device: str
    def __init__(self, model: _Optional[str] = ..., device: _Optional[str] = ...) -> None: ...

class StartStreamOk(_message.Message):
    __slots__ = ("sample_rate",)
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    sample_rate: int
    def __init__(self, sample_rate: _Optional[int] = ...) -> None: ...

class DownloadStartedOk(_message.Message):
    __slots__ = ("started",)
    STARTED_FIELD_NUMBER: _ClassVar[int]
    started: bool
    def __init__(self, started: bool = ...) -> None: ...

class CancelDownloadOk(_message.Message):
    __slots__ = ("active",)
    ACTIVE_FIELD_NUMBER: _ClassVar[int]
    active: bool
    def __init__(self, active: bool = ...) -> None: ...

class Event(_message.Message):
    __slots__ = ("model_phase", "audio_out", "stream_error", "download_progress", "download_done", "download_error")
    MODEL_PHASE_FIELD_NUMBER: _ClassVar[int]
    AUDIO_OUT_FIELD_NUMBER: _ClassVar[int]
    STREAM_ERROR_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_DONE_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_ERROR_FIELD_NUMBER: _ClassVar[int]
    model_phase: ModelPhase
    audio_out: AudioOut
    stream_error: StreamError
    download_progress: DownloadProgress
    download_done: DownloadDone
    download_error: DownloadError
    def __init__(self, model_phase: _Optional[_Union[ModelPhase, _Mapping]] = ..., audio_out: _Optional[_Union[AudioOut, _Mapping]] = ..., stream_error: _Optional[_Union[StreamError, _Mapping]] = ..., download_progress: _Optional[_Union[DownloadProgress, _Mapping]] = ..., download_done: _Optional[_Union[DownloadDone, _Mapping]] = ..., download_error: _Optional[_Union[DownloadError, _Mapping]] = ...) -> None: ...

class ModelPhase(_message.Message):
    __slots__ = ("phase", "device")
    PHASE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    phase: str
    device: str
    def __init__(self, phase: _Optional[str] = ..., device: _Optional[str] = ...) -> None: ...

class AudioOut(_message.Message):
    __slots__ = ("pcm", "text")
    PCM_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    pcm: bytes
    text: str
    def __init__(self, pcm: _Optional[bytes] = ..., text: _Optional[str] = ...) -> None: ...

class StreamError(_message.Message):
    __slots__ = ("error",)
    ERROR_FIELD_NUMBER: _ClassVar[int]
    error: str
    def __init__(self, error: _Optional[str] = ...) -> None: ...

class DownloadProgress(_message.Message):
    __slots__ = ("name", "phase", "downloaded", "total", "file", "file_index", "file_count")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    DOWNLOADED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    FILE_INDEX_FIELD_NUMBER: _ClassVar[int]
    FILE_COUNT_FIELD_NUMBER: _ClassVar[int]
    name: str
    phase: str
    downloaded: int
    total: int
    file: str
    file_index: int
    file_count: int
    def __init__(self, name: _Optional[str] = ..., phase: _Optional[str] = ..., downloaded: _Optional[int] = ..., total: _Optional[int] = ..., file: _Optional[str] = ..., file_index: _Optional[int] = ..., file_count: _Optional[int] = ...) -> None: ...

class DownloadDone(_message.Message):
    __slots__ = ("name", "repo", "revision", "files", "total_bytes")
    NAME_FIELD_NUMBER: _ClassVar[int]
    REPO_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    name: str
    repo: str
    revision: str
    files: _containers.RepeatedScalarFieldContainer[str]
    total_bytes: int
    def __init__(self, name: _Optional[str] = ..., repo: _Optional[str] = ..., revision: _Optional[str] = ..., files: _Optional[_Iterable[str]] = ..., total_bytes: _Optional[int] = ...) -> None: ...

class DownloadError(_message.Message):
    __slots__ = ("name", "error")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    name: str
    error: str
    def __init__(self, name: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

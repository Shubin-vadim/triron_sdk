import model_config_pb2 as _model_config_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ServerLiveRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ServerLiveResponse(_message.Message):
    __slots__ = ("live",)
    LIVE_FIELD_NUMBER: _ClassVar[int]
    live: bool
    def __init__(self, live: _Optional[bool] = ...) -> None: ...

class ServerReadyRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ServerReadyResponse(_message.Message):
    __slots__ = ("ready",)
    READY_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    def __init__(self, ready: _Optional[bool] = ...) -> None: ...

class ModelReadyRequest(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class ModelReadyResponse(_message.Message):
    __slots__ = ("ready",)
    READY_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    def __init__(self, ready: _Optional[bool] = ...) -> None: ...

class ServerMetadataRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ServerMetadataResponse(_message.Message):
    __slots__ = ("name", "version", "extensions")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    EXTENSIONS_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    extensions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., extensions: _Optional[_Iterable[str]] = ...) -> None: ...

class ModelMetadataRequest(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class ModelMetadataResponse(_message.Message):
    __slots__ = ("name", "versions", "platform", "inputs", "outputs")
    class TensorMetadata(_message.Message):
        __slots__ = ("name", "datatype", "shape")
        NAME_FIELD_NUMBER: _ClassVar[int]
        DATATYPE_FIELD_NUMBER: _ClassVar[int]
        SHAPE_FIELD_NUMBER: _ClassVar[int]
        name: str
        datatype: str
        shape: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, name: _Optional[str] = ..., datatype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSIONS_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    versions: _containers.RepeatedScalarFieldContainer[str]
    platform: str
    inputs: _containers.RepeatedCompositeFieldContainer[ModelMetadataResponse.TensorMetadata]
    outputs: _containers.RepeatedCompositeFieldContainer[ModelMetadataResponse.TensorMetadata]
    def __init__(self, name: _Optional[str] = ..., versions: _Optional[_Iterable[str]] = ..., platform: _Optional[str] = ..., inputs: _Optional[_Iterable[_Union[ModelMetadataResponse.TensorMetadata, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[ModelMetadataResponse.TensorMetadata, _Mapping]]] = ...) -> None: ...

class InferParameter(_message.Message):
    __slots__ = ("bool_param", "int64_param", "string_param", "double_param", "uint64_param")
    BOOL_PARAM_FIELD_NUMBER: _ClassVar[int]
    INT64_PARAM_FIELD_NUMBER: _ClassVar[int]
    STRING_PARAM_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_PARAM_FIELD_NUMBER: _ClassVar[int]
    UINT64_PARAM_FIELD_NUMBER: _ClassVar[int]
    bool_param: bool
    int64_param: int
    string_param: str
    double_param: float
    uint64_param: int
    def __init__(self, bool_param: _Optional[bool] = ..., int64_param: _Optional[int] = ..., string_param: _Optional[str] = ..., double_param: _Optional[float] = ..., uint64_param: _Optional[int] = ...) -> None: ...

class InferTensorContents(_message.Message):
    __slots__ = ("bool_contents", "int_contents", "int64_contents", "uint_contents", "uint64_contents", "fp32_contents", "fp64_contents", "bytes_contents")
    BOOL_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    INT_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    INT64_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    UINT_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    UINT64_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    FP32_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    FP64_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    BYTES_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    bool_contents: _containers.RepeatedScalarFieldContainer[bool]
    int_contents: _containers.RepeatedScalarFieldContainer[int]
    int64_contents: _containers.RepeatedScalarFieldContainer[int]
    uint_contents: _containers.RepeatedScalarFieldContainer[int]
    uint64_contents: _containers.RepeatedScalarFieldContainer[int]
    fp32_contents: _containers.RepeatedScalarFieldContainer[float]
    fp64_contents: _containers.RepeatedScalarFieldContainer[float]
    bytes_contents: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, bool_contents: _Optional[_Iterable[bool]] = ..., int_contents: _Optional[_Iterable[int]] = ..., int64_contents: _Optional[_Iterable[int]] = ..., uint_contents: _Optional[_Iterable[int]] = ..., uint64_contents: _Optional[_Iterable[int]] = ..., fp32_contents: _Optional[_Iterable[float]] = ..., fp64_contents: _Optional[_Iterable[float]] = ..., bytes_contents: _Optional[_Iterable[bytes]] = ...) -> None: ...

class ModelInferRequest(_message.Message):
    __slots__ = ("model_name", "model_version", "id", "parameters", "inputs", "outputs", "raw_input_contents")
    class InferInputTensor(_message.Message):
        __slots__ = ("name", "datatype", "shape", "parameters", "contents")
        class ParametersEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: InferParameter
            def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[InferParameter, _Mapping]] = ...) -> None: ...
        NAME_FIELD_NUMBER: _ClassVar[int]
        DATATYPE_FIELD_NUMBER: _ClassVar[int]
        SHAPE_FIELD_NUMBER: _ClassVar[int]
        PARAMETERS_FIELD_NUMBER: _ClassVar[int]
        CONTENTS_FIELD_NUMBER: _ClassVar[int]
        name: str
        datatype: str
        shape: _containers.RepeatedScalarFieldContainer[int]
        parameters: _containers.MessageMap[str, InferParameter]
        contents: InferTensorContents
        def __init__(self, name: _Optional[str] = ..., datatype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., parameters: _Optional[_Mapping[str, InferParameter]] = ..., contents: _Optional[_Union[InferTensorContents, _Mapping]] = ...) -> None: ...
    class InferRequestedOutputTensor(_message.Message):
        __slots__ = ("name", "parameters")
        class ParametersEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: InferParameter
            def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[InferParameter, _Mapping]] = ...) -> None: ...
        NAME_FIELD_NUMBER: _ClassVar[int]
        PARAMETERS_FIELD_NUMBER: _ClassVar[int]
        name: str
        parameters: _containers.MessageMap[str, InferParameter]
        def __init__(self, name: _Optional[str] = ..., parameters: _Optional[_Mapping[str, InferParameter]] = ...) -> None: ...
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: InferParameter
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[InferParameter, _Mapping]] = ...) -> None: ...
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    RAW_INPUT_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    model_version: str
    id: str
    parameters: _containers.MessageMap[str, InferParameter]
    inputs: _containers.RepeatedCompositeFieldContainer[ModelInferRequest.InferInputTensor]
    outputs: _containers.RepeatedCompositeFieldContainer[ModelInferRequest.InferRequestedOutputTensor]
    raw_input_contents: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, model_name: _Optional[str] = ..., model_version: _Optional[str] = ..., id: _Optional[str] = ..., parameters: _Optional[_Mapping[str, InferParameter]] = ..., inputs: _Optional[_Iterable[_Union[ModelInferRequest.InferInputTensor, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[ModelInferRequest.InferRequestedOutputTensor, _Mapping]]] = ..., raw_input_contents: _Optional[_Iterable[bytes]] = ...) -> None: ...

class ModelInferResponse(_message.Message):
    __slots__ = ("model_name", "model_version", "id", "parameters", "outputs", "raw_output_contents")
    class InferOutputTensor(_message.Message):
        __slots__ = ("name", "datatype", "shape", "parameters", "contents")
        class ParametersEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: InferParameter
            def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[InferParameter, _Mapping]] = ...) -> None: ...
        NAME_FIELD_NUMBER: _ClassVar[int]
        DATATYPE_FIELD_NUMBER: _ClassVar[int]
        SHAPE_FIELD_NUMBER: _ClassVar[int]
        PARAMETERS_FIELD_NUMBER: _ClassVar[int]
        CONTENTS_FIELD_NUMBER: _ClassVar[int]
        name: str
        datatype: str
        shape: _containers.RepeatedScalarFieldContainer[int]
        parameters: _containers.MessageMap[str, InferParameter]
        contents: InferTensorContents
        def __init__(self, name: _Optional[str] = ..., datatype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., parameters: _Optional[_Mapping[str, InferParameter]] = ..., contents: _Optional[_Union[InferTensorContents, _Mapping]] = ...) -> None: ...
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: InferParameter
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[InferParameter, _Mapping]] = ...) -> None: ...
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    RAW_OUTPUT_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    model_version: str
    id: str
    parameters: _containers.MessageMap[str, InferParameter]
    outputs: _containers.RepeatedCompositeFieldContainer[ModelInferResponse.InferOutputTensor]
    raw_output_contents: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, model_name: _Optional[str] = ..., model_version: _Optional[str] = ..., id: _Optional[str] = ..., parameters: _Optional[_Mapping[str, InferParameter]] = ..., outputs: _Optional[_Iterable[_Union[ModelInferResponse.InferOutputTensor, _Mapping]]] = ..., raw_output_contents: _Optional[_Iterable[bytes]] = ...) -> None: ...

class ModelStreamInferResponse(_message.Message):
    __slots__ = ("error_message", "infer_response")
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    INFER_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    error_message: str
    infer_response: ModelInferResponse
    def __init__(self, error_message: _Optional[str] = ..., infer_response: _Optional[_Union[ModelInferResponse, _Mapping]] = ...) -> None: ...

class ModelConfigRequest(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class ModelConfigResponse(_message.Message):
    __slots__ = ("config",)
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    config: _model_config_pb2.ModelConfig
    def __init__(self, config: _Optional[_Union[_model_config_pb2.ModelConfig, _Mapping]] = ...) -> None: ...

class ModelStatisticsRequest(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class StatisticDuration(_message.Message):
    __slots__ = ("count", "ns")
    COUNT_FIELD_NUMBER: _ClassVar[int]
    NS_FIELD_NUMBER: _ClassVar[int]
    count: int
    ns: int
    def __init__(self, count: _Optional[int] = ..., ns: _Optional[int] = ...) -> None: ...

class InferStatistics(_message.Message):
    __slots__ = ("success", "fail", "queue", "compute_input", "compute_infer", "compute_output", "cache_hit", "cache_miss")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    FAIL_FIELD_NUMBER: _ClassVar[int]
    QUEUE_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_INPUT_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_INFER_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    CACHE_HIT_FIELD_NUMBER: _ClassVar[int]
    CACHE_MISS_FIELD_NUMBER: _ClassVar[int]
    success: StatisticDuration
    fail: StatisticDuration
    queue: StatisticDuration
    compute_input: StatisticDuration
    compute_infer: StatisticDuration
    compute_output: StatisticDuration
    cache_hit: StatisticDuration
    cache_miss: StatisticDuration
    def __init__(self, success: _Optional[_Union[StatisticDuration, _Mapping]] = ..., fail: _Optional[_Union[StatisticDuration, _Mapping]] = ..., queue: _Optional[_Union[StatisticDuration, _Mapping]] = ..., compute_input: _Optional[_Union[StatisticDuration, _Mapping]] = ..., compute_infer: _Optional[_Union[StatisticDuration, _Mapping]] = ..., compute_output: _Optional[_Union[StatisticDuration, _Mapping]] = ..., cache_hit: _Optional[_Union[StatisticDuration, _Mapping]] = ..., cache_miss: _Optional[_Union[StatisticDuration, _Mapping]] = ...) -> None: ...

class InferResponseStatistics(_message.Message):
    __slots__ = ("compute_infer", "compute_output", "success", "fail", "empty_response", "cancel")
    COMPUTE_INFER_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    FAIL_FIELD_NUMBER: _ClassVar[int]
    EMPTY_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    compute_infer: StatisticDuration
    compute_output: StatisticDuration
    success: StatisticDuration
    fail: StatisticDuration
    empty_response: StatisticDuration
    cancel: StatisticDuration
    def __init__(self, compute_infer: _Optional[_Union[StatisticDuration, _Mapping]] = ..., compute_output: _Optional[_Union[StatisticDuration, _Mapping]] = ..., success: _Optional[_Union[StatisticDuration, _Mapping]] = ..., fail: _Optional[_Union[StatisticDuration, _Mapping]] = ..., empty_response: _Optional[_Union[StatisticDuration, _Mapping]] = ..., cancel: _Optional[_Union[StatisticDuration, _Mapping]] = ...) -> None: ...

class InferBatchStatistics(_message.Message):
    __slots__ = ("batch_size", "compute_input", "compute_infer", "compute_output")
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_INPUT_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_INFER_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    batch_size: int
    compute_input: StatisticDuration
    compute_infer: StatisticDuration
    compute_output: StatisticDuration
    def __init__(self, batch_size: _Optional[int] = ..., compute_input: _Optional[_Union[StatisticDuration, _Mapping]] = ..., compute_infer: _Optional[_Union[StatisticDuration, _Mapping]] = ..., compute_output: _Optional[_Union[StatisticDuration, _Mapping]] = ...) -> None: ...

class MemoryUsage(_message.Message):
    __slots__ = ("type", "id", "byte_size")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    BYTE_SIZE_FIELD_NUMBER: _ClassVar[int]
    type: str
    id: int
    byte_size: int
    def __init__(self, type: _Optional[str] = ..., id: _Optional[int] = ..., byte_size: _Optional[int] = ...) -> None: ...

class ModelStatistics(_message.Message):
    __slots__ = ("name", "version", "last_inference", "inference_count", "execution_count", "inference_stats", "batch_stats", "memory_usage", "response_stats")
    class ResponseStatsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: InferResponseStatistics
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[InferResponseStatistics, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    LAST_INFERENCE_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_COUNT_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_COUNT_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_STATS_FIELD_NUMBER: _ClassVar[int]
    BATCH_STATS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USAGE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_STATS_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    last_inference: int
    inference_count: int
    execution_count: int
    inference_stats: InferStatistics
    batch_stats: _containers.RepeatedCompositeFieldContainer[InferBatchStatistics]
    memory_usage: _containers.RepeatedCompositeFieldContainer[MemoryUsage]
    response_stats: _containers.MessageMap[str, InferResponseStatistics]
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., last_inference: _Optional[int] = ..., inference_count: _Optional[int] = ..., execution_count: _Optional[int] = ..., inference_stats: _Optional[_Union[InferStatistics, _Mapping]] = ..., batch_stats: _Optional[_Iterable[_Union[InferBatchStatistics, _Mapping]]] = ..., memory_usage: _Optional[_Iterable[_Union[MemoryUsage, _Mapping]]] = ..., response_stats: _Optional[_Mapping[str, InferResponseStatistics]] = ...) -> None: ...

class ModelStatisticsResponse(_message.Message):
    __slots__ = ("model_stats",)
    MODEL_STATS_FIELD_NUMBER: _ClassVar[int]
    model_stats: _containers.RepeatedCompositeFieldContainer[ModelStatistics]
    def __init__(self, model_stats: _Optional[_Iterable[_Union[ModelStatistics, _Mapping]]] = ...) -> None: ...

class ModelRepositoryParameter(_message.Message):
    __slots__ = ("bool_param", "int64_param", "string_param", "bytes_param")
    BOOL_PARAM_FIELD_NUMBER: _ClassVar[int]
    INT64_PARAM_FIELD_NUMBER: _ClassVar[int]
    STRING_PARAM_FIELD_NUMBER: _ClassVar[int]
    BYTES_PARAM_FIELD_NUMBER: _ClassVar[int]
    bool_param: bool
    int64_param: int
    string_param: str
    bytes_param: bytes
    def __init__(self, bool_param: _Optional[bool] = ..., int64_param: _Optional[int] = ..., string_param: _Optional[str] = ..., bytes_param: _Optional[bytes] = ...) -> None: ...

class RepositoryIndexRequest(_message.Message):
    __slots__ = ("repository_name", "ready")
    REPOSITORY_NAME_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    repository_name: str
    ready: bool
    def __init__(self, repository_name: _Optional[str] = ..., ready: _Optional[bool] = ...) -> None: ...

class RepositoryIndexResponse(_message.Message):
    __slots__ = ("models",)
    class ModelIndex(_message.Message):
        __slots__ = ("name", "version", "state", "reason")
        NAME_FIELD_NUMBER: _ClassVar[int]
        VERSION_FIELD_NUMBER: _ClassVar[int]
        STATE_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        name: str
        version: str
        state: str
        reason: str
        def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., state: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[RepositoryIndexResponse.ModelIndex]
    def __init__(self, models: _Optional[_Iterable[_Union[RepositoryIndexResponse.ModelIndex, _Mapping]]] = ...) -> None: ...

class RepositoryModelLoadRequest(_message.Message):
    __slots__ = ("repository_name", "model_name", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ModelRepositoryParameter
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ModelRepositoryParameter, _Mapping]] = ...) -> None: ...
    REPOSITORY_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    repository_name: str
    model_name: str
    parameters: _containers.MessageMap[str, ModelRepositoryParameter]
    def __init__(self, repository_name: _Optional[str] = ..., model_name: _Optional[str] = ..., parameters: _Optional[_Mapping[str, ModelRepositoryParameter]] = ...) -> None: ...

class RepositoryModelLoadResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RepositoryModelUnloadRequest(_message.Message):
    __slots__ = ("repository_name", "model_name", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ModelRepositoryParameter
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ModelRepositoryParameter, _Mapping]] = ...) -> None: ...
    REPOSITORY_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    repository_name: str
    model_name: str
    parameters: _containers.MessageMap[str, ModelRepositoryParameter]
    def __init__(self, repository_name: _Optional[str] = ..., model_name: _Optional[str] = ..., parameters: _Optional[_Mapping[str, ModelRepositoryParameter]] = ...) -> None: ...

class RepositoryModelUnloadResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SystemSharedMemoryStatusRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class SystemSharedMemoryStatusResponse(_message.Message):
    __slots__ = ("regions",)
    class RegionStatus(_message.Message):
        __slots__ = ("name", "key", "offset", "byte_size")
        NAME_FIELD_NUMBER: _ClassVar[int]
        KEY_FIELD_NUMBER: _ClassVar[int]
        OFFSET_FIELD_NUMBER: _ClassVar[int]
        BYTE_SIZE_FIELD_NUMBER: _ClassVar[int]
        name: str
        key: str
        offset: int
        byte_size: int
        def __init__(self, name: _Optional[str] = ..., key: _Optional[str] = ..., offset: _Optional[int] = ..., byte_size: _Optional[int] = ...) -> None: ...
    class RegionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: SystemSharedMemoryStatusResponse.RegionStatus
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[SystemSharedMemoryStatusResponse.RegionStatus, _Mapping]] = ...) -> None: ...
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    regions: _containers.MessageMap[str, SystemSharedMemoryStatusResponse.RegionStatus]
    def __init__(self, regions: _Optional[_Mapping[str, SystemSharedMemoryStatusResponse.RegionStatus]] = ...) -> None: ...

class SystemSharedMemoryRegisterRequest(_message.Message):
    __slots__ = ("name", "key", "offset", "byte_size")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    BYTE_SIZE_FIELD_NUMBER: _ClassVar[int]
    name: str
    key: str
    offset: int
    byte_size: int
    def __init__(self, name: _Optional[str] = ..., key: _Optional[str] = ..., offset: _Optional[int] = ..., byte_size: _Optional[int] = ...) -> None: ...

class SystemSharedMemoryRegisterResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SystemSharedMemoryUnregisterRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class SystemSharedMemoryUnregisterResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CudaSharedMemoryStatusRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class CudaSharedMemoryStatusResponse(_message.Message):
    __slots__ = ("regions",)
    class RegionStatus(_message.Message):
        __slots__ = ("name", "device_id", "byte_size")
        NAME_FIELD_NUMBER: _ClassVar[int]
        DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
        BYTE_SIZE_FIELD_NUMBER: _ClassVar[int]
        name: str
        device_id: int
        byte_size: int
        def __init__(self, name: _Optional[str] = ..., device_id: _Optional[int] = ..., byte_size: _Optional[int] = ...) -> None: ...
    class RegionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: CudaSharedMemoryStatusResponse.RegionStatus
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[CudaSharedMemoryStatusResponse.RegionStatus, _Mapping]] = ...) -> None: ...
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    regions: _containers.MessageMap[str, CudaSharedMemoryStatusResponse.RegionStatus]
    def __init__(self, regions: _Optional[_Mapping[str, CudaSharedMemoryStatusResponse.RegionStatus]] = ...) -> None: ...

class CudaSharedMemoryRegisterRequest(_message.Message):
    __slots__ = ("name", "raw_handle", "device_id", "byte_size")
    NAME_FIELD_NUMBER: _ClassVar[int]
    RAW_HANDLE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    BYTE_SIZE_FIELD_NUMBER: _ClassVar[int]
    name: str
    raw_handle: bytes
    device_id: int
    byte_size: int
    def __init__(self, name: _Optional[str] = ..., raw_handle: _Optional[bytes] = ..., device_id: _Optional[int] = ..., byte_size: _Optional[int] = ...) -> None: ...

class CudaSharedMemoryRegisterResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CudaSharedMemoryUnregisterRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class CudaSharedMemoryUnregisterResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TraceSettingRequest(_message.Message):
    __slots__ = ("settings", "model_name")
    class SettingValue(_message.Message):
        __slots__ = ("value",)
        VALUE_FIELD_NUMBER: _ClassVar[int]
        value: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, value: _Optional[_Iterable[str]] = ...) -> None: ...
    class SettingsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: TraceSettingRequest.SettingValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[TraceSettingRequest.SettingValue, _Mapping]] = ...) -> None: ...
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    settings: _containers.MessageMap[str, TraceSettingRequest.SettingValue]
    model_name: str
    def __init__(self, settings: _Optional[_Mapping[str, TraceSettingRequest.SettingValue]] = ..., model_name: _Optional[str] = ...) -> None: ...

class TraceSettingResponse(_message.Message):
    __slots__ = ("settings",)
    class SettingValue(_message.Message):
        __slots__ = ("value",)
        VALUE_FIELD_NUMBER: _ClassVar[int]
        value: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, value: _Optional[_Iterable[str]] = ...) -> None: ...
    class SettingsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: TraceSettingResponse.SettingValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[TraceSettingResponse.SettingValue, _Mapping]] = ...) -> None: ...
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    settings: _containers.MessageMap[str, TraceSettingResponse.SettingValue]
    def __init__(self, settings: _Optional[_Mapping[str, TraceSettingResponse.SettingValue]] = ...) -> None: ...

class LogSettingsRequest(_message.Message):
    __slots__ = ("settings",)
    class SettingValue(_message.Message):
        __slots__ = ("bool_param", "uint32_param", "string_param")
        BOOL_PARAM_FIELD_NUMBER: _ClassVar[int]
        UINT32_PARAM_FIELD_NUMBER: _ClassVar[int]
        STRING_PARAM_FIELD_NUMBER: _ClassVar[int]
        bool_param: bool
        uint32_param: int
        string_param: str
        def __init__(self, bool_param: _Optional[bool] = ..., uint32_param: _Optional[int] = ..., string_param: _Optional[str] = ...) -> None: ...
    class SettingsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: LogSettingsRequest.SettingValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[LogSettingsRequest.SettingValue, _Mapping]] = ...) -> None: ...
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    settings: _containers.MessageMap[str, LogSettingsRequest.SettingValue]
    def __init__(self, settings: _Optional[_Mapping[str, LogSettingsRequest.SettingValue]] = ...) -> None: ...

class LogSettingsResponse(_message.Message):
    __slots__ = ("settings",)
    class SettingValue(_message.Message):
        __slots__ = ("bool_param", "uint32_param", "string_param")
        BOOL_PARAM_FIELD_NUMBER: _ClassVar[int]
        UINT32_PARAM_FIELD_NUMBER: _ClassVar[int]
        STRING_PARAM_FIELD_NUMBER: _ClassVar[int]
        bool_param: bool
        uint32_param: int
        string_param: str
        def __init__(self, bool_param: _Optional[bool] = ..., uint32_param: _Optional[int] = ..., string_param: _Optional[str] = ...) -> None: ...
    class SettingsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: LogSettingsResponse.SettingValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[LogSettingsResponse.SettingValue, _Mapping]] = ...) -> None: ...
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    settings: _containers.MessageMap[str, LogSettingsResponse.SettingValue]
    def __init__(self, settings: _Optional[_Mapping[str, LogSettingsResponse.SettingValue]] = ...) -> None: ...

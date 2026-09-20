from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TYPE_INVALID: _ClassVar[DataType]
    TYPE_BOOL: _ClassVar[DataType]
    TYPE_UINT8: _ClassVar[DataType]
    TYPE_UINT16: _ClassVar[DataType]
    TYPE_UINT32: _ClassVar[DataType]
    TYPE_UINT64: _ClassVar[DataType]
    TYPE_INT8: _ClassVar[DataType]
    TYPE_INT16: _ClassVar[DataType]
    TYPE_INT32: _ClassVar[DataType]
    TYPE_INT64: _ClassVar[DataType]
    TYPE_FP16: _ClassVar[DataType]
    TYPE_FP32: _ClassVar[DataType]
    TYPE_FP64: _ClassVar[DataType]
    TYPE_STRING: _ClassVar[DataType]
    TYPE_BF16: _ClassVar[DataType]
TYPE_INVALID: DataType
TYPE_BOOL: DataType
TYPE_UINT8: DataType
TYPE_UINT16: DataType
TYPE_UINT32: DataType
TYPE_UINT64: DataType
TYPE_INT8: DataType
TYPE_INT16: DataType
TYPE_INT32: DataType
TYPE_INT64: DataType
TYPE_FP16: DataType
TYPE_FP32: DataType
TYPE_FP64: DataType
TYPE_STRING: DataType
TYPE_BF16: DataType

class ModelRateLimiter(_message.Message):
    __slots__ = ("resources", "priority")
    class Resource(_message.Message):
        __slots__ = ("name", "count")
        NAME_FIELD_NUMBER: _ClassVar[int]
        GLOBAL_FIELD_NUMBER: _ClassVar[int]
        COUNT_FIELD_NUMBER: _ClassVar[int]
        name: str
        count: int
        def __init__(self, name: _Optional[str] = ..., count: _Optional[int] = ..., **kwargs) -> None: ...
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    resources: _containers.RepeatedCompositeFieldContainer[ModelRateLimiter.Resource]
    priority: int
    def __init__(self, resources: _Optional[_Iterable[_Union[ModelRateLimiter.Resource, _Mapping]]] = ..., priority: _Optional[int] = ...) -> None: ...

class ModelInstanceGroup(_message.Message):
    __slots__ = ("name", "kind", "count", "rate_limiter", "gpus", "secondary_devices", "profile", "passive", "host_policy")
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        KIND_AUTO: _ClassVar[ModelInstanceGroup.Kind]
        KIND_GPU: _ClassVar[ModelInstanceGroup.Kind]
        KIND_CPU: _ClassVar[ModelInstanceGroup.Kind]
        KIND_MODEL: _ClassVar[ModelInstanceGroup.Kind]
    KIND_AUTO: ModelInstanceGroup.Kind
    KIND_GPU: ModelInstanceGroup.Kind
    KIND_CPU: ModelInstanceGroup.Kind
    KIND_MODEL: ModelInstanceGroup.Kind
    class SecondaryDevice(_message.Message):
        __slots__ = ("kind", "device_id")
        class SecondaryDeviceKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            KIND_NVDLA: _ClassVar[ModelInstanceGroup.SecondaryDevice.SecondaryDeviceKind]
        KIND_NVDLA: ModelInstanceGroup.SecondaryDevice.SecondaryDeviceKind
        KIND_FIELD_NUMBER: _ClassVar[int]
        DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
        kind: ModelInstanceGroup.SecondaryDevice.SecondaryDeviceKind
        device_id: int
        def __init__(self, kind: _Optional[_Union[ModelInstanceGroup.SecondaryDevice.SecondaryDeviceKind, str]] = ..., device_id: _Optional[int] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    RATE_LIMITER_FIELD_NUMBER: _ClassVar[int]
    GPUS_FIELD_NUMBER: _ClassVar[int]
    SECONDARY_DEVICES_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    PASSIVE_FIELD_NUMBER: _ClassVar[int]
    HOST_POLICY_FIELD_NUMBER: _ClassVar[int]
    name: str
    kind: ModelInstanceGroup.Kind
    count: int
    rate_limiter: ModelRateLimiter
    gpus: _containers.RepeatedScalarFieldContainer[int]
    secondary_devices: _containers.RepeatedCompositeFieldContainer[ModelInstanceGroup.SecondaryDevice]
    profile: _containers.RepeatedScalarFieldContainer[str]
    passive: bool
    host_policy: str
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[_Union[ModelInstanceGroup.Kind, str]] = ..., count: _Optional[int] = ..., rate_limiter: _Optional[_Union[ModelRateLimiter, _Mapping]] = ..., gpus: _Optional[_Iterable[int]] = ..., secondary_devices: _Optional[_Iterable[_Union[ModelInstanceGroup.SecondaryDevice, _Mapping]]] = ..., profile: _Optional[_Iterable[str]] = ..., passive: _Optional[bool] = ..., host_policy: _Optional[str] = ...) -> None: ...

class ModelTensorReshape(_message.Message):
    __slots__ = ("shape",)
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, shape: _Optional[_Iterable[int]] = ...) -> None: ...

class ModelInput(_message.Message):
    __slots__ = ("name", "data_type", "format", "dims", "reshape", "is_shape_tensor", "allow_ragged_batch", "optional", "is_non_linear_format_io")
    class Format(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FORMAT_NONE: _ClassVar[ModelInput.Format]
        FORMAT_NHWC: _ClassVar[ModelInput.Format]
        FORMAT_NCHW: _ClassVar[ModelInput.Format]
    FORMAT_NONE: ModelInput.Format
    FORMAT_NHWC: ModelInput.Format
    FORMAT_NCHW: ModelInput.Format
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    DIMS_FIELD_NUMBER: _ClassVar[int]
    RESHAPE_FIELD_NUMBER: _ClassVar[int]
    IS_SHAPE_TENSOR_FIELD_NUMBER: _ClassVar[int]
    ALLOW_RAGGED_BATCH_FIELD_NUMBER: _ClassVar[int]
    OPTIONAL_FIELD_NUMBER: _ClassVar[int]
    IS_NON_LINEAR_FORMAT_IO_FIELD_NUMBER: _ClassVar[int]
    name: str
    data_type: DataType
    format: ModelInput.Format
    dims: _containers.RepeatedScalarFieldContainer[int]
    reshape: ModelTensorReshape
    is_shape_tensor: bool
    allow_ragged_batch: bool
    optional: bool
    is_non_linear_format_io: bool
    def __init__(self, name: _Optional[str] = ..., data_type: _Optional[_Union[DataType, str]] = ..., format: _Optional[_Union[ModelInput.Format, str]] = ..., dims: _Optional[_Iterable[int]] = ..., reshape: _Optional[_Union[ModelTensorReshape, _Mapping]] = ..., is_shape_tensor: _Optional[bool] = ..., allow_ragged_batch: _Optional[bool] = ..., optional: _Optional[bool] = ..., is_non_linear_format_io: _Optional[bool] = ...) -> None: ...

class ModelOutput(_message.Message):
    __slots__ = ("name", "data_type", "dims", "reshape", "label_filename", "is_shape_tensor", "is_non_linear_format_io")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    DIMS_FIELD_NUMBER: _ClassVar[int]
    RESHAPE_FIELD_NUMBER: _ClassVar[int]
    LABEL_FILENAME_FIELD_NUMBER: _ClassVar[int]
    IS_SHAPE_TENSOR_FIELD_NUMBER: _ClassVar[int]
    IS_NON_LINEAR_FORMAT_IO_FIELD_NUMBER: _ClassVar[int]
    name: str
    data_type: DataType
    dims: _containers.RepeatedScalarFieldContainer[int]
    reshape: ModelTensorReshape
    label_filename: str
    is_shape_tensor: bool
    is_non_linear_format_io: bool
    def __init__(self, name: _Optional[str] = ..., data_type: _Optional[_Union[DataType, str]] = ..., dims: _Optional[_Iterable[int]] = ..., reshape: _Optional[_Union[ModelTensorReshape, _Mapping]] = ..., label_filename: _Optional[str] = ..., is_shape_tensor: _Optional[bool] = ..., is_non_linear_format_io: _Optional[bool] = ...) -> None: ...

class BatchInput(_message.Message):
    __slots__ = ("kind", "target_name", "data_type", "source_input")
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        BATCH_ELEMENT_COUNT: _ClassVar[BatchInput.Kind]
        BATCH_ACCUMULATED_ELEMENT_COUNT: _ClassVar[BatchInput.Kind]
        BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO: _ClassVar[BatchInput.Kind]
        BATCH_MAX_ELEMENT_COUNT_AS_SHAPE: _ClassVar[BatchInput.Kind]
        BATCH_ITEM_SHAPE: _ClassVar[BatchInput.Kind]
        BATCH_ITEM_SHAPE_FLATTEN: _ClassVar[BatchInput.Kind]
    BATCH_ELEMENT_COUNT: BatchInput.Kind
    BATCH_ACCUMULATED_ELEMENT_COUNT: BatchInput.Kind
    BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO: BatchInput.Kind
    BATCH_MAX_ELEMENT_COUNT_AS_SHAPE: BatchInput.Kind
    BATCH_ITEM_SHAPE: BatchInput.Kind
    BATCH_ITEM_SHAPE_FLATTEN: BatchInput.Kind
    KIND_FIELD_NUMBER: _ClassVar[int]
    TARGET_NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_INPUT_FIELD_NUMBER: _ClassVar[int]
    kind: BatchInput.Kind
    target_name: _containers.RepeatedScalarFieldContainer[str]
    data_type: DataType
    source_input: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, kind: _Optional[_Union[BatchInput.Kind, str]] = ..., target_name: _Optional[_Iterable[str]] = ..., data_type: _Optional[_Union[DataType, str]] = ..., source_input: _Optional[_Iterable[str]] = ...) -> None: ...

class BatchOutput(_message.Message):
    __slots__ = ("target_name", "kind", "source_input")
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        BATCH_SCATTER_WITH_INPUT_SHAPE: _ClassVar[BatchOutput.Kind]
    BATCH_SCATTER_WITH_INPUT_SHAPE: BatchOutput.Kind
    TARGET_NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    SOURCE_INPUT_FIELD_NUMBER: _ClassVar[int]
    target_name: _containers.RepeatedScalarFieldContainer[str]
    kind: BatchOutput.Kind
    source_input: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, target_name: _Optional[_Iterable[str]] = ..., kind: _Optional[_Union[BatchOutput.Kind, str]] = ..., source_input: _Optional[_Iterable[str]] = ...) -> None: ...

class ModelVersionPolicy(_message.Message):
    __slots__ = ("latest", "all", "specific")
    class Latest(_message.Message):
        __slots__ = ("num_versions",)
        NUM_VERSIONS_FIELD_NUMBER: _ClassVar[int]
        num_versions: int
        def __init__(self, num_versions: _Optional[int] = ...) -> None: ...
    class All(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class Specific(_message.Message):
        __slots__ = ("versions",)
        VERSIONS_FIELD_NUMBER: _ClassVar[int]
        versions: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, versions: _Optional[_Iterable[int]] = ...) -> None: ...
    LATEST_FIELD_NUMBER: _ClassVar[int]
    ALL_FIELD_NUMBER: _ClassVar[int]
    SPECIFIC_FIELD_NUMBER: _ClassVar[int]
    latest: ModelVersionPolicy.Latest
    all: ModelVersionPolicy.All
    specific: ModelVersionPolicy.Specific
    def __init__(self, latest: _Optional[_Union[ModelVersionPolicy.Latest, _Mapping]] = ..., all: _Optional[_Union[ModelVersionPolicy.All, _Mapping]] = ..., specific: _Optional[_Union[ModelVersionPolicy.Specific, _Mapping]] = ...) -> None: ...

class ModelOptimizationPolicy(_message.Message):
    __slots__ = ("graph", "priority", "cuda", "execution_accelerators", "input_pinned_memory", "output_pinned_memory", "gather_kernel_buffer_threshold", "eager_batching")
    class ModelPriority(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        PRIORITY_DEFAULT: _ClassVar[ModelOptimizationPolicy.ModelPriority]
        PRIORITY_MAX: _ClassVar[ModelOptimizationPolicy.ModelPriority]
        PRIORITY_MIN: _ClassVar[ModelOptimizationPolicy.ModelPriority]
    PRIORITY_DEFAULT: ModelOptimizationPolicy.ModelPriority
    PRIORITY_MAX: ModelOptimizationPolicy.ModelPriority
    PRIORITY_MIN: ModelOptimizationPolicy.ModelPriority
    class Graph(_message.Message):
        __slots__ = ("level",)
        LEVEL_FIELD_NUMBER: _ClassVar[int]
        level: int
        def __init__(self, level: _Optional[int] = ...) -> None: ...
    class Cuda(_message.Message):
        __slots__ = ("graphs", "busy_wait_events", "graph_spec", "output_copy_stream")
        class GraphSpec(_message.Message):
            __slots__ = ("batch_size", "input", "graph_lower_bound")
            class Shape(_message.Message):
                __slots__ = ("dim",)
                DIM_FIELD_NUMBER: _ClassVar[int]
                dim: _containers.RepeatedScalarFieldContainer[int]
                def __init__(self, dim: _Optional[_Iterable[int]] = ...) -> None: ...
            class LowerBound(_message.Message):
                __slots__ = ("batch_size", "input")
                class InputEntry(_message.Message):
                    __slots__ = ("key", "value")
                    KEY_FIELD_NUMBER: _ClassVar[int]
                    VALUE_FIELD_NUMBER: _ClassVar[int]
                    key: str
                    value: ModelOptimizationPolicy.Cuda.GraphSpec.Shape
                    def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ModelOptimizationPolicy.Cuda.GraphSpec.Shape, _Mapping]] = ...) -> None: ...
                BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
                INPUT_FIELD_NUMBER: _ClassVar[int]
                batch_size: int
                input: _containers.MessageMap[str, ModelOptimizationPolicy.Cuda.GraphSpec.Shape]
                def __init__(self, batch_size: _Optional[int] = ..., input: _Optional[_Mapping[str, ModelOptimizationPolicy.Cuda.GraphSpec.Shape]] = ...) -> None: ...
            class InputEntry(_message.Message):
                __slots__ = ("key", "value")
                KEY_FIELD_NUMBER: _ClassVar[int]
                VALUE_FIELD_NUMBER: _ClassVar[int]
                key: str
                value: ModelOptimizationPolicy.Cuda.GraphSpec.Shape
                def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ModelOptimizationPolicy.Cuda.GraphSpec.Shape, _Mapping]] = ...) -> None: ...
            BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
            INPUT_FIELD_NUMBER: _ClassVar[int]
            GRAPH_LOWER_BOUND_FIELD_NUMBER: _ClassVar[int]
            batch_size: int
            input: _containers.MessageMap[str, ModelOptimizationPolicy.Cuda.GraphSpec.Shape]
            graph_lower_bound: ModelOptimizationPolicy.Cuda.GraphSpec.LowerBound
            def __init__(self, batch_size: _Optional[int] = ..., input: _Optional[_Mapping[str, ModelOptimizationPolicy.Cuda.GraphSpec.Shape]] = ..., graph_lower_bound: _Optional[_Union[ModelOptimizationPolicy.Cuda.GraphSpec.LowerBound, _Mapping]] = ...) -> None: ...
        GRAPHS_FIELD_NUMBER: _ClassVar[int]
        BUSY_WAIT_EVENTS_FIELD_NUMBER: _ClassVar[int]
        GRAPH_SPEC_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_COPY_STREAM_FIELD_NUMBER: _ClassVar[int]
        graphs: bool
        busy_wait_events: bool
        graph_spec: _containers.RepeatedCompositeFieldContainer[ModelOptimizationPolicy.Cuda.GraphSpec]
        output_copy_stream: bool
        def __init__(self, graphs: _Optional[bool] = ..., busy_wait_events: _Optional[bool] = ..., graph_spec: _Optional[_Iterable[_Union[ModelOptimizationPolicy.Cuda.GraphSpec, _Mapping]]] = ..., output_copy_stream: _Optional[bool] = ...) -> None: ...
    class ExecutionAccelerators(_message.Message):
        __slots__ = ("gpu_execution_accelerator", "cpu_execution_accelerator")
        class Accelerator(_message.Message):
            __slots__ = ("name", "parameters")
            class ParametersEntry(_message.Message):
                __slots__ = ("key", "value")
                KEY_FIELD_NUMBER: _ClassVar[int]
                VALUE_FIELD_NUMBER: _ClassVar[int]
                key: str
                value: str
                def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
            NAME_FIELD_NUMBER: _ClassVar[int]
            PARAMETERS_FIELD_NUMBER: _ClassVar[int]
            name: str
            parameters: _containers.ScalarMap[str, str]
            def __init__(self, name: _Optional[str] = ..., parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...
        GPU_EXECUTION_ACCELERATOR_FIELD_NUMBER: _ClassVar[int]
        CPU_EXECUTION_ACCELERATOR_FIELD_NUMBER: _ClassVar[int]
        gpu_execution_accelerator: _containers.RepeatedCompositeFieldContainer[ModelOptimizationPolicy.ExecutionAccelerators.Accelerator]
        cpu_execution_accelerator: _containers.RepeatedCompositeFieldContainer[ModelOptimizationPolicy.ExecutionAccelerators.Accelerator]
        def __init__(self, gpu_execution_accelerator: _Optional[_Iterable[_Union[ModelOptimizationPolicy.ExecutionAccelerators.Accelerator, _Mapping]]] = ..., cpu_execution_accelerator: _Optional[_Iterable[_Union[ModelOptimizationPolicy.ExecutionAccelerators.Accelerator, _Mapping]]] = ...) -> None: ...
    class PinnedMemoryBuffer(_message.Message):
        __slots__ = ("enable",)
        ENABLE_FIELD_NUMBER: _ClassVar[int]
        enable: bool
        def __init__(self, enable: _Optional[bool] = ...) -> None: ...
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    CUDA_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ACCELERATORS_FIELD_NUMBER: _ClassVar[int]
    INPUT_PINNED_MEMORY_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PINNED_MEMORY_FIELD_NUMBER: _ClassVar[int]
    GATHER_KERNEL_BUFFER_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    EAGER_BATCHING_FIELD_NUMBER: _ClassVar[int]
    graph: ModelOptimizationPolicy.Graph
    priority: ModelOptimizationPolicy.ModelPriority
    cuda: ModelOptimizationPolicy.Cuda
    execution_accelerators: ModelOptimizationPolicy.ExecutionAccelerators
    input_pinned_memory: ModelOptimizationPolicy.PinnedMemoryBuffer
    output_pinned_memory: ModelOptimizationPolicy.PinnedMemoryBuffer
    gather_kernel_buffer_threshold: int
    eager_batching: bool
    def __init__(self, graph: _Optional[_Union[ModelOptimizationPolicy.Graph, _Mapping]] = ..., priority: _Optional[_Union[ModelOptimizationPolicy.ModelPriority, str]] = ..., cuda: _Optional[_Union[ModelOptimizationPolicy.Cuda, _Mapping]] = ..., execution_accelerators: _Optional[_Union[ModelOptimizationPolicy.ExecutionAccelerators, _Mapping]] = ..., input_pinned_memory: _Optional[_Union[ModelOptimizationPolicy.PinnedMemoryBuffer, _Mapping]] = ..., output_pinned_memory: _Optional[_Union[ModelOptimizationPolicy.PinnedMemoryBuffer, _Mapping]] = ..., gather_kernel_buffer_threshold: _Optional[int] = ..., eager_batching: _Optional[bool] = ...) -> None: ...

class ModelQueuePolicy(_message.Message):
    __slots__ = ("timeout_action", "default_timeout_microseconds", "allow_timeout_override", "max_queue_size")
    class TimeoutAction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        REJECT: _ClassVar[ModelQueuePolicy.TimeoutAction]
        DELAY: _ClassVar[ModelQueuePolicy.TimeoutAction]
    REJECT: ModelQueuePolicy.TimeoutAction
    DELAY: ModelQueuePolicy.TimeoutAction
    TIMEOUT_ACTION_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_TIMEOUT_MICROSECONDS_FIELD_NUMBER: _ClassVar[int]
    ALLOW_TIMEOUT_OVERRIDE_FIELD_NUMBER: _ClassVar[int]
    MAX_QUEUE_SIZE_FIELD_NUMBER: _ClassVar[int]
    timeout_action: ModelQueuePolicy.TimeoutAction
    default_timeout_microseconds: int
    allow_timeout_override: bool
    max_queue_size: int
    def __init__(self, timeout_action: _Optional[_Union[ModelQueuePolicy.TimeoutAction, str]] = ..., default_timeout_microseconds: _Optional[int] = ..., allow_timeout_override: _Optional[bool] = ..., max_queue_size: _Optional[int] = ...) -> None: ...

class ModelDynamicBatching(_message.Message):
    __slots__ = ("preferred_batch_size", "max_queue_delay_microseconds", "preserve_ordering", "priority_levels", "default_priority_level", "default_queue_policy", "priority_queue_policy")
    class PriorityQueuePolicyEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: ModelQueuePolicy
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[ModelQueuePolicy, _Mapping]] = ...) -> None: ...
    PREFERRED_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_QUEUE_DELAY_MICROSECONDS_FIELD_NUMBER: _ClassVar[int]
    PRESERVE_ORDERING_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_LEVELS_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_PRIORITY_LEVEL_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_QUEUE_POLICY_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_QUEUE_POLICY_FIELD_NUMBER: _ClassVar[int]
    preferred_batch_size: _containers.RepeatedScalarFieldContainer[int]
    max_queue_delay_microseconds: int
    preserve_ordering: bool
    priority_levels: int
    default_priority_level: int
    default_queue_policy: ModelQueuePolicy
    priority_queue_policy: _containers.MessageMap[int, ModelQueuePolicy]
    def __init__(self, preferred_batch_size: _Optional[_Iterable[int]] = ..., max_queue_delay_microseconds: _Optional[int] = ..., preserve_ordering: _Optional[bool] = ..., priority_levels: _Optional[int] = ..., default_priority_level: _Optional[int] = ..., default_queue_policy: _Optional[_Union[ModelQueuePolicy, _Mapping]] = ..., priority_queue_policy: _Optional[_Mapping[int, ModelQueuePolicy]] = ...) -> None: ...

class ModelSequenceBatching(_message.Message):
    __slots__ = ("direct", "oldest", "max_sequence_idle_microseconds", "control_input", "state", "iterative_sequence")
    class Control(_message.Message):
        __slots__ = ("kind", "int32_false_true", "fp32_false_true", "bool_false_true", "data_type")
        class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            CONTROL_SEQUENCE_START: _ClassVar[ModelSequenceBatching.Control.Kind]
            CONTROL_SEQUENCE_READY: _ClassVar[ModelSequenceBatching.Control.Kind]
            CONTROL_SEQUENCE_END: _ClassVar[ModelSequenceBatching.Control.Kind]
            CONTROL_SEQUENCE_CORRID: _ClassVar[ModelSequenceBatching.Control.Kind]
        CONTROL_SEQUENCE_START: ModelSequenceBatching.Control.Kind
        CONTROL_SEQUENCE_READY: ModelSequenceBatching.Control.Kind
        CONTROL_SEQUENCE_END: ModelSequenceBatching.Control.Kind
        CONTROL_SEQUENCE_CORRID: ModelSequenceBatching.Control.Kind
        KIND_FIELD_NUMBER: _ClassVar[int]
        INT32_FALSE_TRUE_FIELD_NUMBER: _ClassVar[int]
        FP32_FALSE_TRUE_FIELD_NUMBER: _ClassVar[int]
        BOOL_FALSE_TRUE_FIELD_NUMBER: _ClassVar[int]
        DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
        kind: ModelSequenceBatching.Control.Kind
        int32_false_true: _containers.RepeatedScalarFieldContainer[int]
        fp32_false_true: _containers.RepeatedScalarFieldContainer[float]
        bool_false_true: _containers.RepeatedScalarFieldContainer[bool]
        data_type: DataType
        def __init__(self, kind: _Optional[_Union[ModelSequenceBatching.Control.Kind, str]] = ..., int32_false_true: _Optional[_Iterable[int]] = ..., fp32_false_true: _Optional[_Iterable[float]] = ..., bool_false_true: _Optional[_Iterable[bool]] = ..., data_type: _Optional[_Union[DataType, str]] = ...) -> None: ...
    class ControlInput(_message.Message):
        __slots__ = ("name", "control")
        NAME_FIELD_NUMBER: _ClassVar[int]
        CONTROL_FIELD_NUMBER: _ClassVar[int]
        name: str
        control: _containers.RepeatedCompositeFieldContainer[ModelSequenceBatching.Control]
        def __init__(self, name: _Optional[str] = ..., control: _Optional[_Iterable[_Union[ModelSequenceBatching.Control, _Mapping]]] = ...) -> None: ...
    class InitialState(_message.Message):
        __slots__ = ("data_type", "dims", "zero_data", "data_file", "name")
        DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
        DIMS_FIELD_NUMBER: _ClassVar[int]
        ZERO_DATA_FIELD_NUMBER: _ClassVar[int]
        DATA_FILE_FIELD_NUMBER: _ClassVar[int]
        NAME_FIELD_NUMBER: _ClassVar[int]
        data_type: DataType
        dims: _containers.RepeatedScalarFieldContainer[int]
        zero_data: bool
        data_file: str
        name: str
        def __init__(self, data_type: _Optional[_Union[DataType, str]] = ..., dims: _Optional[_Iterable[int]] = ..., zero_data: _Optional[bool] = ..., data_file: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...
    class State(_message.Message):
        __slots__ = ("input_name", "output_name", "data_type", "dims", "initial_state", "use_same_buffer_for_input_output", "use_growable_memory")
        INPUT_NAME_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_NAME_FIELD_NUMBER: _ClassVar[int]
        DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
        DIMS_FIELD_NUMBER: _ClassVar[int]
        INITIAL_STATE_FIELD_NUMBER: _ClassVar[int]
        USE_SAME_BUFFER_FOR_INPUT_OUTPUT_FIELD_NUMBER: _ClassVar[int]
        USE_GROWABLE_MEMORY_FIELD_NUMBER: _ClassVar[int]
        input_name: str
        output_name: str
        data_type: DataType
        dims: _containers.RepeatedScalarFieldContainer[int]
        initial_state: _containers.RepeatedCompositeFieldContainer[ModelSequenceBatching.InitialState]
        use_same_buffer_for_input_output: bool
        use_growable_memory: bool
        def __init__(self, input_name: _Optional[str] = ..., output_name: _Optional[str] = ..., data_type: _Optional[_Union[DataType, str]] = ..., dims: _Optional[_Iterable[int]] = ..., initial_state: _Optional[_Iterable[_Union[ModelSequenceBatching.InitialState, _Mapping]]] = ..., use_same_buffer_for_input_output: _Optional[bool] = ..., use_growable_memory: _Optional[bool] = ...) -> None: ...
    class StrategyDirect(_message.Message):
        __slots__ = ("max_queue_delay_microseconds", "minimum_slot_utilization")
        MAX_QUEUE_DELAY_MICROSECONDS_FIELD_NUMBER: _ClassVar[int]
        MINIMUM_SLOT_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
        max_queue_delay_microseconds: int
        minimum_slot_utilization: float
        def __init__(self, max_queue_delay_microseconds: _Optional[int] = ..., minimum_slot_utilization: _Optional[float] = ...) -> None: ...
    class StrategyOldest(_message.Message):
        __slots__ = ("max_candidate_sequences", "preferred_batch_size", "max_queue_delay_microseconds", "preserve_ordering")
        MAX_CANDIDATE_SEQUENCES_FIELD_NUMBER: _ClassVar[int]
        PREFERRED_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
        MAX_QUEUE_DELAY_MICROSECONDS_FIELD_NUMBER: _ClassVar[int]
        PRESERVE_ORDERING_FIELD_NUMBER: _ClassVar[int]
        max_candidate_sequences: int
        preferred_batch_size: _containers.RepeatedScalarFieldContainer[int]
        max_queue_delay_microseconds: int
        preserve_ordering: bool
        def __init__(self, max_candidate_sequences: _Optional[int] = ..., preferred_batch_size: _Optional[_Iterable[int]] = ..., max_queue_delay_microseconds: _Optional[int] = ..., preserve_ordering: _Optional[bool] = ...) -> None: ...
    DIRECT_FIELD_NUMBER: _ClassVar[int]
    OLDEST_FIELD_NUMBER: _ClassVar[int]
    MAX_SEQUENCE_IDLE_MICROSECONDS_FIELD_NUMBER: _ClassVar[int]
    CONTROL_INPUT_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ITERATIVE_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    direct: ModelSequenceBatching.StrategyDirect
    oldest: ModelSequenceBatching.StrategyOldest
    max_sequence_idle_microseconds: int
    control_input: _containers.RepeatedCompositeFieldContainer[ModelSequenceBatching.ControlInput]
    state: _containers.RepeatedCompositeFieldContainer[ModelSequenceBatching.State]
    iterative_sequence: bool
    def __init__(self, direct: _Optional[_Union[ModelSequenceBatching.StrategyDirect, _Mapping]] = ..., oldest: _Optional[_Union[ModelSequenceBatching.StrategyOldest, _Mapping]] = ..., max_sequence_idle_microseconds: _Optional[int] = ..., control_input: _Optional[_Iterable[_Union[ModelSequenceBatching.ControlInput, _Mapping]]] = ..., state: _Optional[_Iterable[_Union[ModelSequenceBatching.State, _Mapping]]] = ..., iterative_sequence: _Optional[bool] = ...) -> None: ...

class ModelEnsembling(_message.Message):
    __slots__ = ("step",)
    class Step(_message.Message):
        __slots__ = ("model_name", "model_version", "input_map", "output_map", "model_namespace")
        class InputMapEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        class OutputMapEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
        MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
        INPUT_MAP_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_MAP_FIELD_NUMBER: _ClassVar[int]
        MODEL_NAMESPACE_FIELD_NUMBER: _ClassVar[int]
        model_name: str
        model_version: int
        input_map: _containers.ScalarMap[str, str]
        output_map: _containers.ScalarMap[str, str]
        model_namespace: str
        def __init__(self, model_name: _Optional[str] = ..., model_version: _Optional[int] = ..., input_map: _Optional[_Mapping[str, str]] = ..., output_map: _Optional[_Mapping[str, str]] = ..., model_namespace: _Optional[str] = ...) -> None: ...
    STEP_FIELD_NUMBER: _ClassVar[int]
    step: _containers.RepeatedCompositeFieldContainer[ModelEnsembling.Step]
    def __init__(self, step: _Optional[_Iterable[_Union[ModelEnsembling.Step, _Mapping]]] = ...) -> None: ...

class ModelParameter(_message.Message):
    __slots__ = ("string_value",)
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    string_value: str
    def __init__(self, string_value: _Optional[str] = ...) -> None: ...

class ModelWarmup(_message.Message):
    __slots__ = ("name", "batch_size", "inputs", "count")
    class Input(_message.Message):
        __slots__ = ("data_type", "dims", "zero_data", "random_data", "input_data_file")
        DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
        DIMS_FIELD_NUMBER: _ClassVar[int]
        ZERO_DATA_FIELD_NUMBER: _ClassVar[int]
        RANDOM_DATA_FIELD_NUMBER: _ClassVar[int]
        INPUT_DATA_FILE_FIELD_NUMBER: _ClassVar[int]
        data_type: DataType
        dims: _containers.RepeatedScalarFieldContainer[int]
        zero_data: bool
        random_data: bool
        input_data_file: str
        def __init__(self, data_type: _Optional[_Union[DataType, str]] = ..., dims: _Optional[_Iterable[int]] = ..., zero_data: _Optional[bool] = ..., random_data: _Optional[bool] = ..., input_data_file: _Optional[str] = ...) -> None: ...
    class InputsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ModelWarmup.Input
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ModelWarmup.Input, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    name: str
    batch_size: int
    inputs: _containers.MessageMap[str, ModelWarmup.Input]
    count: int
    def __init__(self, name: _Optional[str] = ..., batch_size: _Optional[int] = ..., inputs: _Optional[_Mapping[str, ModelWarmup.Input]] = ..., count: _Optional[int] = ...) -> None: ...

class ModelOperations(_message.Message):
    __slots__ = ("op_library_filename",)
    OP_LIBRARY_FILENAME_FIELD_NUMBER: _ClassVar[int]
    op_library_filename: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, op_library_filename: _Optional[_Iterable[str]] = ...) -> None: ...

class ModelTransactionPolicy(_message.Message):
    __slots__ = ("decoupled",)
    DECOUPLED_FIELD_NUMBER: _ClassVar[int]
    decoupled: bool
    def __init__(self, decoupled: _Optional[bool] = ...) -> None: ...

class ModelRepositoryAgents(_message.Message):
    __slots__ = ("agents",)
    class Agent(_message.Message):
        __slots__ = ("name", "parameters")
        class ParametersEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        NAME_FIELD_NUMBER: _ClassVar[int]
        PARAMETERS_FIELD_NUMBER: _ClassVar[int]
        name: str
        parameters: _containers.ScalarMap[str, str]
        def __init__(self, name: _Optional[str] = ..., parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...
    AGENTS_FIELD_NUMBER: _ClassVar[int]
    agents: _containers.RepeatedCompositeFieldContainer[ModelRepositoryAgents.Agent]
    def __init__(self, agents: _Optional[_Iterable[_Union[ModelRepositoryAgents.Agent, _Mapping]]] = ...) -> None: ...

class ModelResponseCache(_message.Message):
    __slots__ = ("enable",)
    ENABLE_FIELD_NUMBER: _ClassVar[int]
    enable: bool
    def __init__(self, enable: _Optional[bool] = ...) -> None: ...

class ModelMetrics(_message.Message):
    __slots__ = ("metric_control",)
    class MetricControl(_message.Message):
        __slots__ = ("metric_identifier", "histogram_options")
        class MetricIdentifier(_message.Message):
            __slots__ = ("family",)
            FAMILY_FIELD_NUMBER: _ClassVar[int]
            family: str
            def __init__(self, family: _Optional[str] = ...) -> None: ...
        class HistogramOptions(_message.Message):
            __slots__ = ("buckets",)
            BUCKETS_FIELD_NUMBER: _ClassVar[int]
            buckets: _containers.RepeatedScalarFieldContainer[float]
            def __init__(self, buckets: _Optional[_Iterable[float]] = ...) -> None: ...
        METRIC_IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
        HISTOGRAM_OPTIONS_FIELD_NUMBER: _ClassVar[int]
        metric_identifier: ModelMetrics.MetricControl.MetricIdentifier
        histogram_options: ModelMetrics.MetricControl.HistogramOptions
        def __init__(self, metric_identifier: _Optional[_Union[ModelMetrics.MetricControl.MetricIdentifier, _Mapping]] = ..., histogram_options: _Optional[_Union[ModelMetrics.MetricControl.HistogramOptions, _Mapping]] = ...) -> None: ...
    METRIC_CONTROL_FIELD_NUMBER: _ClassVar[int]
    metric_control: _containers.RepeatedCompositeFieldContainer[ModelMetrics.MetricControl]
    def __init__(self, metric_control: _Optional[_Iterable[_Union[ModelMetrics.MetricControl, _Mapping]]] = ...) -> None: ...

class ModelConfig(_message.Message):
    __slots__ = ("name", "platform", "backend", "runtime", "version_policy", "max_batch_size", "input", "output", "batch_input", "batch_output", "optimization", "dynamic_batching", "sequence_batching", "ensemble_scheduling", "instance_group", "default_model_filename", "cc_model_filenames", "metric_tags", "parameters", "model_warmup", "model_operations", "model_transaction_policy", "model_repository_agents", "response_cache", "model_metrics")
    class CcModelFilenamesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class MetricTagsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ModelParameter
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ModelParameter, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    VERSION_POLICY_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    BATCH_INPUT_FIELD_NUMBER: _ClassVar[int]
    BATCH_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZATION_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_BATCHING_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_BATCHING_FIELD_NUMBER: _ClassVar[int]
    ENSEMBLE_SCHEDULING_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_GROUP_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_MODEL_FILENAME_FIELD_NUMBER: _ClassVar[int]
    CC_MODEL_FILENAMES_FIELD_NUMBER: _ClassVar[int]
    METRIC_TAGS_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    MODEL_WARMUP_FIELD_NUMBER: _ClassVar[int]
    MODEL_OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    MODEL_TRANSACTION_POLICY_FIELD_NUMBER: _ClassVar[int]
    MODEL_REPOSITORY_AGENTS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_CACHE_FIELD_NUMBER: _ClassVar[int]
    MODEL_METRICS_FIELD_NUMBER: _ClassVar[int]
    name: str
    platform: str
    backend: str
    runtime: str
    version_policy: ModelVersionPolicy
    max_batch_size: int
    input: _containers.RepeatedCompositeFieldContainer[ModelInput]
    output: _containers.RepeatedCompositeFieldContainer[ModelOutput]
    batch_input: _containers.RepeatedCompositeFieldContainer[BatchInput]
    batch_output: _containers.RepeatedCompositeFieldContainer[BatchOutput]
    optimization: ModelOptimizationPolicy
    dynamic_batching: ModelDynamicBatching
    sequence_batching: ModelSequenceBatching
    ensemble_scheduling: ModelEnsembling
    instance_group: _containers.RepeatedCompositeFieldContainer[ModelInstanceGroup]
    default_model_filename: str
    cc_model_filenames: _containers.ScalarMap[str, str]
    metric_tags: _containers.ScalarMap[str, str]
    parameters: _containers.MessageMap[str, ModelParameter]
    model_warmup: _containers.RepeatedCompositeFieldContainer[ModelWarmup]
    model_operations: ModelOperations
    model_transaction_policy: ModelTransactionPolicy
    model_repository_agents: ModelRepositoryAgents
    response_cache: ModelResponseCache
    model_metrics: ModelMetrics
    def __init__(self, name: _Optional[str] = ..., platform: _Optional[str] = ..., backend: _Optional[str] = ..., runtime: _Optional[str] = ..., version_policy: _Optional[_Union[ModelVersionPolicy, _Mapping]] = ..., max_batch_size: _Optional[int] = ..., input: _Optional[_Iterable[_Union[ModelInput, _Mapping]]] = ..., output: _Optional[_Iterable[_Union[ModelOutput, _Mapping]]] = ..., batch_input: _Optional[_Iterable[_Union[BatchInput, _Mapping]]] = ..., batch_output: _Optional[_Iterable[_Union[BatchOutput, _Mapping]]] = ..., optimization: _Optional[_Union[ModelOptimizationPolicy, _Mapping]] = ..., dynamic_batching: _Optional[_Union[ModelDynamicBatching, _Mapping]] = ..., sequence_batching: _Optional[_Union[ModelSequenceBatching, _Mapping]] = ..., ensemble_scheduling: _Optional[_Union[ModelEnsembling, _Mapping]] = ..., instance_group: _Optional[_Iterable[_Union[ModelInstanceGroup, _Mapping]]] = ..., default_model_filename: _Optional[str] = ..., cc_model_filenames: _Optional[_Mapping[str, str]] = ..., metric_tags: _Optional[_Mapping[str, str]] = ..., parameters: _Optional[_Mapping[str, ModelParameter]] = ..., model_warmup: _Optional[_Iterable[_Union[ModelWarmup, _Mapping]]] = ..., model_operations: _Optional[_Union[ModelOperations, _Mapping]] = ..., model_transaction_policy: _Optional[_Union[ModelTransactionPolicy, _Mapping]] = ..., model_repository_agents: _Optional[_Union[ModelRepositoryAgents, _Mapping]] = ..., response_cache: _Optional[_Union[ModelResponseCache, _Mapping]] = ..., model_metrics: _Optional[_Union[ModelMetrics, _Mapping]] = ...) -> None: ...

"""Well-known tensor names per backend. Used for backend auto-detection and adapters."""

TEXT_INPUT = "text_input"
TEXT_OUTPUT = "text_output"

TRTLLM_INPUTS: frozenset[str] = frozenset(
    {
        TEXT_INPUT,
        "max_tokens",
        "stream",
        "temperature",
        "top_k",
        "top_p",
        "stop_words",
        "bad_words",
        "end_id",
        "pad_id",
        "return_log_probs",
        "random_seed",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "beam_width",
        "length_penalty",
        "min_length",
    }
)
TRTLLM_OUTPUTS: frozenset[str] = frozenset({TEXT_OUTPUT, "cum_log_probs", "output_log_probs"})

VLLM_INPUTS: frozenset[str] = frozenset({TEXT_INPUT, "stream", "sampling_parameters", "exclude_input_in_output"})
VLLM_OUTPUTS: frozenset[str] = frozenset({TEXT_OUTPUT})

FINAL_RESPONSE_PARAM = "triton_final_response"

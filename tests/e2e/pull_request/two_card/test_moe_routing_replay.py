import os
from unittest.mock import patch

import numpy as np
import pytest
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from tests.e2e.conftest import VllmRunner

MODELS = [
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3-30B-A3B",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("async_scheduling", [False, True])
@patch.dict(os.environ, {"OMP_NUM_THREADS": "1"})
def test_qwen3_moe_routing_replay(model, async_scheduling):
    prompts = [
        "Hello, please introduce yourself.",
        "What is two plus three? Explain briefly.",
    ]
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        distributed_executor_backend="mp",
        enable_return_routed_experts=True,
        async_scheduling=async_scheduling,
        enable_prefix_caching=False,
    ) as vllm_model:
        sampling_params = SamplingParams(
            max_tokens=5, temperature=0.8, top_p=0.95, output_kind=RequestOutputKind.FINAL_ONLY
        )
        inputs = vllm_model.get_inputs(prompts=prompts)
        outputs = vllm_model.model.generate(prompts=inputs, sampling_params=sampling_params)
        hf_config = vllm_model.model.llm_engine.vllm_config.model_config.hf_text_config
        for output in outputs:
            assert output.finished
            result = output.outputs[0]
            routes = result.routed_experts
            assert routes.shape == (
                len(output.prompt_token_ids) + len(result.token_ids) - 1,
                hf_config.num_hidden_layers,
                hf_config.num_experts_per_tok,
            )
            assert ((routes >= 0) & (routes < hf_config.num_experts)).all()
            assert (np.diff(np.sort(routes, axis=-1), axis=-1) != 0).all()

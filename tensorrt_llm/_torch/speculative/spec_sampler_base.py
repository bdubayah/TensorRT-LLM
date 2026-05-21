# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base class for speculative decoding samplers.

This module provides a common base class for MTPSampler, SASampler, and
Eagle3OneModelSampler.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import torch

from ...executor.result import Logprob
from ..pyexecutor.llm_request import FLAT_LOGPROBS_MODE, LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.sampler import (
    DEFAULT_BEAM_IDX,
    AsyncWorkerMixin,
    Sampler,
    SampleState,
    SampleStateTensors,
    TorchSampler,
    add_token,
    int_tensor,
)
from ..pyexecutor.scheduler import ScheduledRequests

_DYNAMIC_TEMPERATURE_PAD_TOKEN = -1


@dataclass
class _DynamicTemperatureCapBuffers:
    max_requests: int
    max_rule_count: int
    max_rule_len: int
    max_steps: int
    device: torch.device
    token_dtype: torch.dtype
    request_indices_host: torch.Tensor
    suffix_host: torch.Tensor
    rule_host: torch.Tensor
    rule_lens_host: torch.Tensor
    request_indices: torch.Tensor
    suffix: torch.Tensor
    rule: torch.Tensor
    rule_lens: torch.Tensor
    step_offsets: torch.Tensor
    rule_offsets: torch.Tensor

    @classmethod
    def create(
        cls,
        max_requests: int,
        max_rule_count: int,
        max_rule_len: int,
        max_steps: int,
        device: torch.device,
        token_dtype: torch.dtype,
    ) -> "_DynamicTemperatureCapBuffers":
        pin_host = device.type == "cuda"
        return cls(
            max_requests=max_requests,
            max_rule_count=max_rule_count,
            max_rule_len=max_rule_len,
            max_steps=max_steps,
            device=device,
            token_dtype=token_dtype,
            request_indices_host=torch.empty(
                max_requests, dtype=torch.long, device="cpu", pin_memory=pin_host
            ),
            suffix_host=torch.empty(
                (max_requests, max_rule_len),
                dtype=token_dtype,
                device="cpu",
                pin_memory=pin_host,
            ),
            rule_host=torch.empty(
                (max_requests, max_rule_count, max_rule_len),
                dtype=token_dtype,
                device="cpu",
                pin_memory=pin_host,
            ),
            rule_lens_host=torch.empty(
                (max_requests, max_rule_count),
                dtype=torch.long,
                device="cpu",
                pin_memory=pin_host,
            ),
            request_indices=torch.empty(max_requests, dtype=torch.long, device=device),
            suffix=torch.empty((max_requests, max_rule_len), dtype=token_dtype, device=device),
            rule=torch.empty(
                (max_requests, max_rule_count, max_rule_len),
                dtype=token_dtype,
                device=device,
            ),
            rule_lens=torch.empty((max_requests, max_rule_count), dtype=torch.long, device=device),
            step_offsets=torch.arange(max_steps, dtype=torch.long, device=device),
            rule_offsets=torch.arange(max_rule_len, dtype=torch.long, device=device),
        )

    def ensure(
        self,
        max_requests: int,
        max_rule_count: int,
        max_rule_len: int,
        max_steps: int,
        device: torch.device,
        token_dtype: torch.dtype,
    ) -> "_DynamicTemperatureCapBuffers":
        if (
            max_requests <= self.max_requests
            and max_rule_count <= self.max_rule_count
            and max_rule_len <= self.max_rule_len
            and max_steps <= self.max_steps
            and device == self.device
            and token_dtype == self.token_dtype
        ):
            return self
        updated = self.create(
            max(max_requests, self.max_requests),
            max(max_rule_count, self.max_rule_count),
            max(max_rule_len, self.max_rule_len),
            max(max_steps, self.max_steps),
            device,
            token_dtype,
        )
        self.__dict__.update(updated.__dict__)
        return self


def _find_dynamic_temperature_trigger(
    request: LlmRequest, new_tokens: Iterable[int], num_new_tokens: int
) -> tuple[int, Optional[float]]:
    dynamic_temperature_rules = request.py_dynamic_temperature_rules
    if not dynamic_temperature_rules:
        return num_new_tokens, None

    suffix_tokens = []
    if request.py_dynamic_temperature_suffix_tokens is not None:
        suffix_tokens = list(request.py_dynamic_temperature_suffix_tokens[DEFAULT_BEAM_IDX])
    max_rule_len = request.py_dynamic_temperature_max_rule_len
    if max_rule_len == 0:
        max_rule_len = max(len(token_ids) for token_ids, _ in dynamic_temperature_rules)

    for step, new_token in enumerate(new_tokens):
        if step == num_new_tokens:
            break
        suffix_tokens.append(new_token)
        if len(suffix_tokens) > max_rule_len:
            del suffix_tokens[:-max_rule_len]
        for token_ids, temperature in dynamic_temperature_rules:
            if len(token_ids) > len(suffix_tokens):
                continue
            if suffix_tokens[-len(token_ids) :] == token_ids:
                if request.py_dynamic_temperature_override != temperature:
                    return step + 1, temperature
                break
    return num_new_tokens, None


def _prepare_dynamic_temperature_cap_metadata(
    requests: list[LlmRequest],
    device: torch.device,
    token_dtype: torch.dtype,
    num_steps: int,
    buffers: _DynamicTemperatureCapBuffers,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
):
    request_indices = []
    active_rules_by_request = []
    max_rule_count = 0
    max_rule_len = 0
    for req_idx, request in enumerate(requests):
        if not request.py_dynamic_temperature_rules:
            continue
        if request.state == LlmRequestState.GENERATION_COMPLETE:
            continue
        if getattr(request, "is_attention_dp_dummy", False):
            continue
        active_rules = [
            token_ids
            for token_ids, temperature in request.py_dynamic_temperature_rules
            if request.py_dynamic_temperature_override != temperature
        ]
        if not active_rules:
            continue
        request_max_rule_len = request.py_dynamic_temperature_max_rule_len
        if request_max_rule_len == 0:
            request_max_rule_len = max(len(token_ids) for token_ids in active_rules)
        request_indices.append(req_idx)
        active_rules_by_request.append(active_rules)
        max_rule_count = max(max_rule_count, len(active_rules))
        max_rule_len = max(max_rule_len, request_max_rule_len)

    if not request_indices:
        return None

    num_requests = len(request_indices)
    buffers.ensure(num_requests, max_rule_count, max_rule_len, num_steps, device, token_dtype)
    request_indices_host = buffers.request_indices_host[:num_requests]
    suffix_host = buffers.suffix_host[:num_requests, :max_rule_len]
    rule_host = buffers.rule_host[:num_requests, :max_rule_count, :max_rule_len]
    rule_lens_host = buffers.rule_lens_host[:num_requests, :max_rule_count]

    suffix_host.fill_(_DYNAMIC_TEMPERATURE_PAD_TOKEN)
    rule_host.fill_(_DYNAMIC_TEMPERATURE_PAD_TOKEN)
    rule_lens_host.zero_()

    for local_idx, (req_idx, active_rules) in enumerate(
        zip(request_indices, active_rules_by_request)
    ):
        request_indices_host[local_idx] = req_idx

        request = requests[req_idx]
        suffix_tokens = []
        if request.py_dynamic_temperature_suffix_tokens is not None:
            suffix_tokens = list(request.py_dynamic_temperature_suffix_tokens[DEFAULT_BEAM_IDX])
        suffix_tokens = suffix_tokens[-max_rule_len:]
        suffix_start = max_rule_len - len(suffix_tokens)
        for token_idx, token_id in enumerate(suffix_tokens, start=suffix_start):
            suffix_host[local_idx, token_idx] = token_id

        for rule_idx, token_ids in enumerate(active_rules):
            rule_lens_host[local_idx, rule_idx] = len(token_ids)
            for token_idx, token_id in enumerate(token_ids):
                rule_host[local_idx, rule_idx, token_idx] = token_id

    request_indices_tensor = buffers.request_indices[:num_requests]
    suffix_tensor = buffers.suffix[:num_requests, :max_rule_len]
    rule_tensor = buffers.rule[:num_requests, :max_rule_count, :max_rule_len]
    rule_lens_tensor = buffers.rule_lens[:num_requests, :max_rule_count]
    request_indices_tensor.copy_(request_indices_host, non_blocking=True)
    suffix_tensor.copy_(suffix_host, non_blocking=True)
    rule_tensor.copy_(rule_host, non_blocking=True)
    rule_lens_tensor.copy_(rule_lens_host, non_blocking=True)

    return (
        request_indices_tensor,
        suffix_tensor,
        rule_tensor,
        rule_lens_tensor,
        buffers.step_offsets[:num_steps],
        buffers.rule_offsets[:max_rule_len],
    )


@torch.compile(options={"max-autotune": True})
def _cap_dynamic_temperature_tensors(
    new_tokens: torch.Tensor,
    new_tokens_lens: torch.Tensor,
    request_indices: torch.Tensor,
    suffix_tensor: torch.Tensor,
    rule_tensor: torch.Tensor,
    rule_lens: torch.Tensor,
    step_offsets: torch.Tensor,
    rule_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_new_tokens = new_tokens.index_select(0, request_indices)
    selected_new_tokens_lens = new_tokens_lens.index_select(0, request_indices).long()
    combined_tokens = torch.cat((suffix_tensor, selected_new_tokens), dim=1)

    num_steps = new_tokens.shape[1]
    num_requests = rule_lens.shape[0]
    max_rule_count = rule_lens.shape[1]
    max_rule_len = rule_tensor.shape[2]
    match_starts = (
        max_rule_len
        + step_offsets.view(1, num_steps, 1)
        - rule_lens.view(num_requests, 1, max_rule_count)
        + 1
    )
    match_positions = match_starts.unsqueeze(-1) + rule_offsets.view(1, 1, 1, max_rule_len)
    match_positions = match_positions.clamp(0, combined_tokens.shape[1] - 1)
    windows = combined_tokens.view(num_requests, 1, 1, -1).expand(
        num_requests, num_steps, max_rule_count, -1
    )
    windows = windows.gather(3, match_positions)

    rule_token_mask = rule_offsets.view(1, 1, 1, max_rule_len) < rule_lens.view(
        num_requests, 1, max_rule_count, 1
    )
    rule_matches = torch.logical_or(
        windows == rule_tensor.view(num_requests, 1, max_rule_count, max_rule_len),
        ~rule_token_mask,
    ).all(dim=-1)
    rule_matches = torch.logical_and(
        rule_matches, rule_lens.view(num_requests, 1, max_rule_count) > 0
    )
    step_mask = step_offsets.view(1, num_steps, 1) < selected_new_tokens_lens.view(
        num_requests, 1, 1
    )
    trigger_by_step = torch.logical_and(rule_matches, step_mask).any(dim=-1)
    candidate_lens = torch.where(
        trigger_by_step,
        step_offsets.view(1, num_steps) + 1,
        selected_new_tokens_lens.view(num_requests, 1),
    )
    trigger_lens = candidate_lens.min(dim=1).values
    capped_lens = torch.minimum(selected_new_tokens_lens, trigger_lens)
    anchor_tokens = selected_new_tokens.gather(
        1, (capped_lens - 1).clamp_min(0).view(num_requests, 1)
    ).squeeze(1)

    return capped_lens, anchor_tokens


def _cap_dynamic_temperature_outputs_on_device(
    requests: list[LlmRequest],
    new_tokens: torch.Tensor,
    new_tokens_lens: torch.Tensor,
    next_new_tokens: torch.Tensor,
    buffers: _DynamicTemperatureCapBuffers,
) -> None:
    cap_metadata = _prepare_dynamic_temperature_cap_metadata(
        requests,
        device=new_tokens.device,
        token_dtype=new_tokens.dtype,
        num_steps=new_tokens.shape[1],
        buffers=buffers,
    )
    if cap_metadata is None:
        return

    request_indices, suffix_tensor, rule_tensor, rule_lens, step_offsets, rule_offsets = (
        cap_metadata
    )
    capped_lens, anchor_tokens = _cap_dynamic_temperature_tensors(
        new_tokens,
        new_tokens_lens,
        request_indices,
        suffix_tensor,
        rule_tensor,
        rule_lens,
        step_offsets,
        rule_offsets,
    )

    with torch.inference_mode():
        new_tokens_lens.index_copy_(0, request_indices, capped_lens.to(new_tokens_lens.dtype))
        next_new_tokens[:, 0].index_copy_(0, request_indices, anchor_tokens)


def _host_new_token(state: "SampleStateSpec", seq_slot: int, step: int) -> int:
    assert state.host is not None
    return int(state.host.new_tokens[step, seq_slot, DEFAULT_BEAM_IDX].item())


def _cap_dynamic_temperature_sample_state(state: "SampleStateSpec") -> None:
    if getattr(state, "_dynamic_temperature_finalized", False):
        return
    setattr(state, "_dynamic_temperature_finalized", True)

    if not any(request.py_dynamic_temperature_rules for request in state.requests):
        return

    assert state.device is not None
    assert state.host is not None

    capped_seq_slots = []
    capped_num_new_tokens_list = []
    capped_anchor_tokens = []
    for request in state.requests:
        if not request.py_dynamic_temperature_rules:
            continue
        if request.state == LlmRequestState.GENERATION_COMPLETE:
            continue
        if getattr(request, "is_attention_dp_dummy", False):
            continue

        seq_slot = request.py_seq_slot
        num_new_tokens = int(state.host.new_tokens_lens[seq_slot].item())
        capped_num_new_tokens, temperature = _find_dynamic_temperature_trigger(
            request,
            (_host_new_token(state, seq_slot, step) for step in range(num_new_tokens)),
            num_new_tokens,
        )
        if temperature is None:
            continue
        request.py_dynamic_temperature_override = temperature
        request.py_sampling_strategy = None
        if capped_num_new_tokens == num_new_tokens:
            continue
        capped_seq_slots.append(seq_slot)
        capped_num_new_tokens_list.append(capped_num_new_tokens)
        capped_anchor_tokens.append(_host_new_token(state, seq_slot, capped_num_new_tokens - 1))
    if capped_seq_slots:
        with torch.inference_mode():
            for seq_slot, capped_num_new_tokens in zip(
                capped_seq_slots, capped_num_new_tokens_list
            ):
                state.host.new_tokens_lens[seq_slot] = capped_num_new_tokens

            capped_indices = torch.tensor(
                capped_seq_slots, dtype=torch.long, device=state.device.new_tokens_lens.device
            )
            capped_lens = torch.tensor(
                capped_num_new_tokens_list,
                dtype=state.device.new_tokens_lens.dtype,
                device=state.device.new_tokens_lens.device,
            )
            state.device.new_tokens_lens.index_copy_(0, capped_indices, capped_lens)
            anchor_tokens = torch.tensor(
                capped_anchor_tokens,
                dtype=state.device.new_tokens.dtype,
                device=state.device.new_tokens.device,
            )
            state.device.new_tokens[0, :, DEFAULT_BEAM_IDX].index_copy_(
                0, capped_indices, anchor_tokens
            )


@dataclass(kw_only=True)
class SampleStateTensorsSpec(SampleStateTensors):
    """Tensors for speculative decoding sample state."""

    new_tokens_lens: torch.Tensor
    next_draft_tokens: torch.Tensor


@dataclass(kw_only=True)
class SampleStateSpec(SampleState):
    """Sample state for speculative decoding."""

    device: SampleStateTensorsSpec
    host: SampleStateTensorsSpec


class SpecSamplerBase(Sampler[SampleStateSpec], AsyncWorkerMixin):
    """
    Base class for speculative decoding samplers (MTP, NGram, Eagle3, SA).

    Provides common functionality:
    - Pre-allocated GPU storage buffers
    - Async GPU->CPU copy in sample_async
    - Request state updates in update_requests

    Subclasses can customize behavior by overriding:
    - _get_max_tokens(): How to calculate max_tokens for storage
    - _get_draft_tokens_storage_size(): Size of next_draft_tokens tensor
    - _add_dummy_draft_tokens(): Whether to add dummy drafts for context requests
    """

    SampleState = SampleStateSpec

    def is_generation_model(self) -> bool:
        return True

    @dataclass(kw_only=True)
    class Store:
        """Storage for speculative decoding tensors."""

        new_tokens: torch.Tensor
        next_new_tokens: torch.Tensor
        next_draft_tokens: torch.Tensor
        new_tokens_lens: torch.Tensor

    def __init__(self, args: TorchSampler.Args, *, draft_len: int):
        """
        Initialize the speculative sampler.

        Args:
            args: TorchSampler.Args with max_num_sequences, max_seq_len, etc.
            draft_len: Maximum number of draft tokens per iteration.
        """
        self.mapping = None
        self.draft_len = draft_len
        self.max_seq_len = args.max_seq_len

        seq_slots = args.max_num_sequences
        max_tokens = self._get_max_tokens(args, draft_len)
        draft_tokens_size = self._get_draft_tokens_storage_size(args, draft_len)
        self.max_beam_width = args.max_beam_width
        assert self.max_beam_width == 1, "beam width must be 1 for speculative decoding"

        self.store = self.Store(
            new_tokens=int_tensor((max_tokens, seq_slots, self.max_beam_width)),
            next_new_tokens=int_tensor((max_tokens, seq_slots, self.max_beam_width)),
            next_draft_tokens=int_tensor((seq_slots, draft_tokens_size)),
            new_tokens_lens=int_tensor((seq_slots,)),
        )
        self._dynamic_temperature_cap_buffers = _DynamicTemperatureCapBuffers.create(
            max_requests=seq_slots,
            max_rule_count=1,
            max_rule_len=1,
            max_steps=max_tokens,
            device=self.store.new_tokens.device,
            token_dtype=self.store.new_tokens.dtype,
        )

    def _get_max_tokens(self, args: TorchSampler.Args, draft_len: int) -> int:
        """
        Calculate max_tokens for storage allocation.

        Override in subclasses if needed. Default: draft_len + 1.
        MTP uses args.max_total_draft_tokens + 1 for tree-based speculation.
        """
        return draft_len + 1

    def _get_draft_tokens_storage_size(self, args: TorchSampler.Args, draft_len: int) -> int:
        """
        Calculate storage size for next_draft_tokens tensor.

        Override in subclasses if needed. Default: draft_len.
        MTP uses args.max_total_draft_tokens for tree-based speculation.
        """
        return draft_len

    def _add_dummy_draft_tokens(self) -> bool:
        """
        Whether to add dummy draft tokens for context requests.

        Override in subclasses. Default: True (needed for KV cache preparation).
        """
        return True

    def validate_request(self, request: LlmRequest) -> None:
        if request.py_return_log_probs and (request.py_num_logprobs or 0) > 1:
            raise ValueError(
                "Speculative sampler only supports returning the sampled logprob per token"
            )

    def _request_common_handling(
        self,
        request: LlmRequest,
        next_draft_tokens: list[list[int]],
        runtime_draft_len: Optional[int],
    ) -> None:
        """Common handling for both context and generation requests."""
        assert not request.py_return_context_logits, (
            "return_context_logits not implemented for speculative sampler"
        )
        assert not request.py_return_generation_logits, (
            "return_generation_logits not implemented for speculative sampler"
        )
        request.py_draft_tokens = next_draft_tokens[request.py_seq_slot][:runtime_draft_len]
        request.py_decoding_iter += 1

    def update_requests(
        self,
        state: SampleStateSpec,
        resource_manager: Optional[BaseResourceManager] = None,
    ) -> None:
        """
        CPU-side request updates after GPU->CPU sync.

        Waits for async copy to complete, then updates request state with:
        - Accepted tokens
        - Stop criteria checks
        - Next iteration draft tokens
        """
        assert isinstance(state, SampleStateSpec)

        state.sampler_event.synchronize()
        self.finalize_sample_state_for_next_forward(state)
        new_tokens = state.host.new_tokens.tolist()
        new_tokens_lens_list = state.host.new_tokens_lens.tolist()
        next_draft_tokens_list = state.host.next_draft_tokens.tolist()
        raw_log_probs_list = None if state.host.log_probs is None else state.host.log_probs.tolist()
        beam_idx = DEFAULT_BEAM_IDX
        runtime_draft_len = getattr(state, "runtime_draft_len", self.draft_len)

        for req_idx, req in enumerate(state.requests):
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            if getattr(req, "is_attention_dp_dummy", False):
                continue
            num_new_tokens = new_tokens_lens_list[req.py_seq_slot]
            want_logprobs = req.py_return_log_probs and raw_log_probs_list is not None
            # Flat-mode fast path: collect (token_id, logprob_float) pairs in
            # the inner loop and do one batch append per request per step,
            # deferring Logprob/dict allocation. The original per-token path
            # (dict build + per-token append_log_probs) is retained below as
            # the fallback for when FLAT_LOGPROBS_MODE is disabled.
            use_flat = want_logprobs and FLAT_LOGPROBS_MODE
            flat_token_ids: list[int] = []
            flat_logprobs: list[float] = []
            req_logprob_row = raw_log_probs_list[req_idx] if want_logprobs else None
            for i in range(num_new_tokens):
                new_token = add_token(req, new_tokens, beam_idx=beam_idx, step=i)
                if want_logprobs:
                    if use_flat:
                        flat_token_ids.append(new_token)
                        flat_logprobs.append(req_logprob_row[i])
                    else:
                        req.py_result.append_log_probs(
                            [[{new_token: Logprob(logprob=req_logprob_row[i])}]]
                        )
                if TorchSampler._handle_stop_criteria(
                    req, new_token, max_seq_len=self.max_seq_len, beam_idx=beam_idx
                ):
                    break
            if use_flat and flat_token_ids:
                # Truncate in case _handle_stop_criteria broke the loop early
                # — flat_token_ids already reflects only the tokens that were
                # actually added, so this is a no-op most of the time.
                req.py_result.append_log_probs_flat(
                    flat_token_ids, flat_logprobs, beam_idx=beam_idx
                )
            req.py_num_accepted_draft_tokens = num_new_tokens - 1
            req.py_rewind_len = runtime_draft_len - req.py_num_accepted_draft_tokens
            self._request_common_handling(req, next_draft_tokens_list, runtime_draft_len)

    def finalize_sample_state_for_next_forward(self, state: SampleStateSpec) -> None:
        assert isinstance(state, SampleStateSpec)
        _cap_dynamic_temperature_sample_state(state)

    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        outputs: dict[str, torch.Tensor],
        num_context_logits_prefix_sum: list[int],
    ) -> SampleStateSpec:
        """
        Async sampling - schedules GPU->CPU copy.
        Called after CUDA graph replay.

        Args:
            scheduled_requests: Batch of scheduled requests
            outputs: Dict from worker forward() containing:
                - new_tokens: [batch, max_draft_len + 1] accepted tokens
                - new_tokens_lens: [batch] number of accepted tokens
                - next_draft_tokens: [batch, max_draft_len] draft tokens for next iter
                - next_new_tokens: [batch, max_draft_len + 1] input for next iter
            num_context_logits_prefix_sum: Prefix sum of context logits (unused)

        Returns:
            SampleStateSpec with device and host tensors
        """
        num_skip = len(scheduled_requests.context_requests_chunking)
        finished_context_requests = scheduled_requests.context_requests_last_chunk
        sampling_requests = finished_context_requests + scheduled_requests.generation_requests
        num_sampling_requests = len(sampling_requests)

        slots = torch.as_tensor([r.py_seq_slot for r in sampling_requests], dtype=torch.long)
        slots = slots.to(device="cuda", non_blocking=True)

        o_new_tokens = outputs["new_tokens"][num_skip : num_skip + num_sampling_requests]
        o_new_tokens_lens = outputs["new_tokens_lens"][num_skip : num_skip + num_sampling_requests]
        o_next_draft_tokens = outputs["next_draft_tokens"][
            num_skip : num_skip + num_sampling_requests
        ]
        o_next_new_tokens = outputs["next_new_tokens"][num_skip : num_skip + num_sampling_requests]
        runtime_draft_len = o_next_draft_tokens.shape[1]
        sampled_log_probs = outputs.get("sampled_log_probs")
        if sampled_log_probs is not None:
            sampled_log_probs = sampled_log_probs[num_skip : num_skip + num_sampling_requests]
        elif any(req.py_return_log_probs for req in sampling_requests):
            raise RuntimeError(
                "Speculative logprob requests require sampled_log_probs in worker outputs"
            )

        # Pad to match fixed-size store buffers for index_copy_.
        if o_new_tokens.shape[1] < (self.draft_len + 1):
            o_new_tokens = torch.nn.functional.pad(
                o_new_tokens, (0, (self.draft_len + 1) - o_new_tokens.shape[1])
            )
        if o_next_draft_tokens.shape[1] < self.draft_len:
            o_next_draft_tokens = torch.nn.functional.pad(
                o_next_draft_tokens, (0, self.draft_len - o_next_draft_tokens.shape[1])
            )
        if o_next_new_tokens.shape[1] < (self.draft_len + 1):
            o_next_new_tokens = torch.nn.functional.pad(
                o_next_new_tokens, (0, (self.draft_len + 1) - o_next_new_tokens.shape[1])
            )

        _cap_dynamic_temperature_outputs_on_device(
            sampling_requests,
            o_new_tokens,
            o_new_tokens_lens,
            o_next_new_tokens,
            self._dynamic_temperature_cap_buffers,
        )

        # Use index_copy_ for efficient copying (slots are unique)
        self.store.new_tokens.squeeze(-1).T.index_copy_(0, slots, o_new_tokens)
        self.store.next_new_tokens.squeeze(-1).T.index_copy_(0, slots, o_next_new_tokens)
        self.store.new_tokens_lens.index_copy_(0, slots, o_new_tokens_lens)
        self.store.next_draft_tokens.index_copy_(0, slots, o_next_draft_tokens)

        # Create sample state with async D2H copy
        device_tensors = SampleStateTensorsSpec(
            new_tokens=self.store.next_new_tokens,
            new_tokens_lens=self.store.new_tokens_lens,
            next_draft_tokens=self.store.next_draft_tokens,
            log_probs=sampled_log_probs,
        )

        host_tensors = SampleStateTensorsSpec(
            new_tokens=self._copy_to_host(self.store.new_tokens),
            new_tokens_lens=self._copy_to_host(self.store.new_tokens_lens),
            next_draft_tokens=self._copy_to_host(self.store.next_draft_tokens),
            log_probs=None if sampled_log_probs is None else self._copy_to_host(sampled_log_probs),
        )
        sampler_event = self._record_sampler_event()

        # Add dummy draft tokens to context requests for KV cache preparation
        if self._add_dummy_draft_tokens():
            for request in finished_context_requests:
                request.py_draft_tokens = [1] * self.draft_len

        return SampleStateSpec(
            requests=sampling_requests,
            device=device_tensors,
            host=host_tensors,
            sampler_event=sampler_event,
            runtime_draft_len=runtime_draft_len,
        )

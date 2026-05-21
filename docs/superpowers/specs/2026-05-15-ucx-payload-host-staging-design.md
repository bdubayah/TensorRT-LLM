# UCX Payload Host Staging Design

## Context

UCX payload transfers can currently block forever while UCX still owns a live KV buffer pointer. The observed
prefill trace is stuck in `waitForRequestCompletion` from `UcxConnection::send`, reached through
`TransferSession::send`, `MLACacheFormatter::format`, and `CacheSender::Impl::sendAndRemoveResponse`. HEAD added
timeouts for host-control messages, but payload sends and receives still use the original direct pointer path with no
payload timeout.

## Goals

- Allow UCX payload send and receive operations to unwind on timeout.
- Prevent UCX from retaining direct access to KV/device memory after the request has failed.
- Preserve a rollback path through environment variables.
- Keep the first change local to the UCX connection layer.

## Non-Goals

- Replacing UCX, UCXX, or the data transceiver request lifecycle.
- Solving all UCXX worker teardown cases.
- Preserving GPUDirect payload performance in the timeout-safe path.

## Proposed Design

Enable payload host staging by default for UCX transfers, controlled by
`TRTLLM_UCX_ENABLE_PAYLOAD_STAGING`. The default is enabled. Setting the variable to `0`, `false`, or `off` restores the
current direct payload pointer behavior.

Add `TRTLLM_UCX_PAYLOAD_TIMEOUT_MS`, defaulting to `30000`. A value of `0` disables payload timeout handling and leaves
the operation blocking until UCX completes or the existing termination path fires.

For payload sends, copy the source pointer into a pinned host buffer before posting the UCX send. UCX only receives the
host buffer pointer. If the UCX request times out, cancel the request and throw after the existing cancellation grace
period even if UCX has not drained the cancellation. Any retained buffer is host staging memory rather than KV memory.

For payload receives, post the UCX receive into a pinned host buffer. Copy the staged host buffer into the destination
pointer only after UCX reports successful completion and the request error check passes. If the receive times out or
fails, the destination KV/device buffer is left untouched.

Host-control messages keep their existing small `std::vector<char>` staging path and
`TRTLLM_UCX_HOST_CONTROL_TIMEOUT_MS` behavior.

## Error Handling

- On payload operation timeout, log the operation, tag, rank, and timeout.
- Call UCXX request cancellation.
- If cancellation has not completed after the grace period, throw a TensorRT-LLM error and let the caller unwind.
- Treat retained staging memory as acceptable bounded damage compared with retaining KV/device pointers.

## Performance And Risk

This intentionally trades direct UCX access to KV memory for an extra CUDA copy:

- send: device/KV to pinned host, then UCX send
- receive: UCX receive to pinned host, then pinned host to device/KV

The likely cost is lower KV transfer throughput and additional pinned host memory pressure. The default-on behavior is
chosen because the current direct path can wedge a rank permanently. The environment variables provide immediate rollback
for performance or compatibility issues.

## Validation

- Unit coverage for environment parsing defaults and disable cases.
- Build coverage for the UCX connection code.
- Manual repro validation with a bad or unreachable peer:
  - payload timeout is logged
  - request unwinds
  - KV buffer is not exposed to UCX after timeout
  - disabling staging restores the old behavior for comparison

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/executor/cache_transmission/ucx_utils/payloadStaging.h"
#if ENABLE_UCX

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

namespace
{
constexpr int kRequestPollMs = 1;
constexpr int kRequestCancelGraceMs = 1000;
constexpr int kDefaultPayloadRequestTimeoutMs = 30000;
constexpr int kDefaultUnsafeReclaimQuarantinedStagingMs = 0;
constexpr size_t kPayloadChunkBytes = 512ULL * 1024 * 1024;
constexpr size_t kDefaultPayloadStagingChunkBytes = kPayloadChunkBytes;
constexpr size_t kDefaultPayloadStagingPoolSize = 32;
constexpr size_t kDefaultPayloadStagingPipelineDepth = 4;
constexpr size_t kMaxQuarantinedStagedRequests = 64;
constexpr char const* kUcxPayloadStagingEnv = "TRTLLM_UCX_ENABLE_PAYLOAD_STAGING";
constexpr char const* kUcxPayloadTimeoutMsEnv = "TRTLLM_UCX_PAYLOAD_TIMEOUT_MS";
constexpr char const* kUcxPayloadStagingPoolSizeEnv = "TRTLLM_UCX_PAYLOAD_STAGING_POOL_SIZE";
constexpr char const* kUcxPayloadStagingChunkBytesEnv = "TRTLLM_UCX_PAYLOAD_STAGING_CHUNK_BYTES";
constexpr char const* kUcxPayloadStagingPipelineDepthEnv = "TRTLLM_UCX_PAYLOAD_STAGING_PIPELINE_DEPTH";
constexpr char const* kUcxUnsafeReclaimQuarantinedStagingMsEnv = "TRTLLM_UCX_UNSAFE_RECLAIM_QUARANTINED_STAGING_MS";
constexpr int kDecimalBase = 10;
constexpr int32_t kTagTypeBits = 8;
constexpr int32_t kTagTypeMask = (1 << kTagTypeBits) - 1;
constexpr size_t kPayloadChunkTagStartOffset = 1;

template <typename EnvGetter>
size_t getSizeEnvOrDefault(
    int rank, char const* envName, size_t defaultValue, char const* description, char const* unitSuffix,
    EnvGetter&& envGetter)
{
    try
    {
        auto const value = envGetter(envName);
        return value.has_value() ? static_cast<size_t>(*value) : defaultValue;
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(rank, "Invalid %s; using default UCX %s value of %zu%s: %s", envName, description,
            defaultValue, unitSuffix, e.what());
        return defaultValue;
    }
}

size_t getMemorySizeEnvOrDefault(int rank, char const* envName, size_t defaultValue, char const* description)
{
    return getSizeEnvOrDefault(rank, envName, defaultValue, description, " bytes", common::getMemorySizeEnv);
}

size_t getCountEnvOrDefault(int rank, char const* envName, size_t defaultValue, char const* description)
{
    return getSizeEnvOrDefault(rank, envName, defaultValue, description, "", common::getUInt64Env);
}

size_t ceilDiv(size_t dividend, size_t divisor)
{
    TLLM_CHECK_WITH_INFO(divisor != 0, "ceilDiv divisor must be nonzero");
    return dividend / divisor + static_cast<size_t>(dividend % divisor != 0);
}

class PinnedHostBufferPool;

void pruneCompletedQuarantinedRequestsForPoolWait(int rank);

PinnedHostBufferPool& getPayloadStagingBufferPool(int rank);

class PinnedHostBuffer
{
public:
    explicit PinnedHostBuffer(size_t size, int rank)
        : mSize(size)
        , mRank(rank)
    {
        if (mSize > 0)
        {
            TLLM_CUDA_CHECK(cudaHostAlloc(&mData, mSize, cudaHostAllocDefault));
        }
    }

    ~PinnedHostBuffer() noexcept
    {
        if (mData != nullptr)
        {
            auto const status = cudaFreeHost(mData);
            if (status != cudaSuccess && status != cudaErrorCudartUnloading)
            {
                TLLM_LOG_ERROR(mRank, "Failed to free pinned UCX staging buffer of %zu bytes on rank %d: %s", mSize,
                    mRank, cudaGetErrorString(status));
            }
        }
    }

    PinnedHostBuffer(PinnedHostBuffer const&) = delete;
    PinnedHostBuffer& operator=(PinnedHostBuffer const&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&&) = delete;
    PinnedHostBuffer& operator=(PinnedHostBuffer&&) = delete;

    [[nodiscard]] void* data() const
    {
        return mData;
    }

    [[nodiscard]] size_t capacity() const
    {
        return mSize;
    }

private:
    void* mData{nullptr};
    size_t mSize{0};
    int mRank{-1};
};

class PinnedHostBufferPool
{
public:
    explicit PinnedHostBufferPool(int rank)
        : mRank(rank)
    {
    }

    void preallocate()
    {
        (void) ensureInitialized();
    }

    std::shared_ptr<PinnedHostBuffer> acquire(
        size_t size, std::chrono::steady_clock::time_point deadline, std::atomic<bool> const& transferTerminate)
    {
        if (size == 0 || !ensureInitialized())
        {
            return nullptr;
        }
        if (mPoolSize == 0 || size > mBufferBytes)
        {
            return nullptr;
        }

        std::unique_lock<std::mutex> lock(mMutex);
        while (mAvailable.empty())
        {
            lock.unlock();
            pruneCompletedQuarantinedRequestsForPoolWait(mRank);
            lock.lock();
            if (!mAvailable.empty())
            {
                break;
            }
            if (transferTerminate.load())
            {
                TLLM_THROW("Transfer terminated while waiting for UCX payload staging pool buffer");
            }
            auto const now = std::chrono::steady_clock::now();
            if (deadline != std::chrono::steady_clock::time_point::max() && now >= deadline)
            {
                TLLM_THROW("Timed out waiting for UCX payload staging pool buffer");
            }
            auto const waitTime = deadline == std::chrono::steady_clock::time_point::max()
                ? std::chrono::milliseconds(kRequestPollMs)
                : std::min(std::chrono::milliseconds(kRequestPollMs),
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now));
            mAvailableCv.wait_for(lock, waitTime);
            if (transferTerminate.load())
            {
                TLLM_THROW("Transfer terminated while waiting for UCX payload staging pool buffer");
            }
        }
        auto buffer = std::move(mAvailable.back());
        mAvailable.pop_back();
        return {buffer.get(),
            [this, buffer = std::move(buffer)](PinnedHostBuffer*) mutable { release(std::move(buffer)); }};
    }

    bool ownsSize(size_t size)
    {
        if (size == 0 || !ensureInitialized())
        {
            return false;
        }
        return mPoolSize != 0 && size <= mBufferBytes;
    }

    size_t poolSizeForSize(size_t size)
    {
        if (size == 0 || !ensureInitialized())
        {
            return 0;
        }
        return mPoolSize != 0 && size <= mBufferBytes ? mPoolSize : 0;
    }

    void notifyBufferAvailability() noexcept
    {
        mAvailableCv.notify_all();
    }

private:
    bool ensureInitialized()
    {
        if (mDisabled.load())
        {
            return false;
        }
        try
        {
            std::call_once(mInitFlag, [this]() { initialize(); });
        }
        catch (std::exception const& e)
        {
            disableAfterInitializationFailure(e.what());
            return false;
        }
        return !mDisabled.load();
    }

    void initialize()
    {
        mBufferBytes = getMemorySizeEnvOrDefault(
            mRank, kUcxPayloadStagingChunkBytesEnv, kDefaultPayloadStagingChunkBytes, "payload staging chunk");
        mPoolSize = getCountEnvOrDefault(
            mRank, kUcxPayloadStagingPoolSizeEnv, kDefaultPayloadStagingPoolSize, "payload staging pool");
        if (mPoolSize == 0 || mBufferBytes == 0)
        {
            TLLM_LOG_INFO(mRank, "UCX payload staging buffer pool disabled");
            mPoolSize = 0;
            return;
        }

        std::vector<std::shared_ptr<PinnedHostBuffer>> buffers;
        std::vector<std::shared_ptr<PinnedHostBuffer>> available;
        buffers.reserve(mPoolSize);
        available.reserve(mPoolSize);
        // Preallocate the bounded pool so payload transfers fail fast instead of allocating inside UCX progress.
        for (size_t i = 0; i < mPoolSize; ++i)
        {
            buffers.emplace_back(std::make_shared<PinnedHostBuffer>(mBufferBytes, mRank));
            available.push_back(buffers.back());
        }
        mBuffers = std::move(buffers);
        mAvailable = std::move(available);
        TLLM_LOG_INFO(mRank, "Initialized UCX payload staging buffer pool with %zu buffers of %zu bytes", mPoolSize,
            mBufferBytes);
    }

    void release(std::shared_ptr<PinnedHostBuffer> buffer) noexcept
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mAvailable.push_back(std::move(buffer));
        }
        mAvailableCv.notify_one();
    }

    void disableAfterInitializationFailure(std::string const& reason) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mBuffers.clear();
        mAvailable.clear();
        mPoolSize = 0;
        mBufferBytes = 0;
        mDisabled.store(true);
        TLLM_LOG_WARNING(mRank, "Disabling UCX payload staging buffer pool after initialization failure: %s", reason);
    }

    int mRank{-1};
    std::once_flag mInitFlag;
    std::mutex mMutex;
    std::condition_variable mAvailableCv;
    std::atomic<bool> mDisabled{false};
    size_t mPoolSize{0};
    size_t mBufferBytes{0};
    std::vector<std::shared_ptr<PinnedHostBuffer>> mBuffers;
    std::vector<std::shared_ptr<PinnedHostBuffer>> mAvailable;
};

struct QuarantinedRequest
{
    std::shared_ptr<ucxx::Request> request;
    ucxx::RequestCallbackUserData callbackData;
    size_t stagedBytes{0};
    std::chrono::steady_clock::time_point quarantinedAt{};
    // Operation name used only for quarantine diagnostics, e.g. "send" or "recv".
    char const* operation{nullptr};
    int tag{0};
    int rank{-1};
};

bool parseEnvBoolDefaultTrue(int rank, char const* envName, char const* description)
{
    char const* value = std::getenv(envName);
    if (value == nullptr)
    {
        return true;
    }

    std::string normalized{value};
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes")
    {
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no")
    {
        return false;
    }

    TLLM_LOG_WARNING(rank, "Invalid %s=%s; enabling UCX %s by default", envName, value, description);
    return true;
}

int getPayloadRequestTimeoutMs(int rank)
{
    static int const timeoutMs
        = getUcxRequestTimeoutMs(rank, kUcxPayloadTimeoutMsEnv, kDefaultPayloadRequestTimeoutMs, "payload");
    return timeoutMs;
}

int getUnsafeReclaimQuarantinedStagingMs(int rank)
{
    static int const timeoutMs = getUcxRequestTimeoutMs(rank, kUcxUnsafeReclaimQuarantinedStagingMsEnv,
        kDefaultUnsafeReclaimQuarantinedStagingMs, "unsafe quarantined staging reclaim");
    return timeoutMs;
}

PinnedHostBufferPool& getPayloadStagingBufferPool(int rank)
{
    static auto* pool = new PinnedHostBufferPool(rank);
    return *pool;
}

size_t getPayloadStagingChunkBytes(int rank)
{
    static size_t const chunkBytes = getMemorySizeEnvOrDefault(
        rank, kUcxPayloadStagingChunkBytesEnv, kDefaultPayloadStagingChunkBytes, "payload staging chunk");
    return chunkBytes;
}

size_t getPayloadStagingPipelineDepth(int rank)
{
    static size_t const pipelineDepth = [rank]()
    {
        size_t const configuredDepth = getCountEnvOrDefault(
            rank, kUcxPayloadStagingPipelineDepthEnv, kDefaultPayloadStagingPipelineDepth, "payload staging pipeline");
        if (configuredDepth == 0)
        {
            TLLM_LOG_WARNING(rank, "Invalid %s=0; using default UCX payload staging pipeline depth of %zu",
                kUcxPayloadStagingPipelineDepthEnv, kDefaultPayloadStagingPipelineDepth);
            return kDefaultPayloadStagingPipelineDepth;
        }
        return configuredDepth;
    }();
    return pipelineDepth;
}

size_t getEffectivePayloadStagingPipelineDepth(int rank, size_t chunkBytes)
{
    size_t const configuredDepth = getPayloadStagingPipelineDepth(rank);
    size_t const poolSize = getPayloadStagingBufferPool(rank).poolSizeForSize(chunkBytes);
    if (poolSize == 0)
    {
        return 1;
    }
    return std::min(configuredDepth, poolSize);
}

bool shouldChunkPayload(size_t size, int rank)
{
    return size > getPayloadStagingChunkBytes(rank);
}

uint64_t makePayloadChunkTag(uint64_t baseTag, size_t chunkIndex)
{
    constexpr uint64_t kTagTypeMask64 = static_cast<uint64_t>(kTagTypeMask);
    uint64_t const baseTagType = baseTag & kTagTypeMask64;
    size_t const chunkTagType = static_cast<size_t>(baseTagType) + kPayloadChunkTagStartOffset + chunkIndex;
    TLLM_CHECK_WITH_INFO(
        chunkTagType <= kTagTypeMask, "UCX payload chunk index exceeds available request tag type range");
    return (baseTag & ~kTagTypeMask64) | chunkTagType;
}

void validatePayloadChunkTags(uint64_t baseTag, size_t chunkCount)
{
    if (chunkCount == 0)
    {
        return;
    }
    (void) makePayloadChunkTag(baseTag, chunkCount - 1);
}

size_t getPayloadChunkPipelineDepth(int rank)
{
    return getEffectivePayloadStagingPipelineDepth(rank, getPayloadStagingChunkBytes(rank));
}

bool isCudaEventReady(cudaEvent_t event)
{
    auto const status = cudaEventQuery(event);
    if (status == cudaSuccess)
    {
        return true;
    }
    if (status == cudaErrorNotReady)
    {
        return false;
    }
    TLLM_CUDA_CHECK(status);
    return false;
}

using CudaEventPtr = std::shared_ptr<std::remove_pointer_t<cudaEvent_t>>;

CudaEventPtr makeCudaEvent(int rank)
{
    cudaEvent_t event{};
    TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return CudaEventPtr{event,
        [rank](cudaEvent_t cudaEvent) noexcept
        {
            if (cudaEvent == nullptr)
            {
                return;
            }
            auto const status = cudaEventDestroy(cudaEvent);
            if (status != cudaSuccess && status != cudaErrorCudartUnloading)
            {
                TLLM_LOG_ERROR(
                    rank, "Failed to destroy UCX payload staging CUDA event: %s", cudaGetErrorString(status));
            }
        }};
}

std::mutex& getQuarantinedRequestsMutex()
{
    static auto* mutex = new std::mutex();
    return *mutex;
}

std::vector<QuarantinedRequest>& getQuarantinedRequests()
{
    static auto* requests = new std::vector<QuarantinedRequest>();
    return *requests;
}

void pruneReclaimableQuarantinedRequests(std::vector<QuarantinedRequest>& requests, int rank)
{
    int const unsafeReclaimMs = getUnsafeReclaimQuarantinedStagingMs(rank);
    auto const now = std::chrono::steady_clock::now();
    requests.erase(
        std::remove_if(requests.begin(), requests.end(),
            [unsafeReclaimMs, now](QuarantinedRequest const& entry)
            {
                if (entry.request->isCompleted())
                {
                    return true;
                }
                if (unsafeReclaimMs > 0 && now - entry.quarantinedAt >= std::chrono::milliseconds(unsafeReclaimMs))
                {
                    auto const ageMs
                        = std::chrono::duration_cast<std::chrono::milliseconds>(now - entry.quarantinedAt).count();
                    TLLM_LOG_ERROR(entry.rank,
                        "UNSAFE reclaiming canceled UCX %s for tag %d after %ld ms in staged request "
                        "quarantine; staged bytes: %zu",
                        entry.operation, entry.tag, ageMs, entry.stagedBytes);
                    return true;
                }
                return false;
            }),
        requests.end());
}

void pruneQuarantinedStagedRequests(int rank)
{
    std::lock_guard<std::mutex> lock(getQuarantinedRequestsMutex());
    pruneReclaimableQuarantinedRequests(getQuarantinedRequests(), rank);
}

void pruneCompletedQuarantinedRequestsForPoolWait(int rank)
{
    {
        std::lock_guard<std::mutex> lock(getQuarantinedRequestsMutex());
        pruneReclaimableQuarantinedRequests(getQuarantinedRequests(), rank);
    }
    getPayloadStagingBufferPool(rank).notifyBufferAvailability();
}

std::shared_ptr<PinnedHostBuffer> makePinnedHostBuffer(
    size_t size, int rank, DataContext const& ctx, std::chrono::steady_clock::time_point deadline)
{
    auto& pool = getPayloadStagingBufferPool(rank);
    pruneQuarantinedStagedRequests(rank);
    if (auto buffer = pool.acquire(size, deadline, ctx.getTransferTerminate()))
    {
        return buffer;
    }
    if (pool.ownsSize(size))
    {
        TLLM_THROW(
            "UCX payload staging pool exhausted for %zu byte payload; increase %s or %s, or reduce transfer "
            "concurrency",
            size, kUcxPayloadStagingPoolSizeEnv, kUcxPayloadStagingChunkBytesEnv);
    }
    return std::make_shared<PinnedHostBuffer>(size, rank);
}

size_t getQuarantinedBytes(std::vector<QuarantinedRequest> const& requests)
{
    size_t bytes = 0;
    for (auto const& entry : requests)
    {
        bytes += entry.stagedBytes;
    }
    return bytes;
}

void quarantineStagedRequest(std::shared_ptr<ucxx::Request> const& req,
    ucxx::RequestCallbackUserData const& callbackData, size_t stagedBytes, int rank, char const* operation, int tag)
{
    std::lock_guard<std::mutex> lock(getQuarantinedRequestsMutex());
    auto& requests = getQuarantinedRequests();
    pruneReclaimableQuarantinedRequests(requests, rank);
    requests.push_back(
        QuarantinedRequest{req, callbackData, stagedBytes, std::chrono::steady_clock::now(), operation, tag, rank});
    size_t const retainedBytes = getQuarantinedBytes(requests);
    TLLM_LOG_WARNING(rank,
        "Retaining canceled UCX %s for tag %d in staged request quarantine; retained requests: %zu, retained bytes: "
        "%zu",
        operation, tag, requests.size(), retainedBytes);
    if (requests.size() > kMaxQuarantinedStagedRequests)
    {
        TLLM_LOG_ERROR(rank,
            "UCX staged request quarantine exceeded limit: retained requests %zu/%zu, retained bytes %zu; "
            "terminating process",
            requests.size(), kMaxQuarantinedStagedRequests, retainedBytes);
        std::terminate();
    }
}

void cancelRequestWithLog(std::shared_ptr<ucxx::Request> const& req, DataContext const& ctx, int rank,
    char const* operation, char const* reason)
{
    TLLM_LOG_WARNING(rank, "Canceling UCX %s for tag %d: %s", operation, ctx.getTag(), reason);
    req->cancel();
}

std::chrono::steady_clock::time_point getRequestDeadline(int timeoutMs)
{
    if (timeoutMs <= 0)
    {
        return std::chrono::steady_clock::time_point::max();
    }
    return std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
}

int getRemainingTimeoutMs(std::chrono::steady_clock::time_point deadline)
{
    if (deadline == std::chrono::steady_clock::time_point::max())
    {
        return 0;
    }
    auto const now = std::chrono::steady_clock::now();
    if (now >= deadline)
    {
        return 1;
    }
    auto const remainingMs = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    return static_cast<int>(std::max<int64_t>(remainingMs, 1));
}

struct PayloadChunkRequest
{
    std::shared_ptr<ucxx::Request> request;
    std::future<void> future;
    ucxx::RequestCallbackUserData callbackData;
    std::shared_ptr<PinnedHostBuffer> buffer;
    size_t offset{0};
    size_t transferBytes{0};
    size_t stagedBytes{0};
    bool stagedBuffer{false};
};

struct PayloadSendCopy
{
    std::shared_ptr<PinnedHostBuffer> buffer;
    CudaEventPtr copyDone;
    size_t chunkIndex{0};
    size_t transferBytes{0};
    size_t stagedBytes{0};
};

struct PayloadDeviceCopy
{
    std::shared_ptr<PinnedHostBuffer> buffer;
    CudaEventPtr copyDone;
};

void waitForPayloadChunk(PayloadChunkRequest& chunk, DataContext const& ctx, int rank, char const* operation,
    std::chrono::steady_clock::time_point deadline)
{
    waitForUcxRequestCompletion(chunk.request, chunk.future, ctx, rank, operation, chunk.stagedBuffer,
        chunk.callbackData, chunk.stagedBytes, getRemainingTimeoutMs(deadline));
    TLLM_CHECK_WITH_INFO(chunk.request->isCompleted(), "UCX payload chunk should be completed");
    chunk.request->checkError();
}

void quarantineActivePayloadChunks(
    std::deque<PayloadChunkRequest>& chunks, DataContext const& ctx, int rank, char const* operation) noexcept
{
    while (!chunks.empty())
    {
        auto& chunk = chunks.front();
        try
        {
            if (!chunk.request->isCompleted())
            {
                cancelRequestWithLog(chunk.request, ctx, rank, operation, "pipelined payload transfer aborted");
                if (chunk.stagedBuffer)
                {
                    quarantineStagedRequest(
                        chunk.request, chunk.callbackData, chunk.stagedBytes, rank, operation, ctx.getTag());
                }
            }
        }
        catch (...)
        {
            TLLM_LOG_ERROR(rank, "Failed to cancel aborted UCX %s chunk for tag %d", operation, ctx.getTag());
        }
        chunks.pop_front();
    }
}

void sendPayloadChunks(ucxx::Endpoint& endpoint, uint64_t sendTag, DataContext const& ctx, void const* data,
    size_t size, int rank)
{
    size_t const chunkBytes = getPayloadStagingChunkBytes(rank);
    size_t const chunkCount = ceilDiv(size, chunkBytes);
    validatePayloadChunkTags(sendTag, chunkCount);
    size_t const pipelineDepth = getPayloadChunkPipelineDepth(rank);
    auto const deadline = getRequestDeadline(getPayloadRequestTimeoutMs(rank));
    auto const* source = static_cast<char const*>(data);

    cudaStream_t stream{};
    std::deque<PayloadChunkRequest> activeChunks;
    std::deque<PayloadSendCopy> pendingCopies;
    std::shared_ptr<PinnedHostBuffer> pendingCopyBuffer;
    try
    {
        TLLM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        auto waitForOldestSendChunk = [&]()
        {
            auto chunk = std::move(activeChunks.front());
            activeChunks.pop_front();
            waitForPayloadChunk(chunk, ctx, rank, "send", deadline);
        };
        auto submitPendingSendCopy = [&](bool force) -> bool
        {
            if (pendingCopies.empty())
            {
                return false;
            }
            auto& pendingCopy = pendingCopies.front();
            if (force)
            {
                TLLM_CUDA_CHECK(cudaEventSynchronize(pendingCopy.copyDone.get()));
            }
            else if (!isCudaEventReady(pendingCopy.copyDone.get()))
            {
                return false;
            }

            auto copy = std::move(pendingCopy);
            pendingCopies.pop_front();
            void* sendBuffer = copy.buffer->data();
            ucxx::RequestCallbackUserData callbackData = copy.buffer;
            auto promise = std::make_shared<std::promise<void>>();
            std::future<void> future = promise->get_future();
            auto completionCallback
                = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };
            auto req = endpoint.tagSend(sendBuffer, copy.transferBytes,
                ucxx::Tag(makePayloadChunkTag(sendTag, copy.chunkIndex)), false, completionCallback, callbackData);
            PayloadChunkRequest chunk{
                req, std::move(future), callbackData, copy.buffer, 0, copy.transferBytes, copy.stagedBytes, true};
            if (req->isCompleted())
            {
                waitForPayloadChunk(chunk, ctx, rank, "send", deadline);
            }
            else
            {
                activeChunks.push_back(std::move(chunk));
            }
            return true;
        };
        auto submitReadySendCopies = [&](bool force)
        {
            while (submitPendingSendCopy(force))
            {
            }
        };

        size_t chunkIndex = 0;
        for (size_t offset = 0; offset < size; offset += chunkBytes, ++chunkIndex)
        {
            while (activeChunks.size() + pendingCopies.size() >= pipelineDepth)
            {
                submitReadySendCopies(false);
                if (activeChunks.size() + pendingCopies.size() < pipelineDepth)
                {
                    break;
                }
                if (!activeChunks.empty())
                {
                    waitForOldestSendChunk();
                }
                else
                {
                    (void) submitPendingSendCopy(true);
                }
            }

            size_t const chunkSize = std::min(chunkBytes, size - offset);
            auto buffer = makePinnedHostBuffer(chunkSize, rank, ctx, deadline);
            auto copyDone = makeCudaEvent(rank);
            pendingCopyBuffer = buffer;
            TLLM_CUDA_CHECK(cudaMemcpyAsync(buffer->data(), source + offset, chunkSize, cudaMemcpyDefault, stream));
            TLLM_CUDA_CHECK(cudaEventRecord(copyDone.get(), stream));
            pendingCopies.push_back(
                PayloadSendCopy{buffer, std::move(copyDone), chunkIndex, chunkSize, buffer->capacity()});
            pendingCopyBuffer.reset();
            submitReadySendCopies(false);
        }
        submitReadySendCopies(true);
        while (!activeChunks.empty())
        {
            waitForOldestSendChunk();
        }
        if (stream != nullptr)
        {
            TLLM_CUDA_CHECK(cudaStreamDestroy(stream));
            stream = nullptr;
        }
    }
    catch (...)
    {
        if (stream != nullptr)
        {
            (void) cudaStreamSynchronize(stream);
        }
        pendingCopyBuffer.reset();
        pendingCopies.clear();
        quarantineActivePayloadChunks(activeChunks, ctx, rank, "send");
        if (stream != nullptr)
        {
            (void) cudaStreamDestroy(stream);
        }
        throw;
    }
}

void recvPayloadChunks(
    ucxx::Endpoint& endpoint, uint64_t recvTag, DataContext const& ctx, void* data, size_t size, int rank)
{
    size_t const chunkBytes = getPayloadStagingChunkBytes(rank);
    size_t const chunkCount = ceilDiv(size, chunkBytes);
    validatePayloadChunkTags(recvTag, chunkCount);
    size_t const pipelineDepth = getPayloadChunkPipelineDepth(rank);
    auto const deadline = getRequestDeadline(getPayloadRequestTimeoutMs(rank));
    auto* destination = static_cast<char*>(data);

    cudaStream_t stream{};
    TLLM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::deque<PayloadChunkRequest> activeChunks;
    std::deque<PayloadDeviceCopy> pendingDeviceCopies;
    std::shared_ptr<PinnedHostBuffer> pendingCopyBuffer;
    try
    {
        auto releasePendingDeviceCopy = [&](bool force) -> bool
        {
            if (pendingDeviceCopies.empty())
            {
                return false;
            }
            auto& pendingCopy = pendingDeviceCopies.front();
            if (force)
            {
                TLLM_CUDA_CHECK(cudaEventSynchronize(pendingCopy.copyDone.get()));
            }
            else if (!isCudaEventReady(pendingCopy.copyDone.get()))
            {
                return false;
            }
            pendingDeviceCopies.pop_front();
            return true;
        };
        auto releaseReadyDeviceCopies = [&](bool force)
        {
            while (releasePendingDeviceCopy(force))
            {
            }
        };
        size_t nextOffset = 0;
        size_t nextChunkIndex = 0;
        auto postRecv = [&]()
        {
            size_t const offset = nextOffset;
            size_t const chunkIndex = nextChunkIndex;
            size_t const chunkSize = std::min(chunkBytes, size - nextOffset);
            auto buffer = makePinnedHostBuffer(chunkSize, rank, ctx, deadline);
            void* recvBuffer = buffer->data();
            ucxx::RequestCallbackUserData callbackData = buffer;
            size_t stagedBytes = buffer->capacity();
            auto promise = std::make_shared<std::promise<void>>();
            std::future<void> future = promise->get_future();
            auto completionCallback
                = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };
            auto req = endpoint.tagRecv(recvBuffer, chunkSize, ucxx::Tag(makePayloadChunkTag(recvTag, chunkIndex)),
                ucxx::TagMaskFull, false, completionCallback, callbackData);
            activeChunks.push_back(PayloadChunkRequest{
                req, std::move(future), callbackData, buffer, offset, chunkSize, stagedBytes, true});
            nextOffset += chunkSize;
            ++nextChunkIndex;
        };

        while (nextOffset < size && activeChunks.size() + pendingDeviceCopies.size() < pipelineDepth)
        {
            postRecv();
        }
        while (!activeChunks.empty() || nextOffset < size)
        {
            while (nextOffset < size && activeChunks.size() + pendingDeviceCopies.size() >= pipelineDepth)
            {
                releaseReadyDeviceCopies(false);
                if (activeChunks.size() + pendingDeviceCopies.size() < pipelineDepth)
                {
                    break;
                }
                if (!activeChunks.empty())
                {
                    releaseReadyDeviceCopies(false);
                    break;
                }
                (void) releasePendingDeviceCopy(true);
            }
            while (nextOffset < size && activeChunks.size() + pendingDeviceCopies.size() < pipelineDepth)
            {
                postRecv();
            }
            if (activeChunks.empty())
            {
                continue;
            }

            auto chunk = std::move(activeChunks.front());
            activeChunks.pop_front();
            waitForPayloadChunk(chunk, ctx, rank, "recv", deadline);
            auto copyDone = makeCudaEvent(rank);
            pendingCopyBuffer = chunk.buffer;
            TLLM_CUDA_CHECK(cudaMemcpyAsync(
                destination + chunk.offset, chunk.buffer->data(), chunk.transferBytes, cudaMemcpyDefault, stream));
            TLLM_CUDA_CHECK(cudaEventRecord(copyDone.get(), stream));
            pendingDeviceCopies.push_back(PayloadDeviceCopy{chunk.buffer, std::move(copyDone)});
            pendingCopyBuffer.reset();
            releaseReadyDeviceCopies(false);
            chunk.request.reset();
            chunk.callbackData.reset();
            chunk.buffer.reset();
        }
        releaseReadyDeviceCopies(true);
        if (stream != nullptr)
        {
            TLLM_CUDA_CHECK(cudaStreamDestroy(stream));
            stream = nullptr;
        }
    }
    catch (...)
    {
        if (stream != nullptr)
        {
            (void) cudaStreamSynchronize(stream);
        }
        pendingCopyBuffer.reset();
        pendingDeviceCopies.clear();
        quarantineActivePayloadChunks(activeChunks, ctx, rank, "recv");
        if (stream != nullptr)
        {
            (void) cudaStreamDestroy(stream);
        }
        throw;
    }
}

} // namespace

int getUcxRequestTimeoutMs(int rank, char const* envName, int defaultTimeoutMs, char const* timeoutDescription)
{
    char const* value = std::getenv(envName);
    if (value == nullptr)
    {
        return defaultTimeoutMs;
    }
    errno = 0;
    char* end = nullptr;
    long const timeoutMs = std::strtol(value, &end, kDecimalBase);
    if (errno != 0 || end == value || *end != '\0' || timeoutMs < 0 || timeoutMs > std::numeric_limits<int>::max())
    {
        TLLM_LOG_WARNING(rank, "Invalid %s=%s; using default UCX %s timeout of %d ms", envName, value,
            timeoutDescription, defaultTimeoutMs);
        return defaultTimeoutMs;
    }
    return static_cast<int>(timeoutMs);
}

bool isPayloadStagingEnabled(int rank)
{
    static bool const enabled = parseEnvBoolDefaultTrue(rank, kUcxPayloadStagingEnv, "payload staging");
    return enabled;
}

void preallocatePayloadStagingBufferPool(int rank)
{
    if (!isPayloadStagingEnabled(rank))
    {
        return;
    }
    getPayloadStagingBufferPool(rank).preallocate();
}

void waitForUcxRequestCompletion(std::shared_ptr<ucxx::Request> const& req, std::future<void>& future,
    DataContext const& ctx, int rank, char const* operation, bool stagedBuffer,
    ucxx::RequestCallbackUserData const& callbackData, size_t stagedBytes, int timeoutMs)
{
    bool cancelRequested = false;
    bool operationTimedOut = false;
    auto const operationDeadline = timeoutMs > 0
        ? std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs)
        : std::chrono::steady_clock::time_point::max();
    std::chrono::steady_clock::time_point cancelDeadline{};
    while (!req->isCompleted())
    {
        auto const now = std::chrono::steady_clock::now();
        if (!operationTimedOut && now >= operationDeadline)
        {
            operationTimedOut = true;
            if (!cancelRequested)
            {
                cancelRequestWithLog(req, ctx, rank, operation, "operation timeout");
                cancelRequested = true;
                cancelDeadline = now + std::chrono::milliseconds(kRequestCancelGraceMs);
            }
        }
        else if (ctx.getTransferTerminate().load() && !cancelRequested)
        {
            cancelRequestWithLog(req, ctx, rank, operation, "transfer terminated");
            cancelRequested = true;
            cancelDeadline = now + std::chrono::milliseconds(kRequestCancelGraceMs);
        }
        if (cancelRequested && stagedBuffer && now >= cancelDeadline)
        {
            TLLM_LOG_ERROR(rank, "Timed out waiting for canceled UCX %s for tag %d to complete after %d ms", operation,
                ctx.getTag(), kRequestCancelGraceMs);
            quarantineStagedRequest(req, callbackData, stagedBytes, rank, operation, ctx.getTag());
            TLLM_THROW("Timed out waiting for canceled UCX %s for tag %d to complete after %d ms", operation,
                ctx.getTag(), kRequestCancelGraceMs);
        }
        future.wait_for(std::chrono::milliseconds(kRequestPollMs));
    }
    if (operationTimedOut && timeoutMs > 0)
    {
        TLLM_LOG_ERROR(rank, "Timed out waiting for UCX %s for tag %d after %d ms", operation, ctx.getTag(), timeoutMs);
        TLLM_THROW("Timed out waiting for UCX %s for tag %d after %d ms", operation, ctx.getTag(), timeoutMs);
    }
}

void sendPayloadWithStaging(
    ucxx::Endpoint& endpoint, uint64_t sendTag, DataContext const& ctx, void const* data, size_t size, int rank)
{
    TLLM_CHECK_WITH_INFO(isPayloadStagingEnabled(rank), "UCX payload staging is disabled");
    if (shouldChunkPayload(size, rank))
    {
        sendPayloadChunks(endpoint, sendTag, ctx, data, size, rank);
        return;
    }

    int const timeoutMs = getPayloadRequestTimeoutMs(rank);
    auto buffer = makePinnedHostBuffer(size, rank, ctx, getRequestDeadline(timeoutMs));
    TLLM_CUDA_CHECK(cudaMemcpy(buffer->data(), data, size, cudaMemcpyDefault));
    void* sendBuffer = buffer->data();
    ucxx::RequestCallbackUserData callbackData = buffer;
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    auto completionCallback = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };
    auto req = endpoint.tagSend(sendBuffer, size, ucxx::Tag(sendTag), false, completionCallback, callbackData);
    if (!req->isCompleted())
    {
        waitForUcxRequestCompletion(
            req, future, ctx, rank, "send", true, callbackData, buffer->capacity(), timeoutMs);
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "send should be completed");
    req->checkError();
}

void recvPayloadWithStaging(
    ucxx::Endpoint& endpoint, uint64_t recvTag, DataContext const& ctx, void* data, size_t size, int rank)
{
    TLLM_CHECK_WITH_INFO(isPayloadStagingEnabled(rank), "UCX payload staging is disabled");
    if (shouldChunkPayload(size, rank))
    {
        recvPayloadChunks(endpoint, recvTag, ctx, data, size, rank);
        return;
    }

    int const timeoutMs = getPayloadRequestTimeoutMs(rank);
    auto buffer = makePinnedHostBuffer(size, rank, ctx, getRequestDeadline(timeoutMs));
    void* recvBuffer = buffer->data();
    ucxx::RequestCallbackUserData callbackData = buffer;
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    auto completionCallback = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };
    auto req = endpoint.tagRecv(recvBuffer, size, ucxx::Tag(recvTag), ucxx::TagMaskFull, false, completionCallback,
        callbackData);
    if (!req->isCompleted())
    {
        waitForUcxRequestCompletion(
            req, future, ctx, rank, "recv", true, callbackData, buffer->capacity(), timeoutMs);
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "recv should be completed");
    req->checkError();
    TLLM_CUDA_CHECK(cudaMemcpy(data, buffer->data(), size, cudaMemcpyDefault));
}

} // namespace tensorrt_llm::executor::kv_cache

#endif

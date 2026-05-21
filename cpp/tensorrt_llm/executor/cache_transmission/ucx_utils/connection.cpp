/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ucxCacheCommunicator.h"
#if ENABLE_UCX

#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/executor/cache_transmission/ucx_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/ucx_utils/payloadStaging.h"
#include <cstdint>
#include <cstring>
#include <exception>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

namespace
{
constexpr int kDefaultHostControlRequestTimeoutMs = 0;
constexpr char const* kUcxHostControlTimeoutMsEnv = "TRTLLM_UCX_HOST_CONTROL_TIMEOUT_MS";
constexpr int32_t kTagTypeBits = 8;
constexpr int32_t kTagTypeMask = (1 << kTagTypeBits) - 1;

bool isHostControlTag(int tag)
{
    using tensorrt_llm::batch_manager::TransceiverTag;
    // Request data tags encode their type in the low bits. Keep ready-signal tags on the host-control path even if a
    // future caller passes an encoded tag instead of the bare kREADY_SIGNAL_TAG value.
    return tag == TransceiverTag::kINFO_SIZE_TAG || tag == TransceiverTag::kINFO_TAG
        || tag == TransceiverTag::kREADY_SIGNAL_TAG || (tag & kTagTypeMask) == TransceiverTag::kREADY_SIGNAL_TAG;
}

int getHostControlRequestTimeoutMs(int rank)
{
    return getUcxRequestTimeoutMs(
        rank, kUcxHostControlTimeoutMsEnv, kDefaultHostControlRequestTimeoutMs, "host control");
}

void sendPayloadWithoutStaging(ucxx::Endpoint& endpoint, uint64_t sendTag, void const* data, size_t size)
{
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    auto completionCallback = [&](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise.set_value(); };
    auto req = endpoint.tagSend(const_cast<void*>(data), size, ucxx::Tag(sendTag), false, completionCallback);
    if (!req->isCompleted())
    {
        future.get();
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "send should be completed");
    req->checkError();
}

void recvPayloadWithoutStaging(ucxx::Endpoint& endpoint, uint64_t recvTag, void* data, size_t size)
{
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    auto completionCallback = [&](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise.set_value(); };
    auto req = endpoint.tagRecv(data, size, ucxx::Tag(recvTag), ucxx::TagMaskFull, false, completionCallback);
    if (!req->isCompleted())
    {
        future.get();
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "recv should be completed");
    req->checkError();
}

void logSendEnd(UcxConnection::ConnectionIdType connectionId, UcxConnection::ConnectionIdType connectionIdInPeer,
    bool fromRequester, int rank)
{
    TLLM_LOG_DEBUG(rank, "end UcxConnection::send , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        connectionId, connectionIdInPeer, fromRequester);
}

void logRecvEnd(UcxConnection::ConnectionIdType connectionId, UcxConnection::ConnectionIdType connectionIdInPeer,
    bool fromRequester, int rank)
{
    TLLM_LOG_DEBUG(rank, "end UcxConnection::recv , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        connectionId, connectionIdInPeer, fromRequester);
}

} // namespace

UcxConnection::UcxConnection(ConnectionIdType connectionId, std::shared_ptr<ucxx::Endpoint> endpoint,
    UcxConnectionManager* manager, bool fromRequester)
    : mConnectionId(connectionId)
    , mEndpoint(std::move(endpoint))
    , mManager(manager)
    , mFromRequester(fromRequester)
{

    try
    {
        if (mFromRequester)
        {

            // since the tag don't contain the information of the connection id or mConnectionIdInPeer, we need to
            // lock the mutex ,to ensure only one tagRecv is called in the same time.
            std::shared_ptr<ucxx::Request> recvRequest
                = mEndpoint->tagRecv(reinterpret_cast<void*>(&mConnectionIdInPeer), sizeof(mConnectionIdInPeer),
                    ucxx::Tag(ResponserTag), ucxx::TagMaskFull);
            while (!recvRequest->isCompleted())
                ;

            recvRequest->checkError();

            auto sendTag = ucxx::Tag(mConnectionIdInPeer << 32 | (RequesterTag & 0xFFFFFFFF));
            std::shared_ptr<ucxx::Request> sendRequest
                = mEndpoint->tagSend(reinterpret_cast<void*>(&mConnectionId), sizeof(mConnectionId), sendTag);
            while (!sendRequest->isCompleted())
                ;
            sendRequest->checkError();
        }
        else
        {

            // Since Responder may recv from multiple Requesters, we need to send the mConnectionId to the Reqester
            // first and use ConnectionId as the tag to recv the mConnectionIdInPeer from the Requester
            std::shared_ptr<ucxx::Request> sendRequest = mEndpoint->tagSend(
                reinterpret_cast<void*>(&mConnectionId), sizeof(mConnectionId), ucxx::Tag(ResponserTag));
            while (!sendRequest->isCompleted())
                ;
            sendRequest->checkError();

            auto recvTag = ucxx::Tag(mConnectionId << 32 | (RequesterTag & 0xFFFFFFFF));
            std::shared_ptr<ucxx::Request> recvRequest = mEndpoint->tagRecv(
                reinterpret_cast<void*>(&mConnectionIdInPeer), sizeof(mConnectionIdInPeer), recvTag, ucxx::TagMaskFull);
            while (!recvRequest->isCompleted())
                ;
            recvRequest->checkError();
        }
    }
    catch (std::exception const& e)
    {
        std::string error = std::string("Error in UcxConnection constructor for rank ")
            + std::to_string(mManager->getRank()) + ": " + e.what();
        TLLM_THROW(error);
    }

    mSendTagPrefix = mConnectionIdInPeer;
    mRecvTagPrefix = mConnectionId;

    TLLM_LOG_DEBUG(mManager->getRank(),
        "UcxConnection::UcxConnection, mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
}

UcxConnection::~UcxConnection()
{

    TLLM_LOG_DEBUG(mManager->getRank(),
        "UcxConnection::~UcxConnection, mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
    // TODO: how to close the endpoint safely?
}

void UcxConnection::sendConnectionId(DataContext const& ctx, void const* data, size_t size) const
{
    TLLM_LOG_DEBUG(mManager->getRank(),
        "start UcxConnection::sendConnectionId , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        mConnectionId, mConnectionIdInPeer, mFromRequester);

    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    auto completionCallback = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };

    uint64_t tag = ((mSendTagPrefix & 0xFFFFFFFF) << 32)
        | static_cast<uint64_t>(tensorrt_llm::batch_manager::TransceiverTag::kID_TAG);
    auto buffer = std::make_shared<std::vector<char>>(size + sizeof(mConnectionId));
    memcpy(buffer->data(), data, size);
    memcpy(buffer->data() + size, &mConnectionIdInPeer, sizeof(mConnectionIdInPeer));
    auto req = mEndpoint->tagSend(buffer->data(), buffer->size(), ucxx::Tag(tag), false, completionCallback, buffer);
    if (!req->isCompleted())
    {
        waitForUcxRequestCompletion(req, future, ctx, mManager->getRank(), "sendConnectionId", true, buffer,
            buffer->size(), getHostControlRequestTimeoutMs(mManager->getRank()));
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "sendConnectionId should be completed");
    req->checkError();
    TLLM_LOG_DEBUG(mManager->getRank(),
        "end UcxConnection::sendConnectionId , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        mConnectionId, mConnectionIdInPeer, mFromRequester);
}

void UcxConnection::send(DataContext const& ctx, void const* data, size_t size) const
{
    if (ctx.getTag() == tensorrt_llm::batch_manager::TransceiverTag::kID_TAG)
    {
        sendConnectionId(ctx, data, size);
        return;
    }
    TLLM_LOG_DEBUG(mManager->getRank(),
        "start UcxConnection::send , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);

    TLLM_CHECK_WITH_INFO((mEndpoint), "sendBuffer called without established communicator channel.");
    uint64_t sendTag = ((mSendTagPrefix & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(ctx.getTag()) & (0xFFFFFFFF));
    int const rank = mManager->getRank();
    bool const hostControlTag = isHostControlTag(ctx.getTag());
    if (!hostControlTag)
    {
        if (isPayloadStagingEnabled(rank))
        {
            sendPayloadWithStaging(*mEndpoint, sendTag, ctx, data, size, rank);
        }
        else
        {
            sendPayloadWithoutStaging(*mEndpoint, sendTag, data, size);
        }
        logSendEnd(mConnectionId, mConnectionIdInPeer, mFromRequester, rank);
        return;
    }

    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    auto completionCallback = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };
    auto hostControlBuffer = std::make_shared<std::vector<char>>(size);
    memcpy(hostControlBuffer->data(), data, size);
    ucxx::RequestCallbackUserData callbackData = hostControlBuffer;
    auto req = mEndpoint->tagSend(
        hostControlBuffer->data(), size, ucxx::Tag(sendTag), false, completionCallback, callbackData);
    if (!req->isCompleted())
    {
        waitForUcxRequestCompletion(req, future, ctx, rank, "send", true, callbackData, hostControlBuffer->size(),
            getHostControlRequestTimeoutMs(rank));
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "send should be completed");
    req->checkError();

    logSendEnd(mConnectionId, mConnectionIdInPeer, mFromRequester, rank);
}

void UcxConnection::recv(DataContext const& ctx, void* data, size_t size) const
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_LOG_DEBUG(mManager->getRank(),
        "start UcxConnection::recv , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
    TLLM_CHECK_WITH_INFO((mEndpoint), "recvBuffer called without established communicator channel.");
    uint64_t recvTag = ((mRecvTagPrefix & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(ctx.getTag()) & (0xFFFFFFFF));
    int const rank = mManager->getRank();
    bool const hostControlTag = isHostControlTag(ctx.getTag());
    if (!hostControlTag)
    {
        if (isPayloadStagingEnabled(rank))
        {
            recvPayloadWithStaging(*mEndpoint, recvTag, ctx, data, size, rank);
        }
        else
        {
            recvPayloadWithoutStaging(*mEndpoint, recvTag, data, size);
        }
        logRecvEnd(mConnectionId, mConnectionIdInPeer, mFromRequester, rank);
        return;
    }

    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    auto completionCallback = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };
    auto hostControlBuffer = std::make_shared<std::vector<char>>(size);
    ucxx::RequestCallbackUserData callbackData = hostControlBuffer;
    auto req = mEndpoint->tagRecv(
        hostControlBuffer->data(), size, ucxx::Tag(recvTag), ucxx::TagMaskFull, false, completionCallback, callbackData);
    if (!req->isCompleted())
    {
        waitForUcxRequestCompletion(req, future, ctx, rank, "recv", true, callbackData, hostControlBuffer->size(),
            getHostControlRequestTimeoutMs(rank));
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "recv should be completed");
    req->checkError();
    memcpy(data, hostControlBuffer->data(), size);

    logRecvEnd(mConnectionId, mConnectionIdInPeer, mFromRequester, rank);
}

} // namespace tensorrt_llm::executor::kv_cache

#endif

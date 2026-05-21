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

#pragma once

#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "ucxx/api.h"
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>

namespace tensorrt_llm::executor::kv_cache
{

bool isPayloadStagingEnabled(int rank);

void preallocatePayloadStagingBufferPool(int rank);

int getUcxRequestTimeoutMs(int rank, char const* envName, int defaultTimeoutMs, char const* timeoutDescription);

void waitForUcxRequestCompletion(std::shared_ptr<ucxx::Request> const& req, std::future<void>& future,
    DataContext const& ctx, int rank, char const* operation, bool stagedBuffer,
    ucxx::RequestCallbackUserData const& callbackData, size_t stagedBytes, int timeoutMs = 0);

void sendPayloadWithStaging(
    ucxx::Endpoint& endpoint, uint64_t sendTag, DataContext const& ctx, void const* data, size_t size, int rank);

void recvPayloadWithStaging(
    ucxx::Endpoint& endpoint, uint64_t recvTag, DataContext const& ctx, void* data, size_t size, int rank);

} // namespace tensorrt_llm::executor::kv_cache

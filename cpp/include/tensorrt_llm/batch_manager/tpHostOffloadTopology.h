/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#pragma once

#include "tensorrt_llm/runtime/common.h"

#include <cstddef>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

using SizeType32 = tensorrt_llm::runtime::SizeType32;

struct TpHostOffloadBlockMapping
{
    int ownerRank{-1};
    SizeType32 ownerLocalBlockIdx{0};
};

//! Describes deterministic ownership for replicated TP MLA host offload.
//!
//! All TP ranks keep replicated block metadata, but only the rank that owns a secondary host block allocates and copies
//! that block. Global secondary blocks are striped across TP ranks in TP-group order; the owner-local block is the index
//! within that rank's sharded host pool.
class TpHostOffloadTopology
{
public:
    static TpHostOffloadTopology fromTpGroupRanks(
        std::vector<SizeType32> const& tpGroupRanks, SizeType32 numSecondaryBlocks);

    [[nodiscard]] TpHostOffloadBlockMapping blockMapping(SizeType32 globalBlockIdx) const;

    [[nodiscard]] SizeType32 localOwnedBlockCount(int ownerRank) const;

private:
    TpHostOffloadTopology(std::vector<int> ownerRanks, SizeType32 numSecondaryBlocks);

    std::vector<int> mOwnerRanks;
    SizeType32 mNumSecondaryBlocks{0};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager

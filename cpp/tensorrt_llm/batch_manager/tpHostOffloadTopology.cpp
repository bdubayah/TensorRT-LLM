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

#include "tensorrt_llm/batch_manager/tpHostOffloadTopology.h"

#include "tensorrt_llm/common/assert.h"

#include <algorithm>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
TpHostOffloadTopology::TpHostOffloadTopology(std::vector<int> ownerRanks, SizeType32 numSecondaryBlocks)
    : mOwnerRanks{std::move(ownerRanks)}
    , mNumSecondaryBlocks{numSecondaryBlocks}
{
}

TpHostOffloadTopology TpHostOffloadTopology::fromTpGroupRanks(
    std::vector<SizeType32> const& tpGroupRanks, SizeType32 numSecondaryBlocks)
{
    std::vector<int> ownerRanks;
    ownerRanks.reserve(tpGroupRanks.size());

    for (auto const rank : tpGroupRanks)
    {
        int const worldRank = static_cast<int>(rank);
        TLLM_CHECK_WITH_INFO(worldRank >= 0, "Invalid TP rank %d", worldRank);
        TLLM_CHECK_WITH_INFO(std::find(ownerRanks.begin(), ownerRanks.end(), worldRank) == ownerRanks.end(),
            "Duplicate TP rank %d", worldRank);

        ownerRanks.push_back(worldRank);
    }

    TLLM_CHECK_WITH_INFO(!ownerRanks.empty(), "TP host offload topology requires at least one owner rank");
    return TpHostOffloadTopology{std::move(ownerRanks), numSecondaryBlocks};
}

TpHostOffloadBlockMapping TpHostOffloadTopology::blockMapping(SizeType32 globalBlockIdx) const
{
    TLLM_CHECK_WITH_INFO(globalBlockIdx < mNumSecondaryBlocks, "globalBlockIdx=%u is out of range [%u)",
        globalBlockIdx, mNumSecondaryBlocks);
    auto const ownerBucket = globalBlockIdx % static_cast<SizeType32>(mOwnerRanks.size());
    auto const ownerLocalBlockIdx = globalBlockIdx / static_cast<SizeType32>(mOwnerRanks.size());
    return TpHostOffloadBlockMapping{mOwnerRanks.at(ownerBucket), ownerLocalBlockIdx};
}

SizeType32 TpHostOffloadTopology::localOwnedBlockCount(int ownerRank) const
{
    auto const ownerRankIt = std::find(mOwnerRanks.begin(), mOwnerRanks.end(), ownerRank);
    if (ownerRankIt == mOwnerRanks.end())
    {
        return 0U;
    }
    auto const ownerBucket = static_cast<SizeType32>(std::distance(mOwnerRanks.begin(), ownerRankIt));
    if (ownerBucket >= mNumSecondaryBlocks)
    {
        return 0U;
    }
    return ((mNumSecondaryBlocks - 1 - ownerBucket) / static_cast<SizeType32>(mOwnerRanks.size())) + 1;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager

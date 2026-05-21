/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "RoutingKernel.cuh"

#include <algorithm>

namespace moe::dev::routing
{
namespace routingMiniMax
{

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumExperts128 = 128;
static constexpr int NumExperts256 = 256;
static constexpr int MaxSupportedExperts = NumExperts256;
static constexpr int MaxSupportedTopExperts = 8;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024)
    routingIndicesHistogramScoresKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    static constexpr int NumThreadsBlock = KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024;
    static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const warpIdx = threadIdx.x / WarpSize;
    int32_t const globalWarpIdx = blockIdx.x * NumThreadsBlock / WarpSize + warpIdx;
    int32_t const globalWarpStride = gridDim.x * NumThreadsBlock / WarpSize;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

    int32_t expertCountsNum = 2 * params.mNumExperts;
    int32_t globalThreadIdx = blockIdx.x * NumThreadsBlock + threadIdx.x;
    int32_t globalThreadStride = gridDim.x * NumThreadsBlock;
    initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

    float constexpr minScore = -INFINITY;
    float selectScores[VecSize];
    int32_t expertIdx[VecSize];
    float warpTopKScore[KernelParams::MaxNumTopExperts];
    int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];
    for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride)
    {
        auto scoreOffset = tokenIdx * params.mNumExperts;

#pragma unroll
        for (int ii = 0; ii < VecSize; ++ii)
        {
            auto idx = ii * WarpSize + laneIdx;
            expertIdx[ii] = idx;
            if (idx < params.mNumExperts)
            {
                auto const prob = sigmoid_accurate(static_cast<float>(params.mPtrScores[scoreOffset + idx]));
                auto const bias = static_cast<float>(params.mPtrRoutingBias[idx]);
                selectScores[ii] = prob + bias;
            }
            else
            {
                selectScores[ii] = minScore;
            }
        }

        topk::reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, selectScores, expertIdx, minScore, params.mTopK);

        float prob = 0.f;
        if (laneIdx < params.mTopK)
        {
            auto idx = warpTopKExpertIdx[laneIdx];
            if (idx >= 0 && idx < params.mNumExperts)
            {
                prob = sigmoid_accurate(static_cast<float>(params.mPtrScores[scoreOffset + idx]));
            }
        }

        float finalWeight = prob;
        if (params.mNormTopkProb)
        {
            float denom = cg::reduce(warp, laneIdx < params.mTopK ? prob : 0.f, cg::plus<float>()) + 1e-20f;
            finalWeight = laneIdx < params.mTopK ? prob / denom : 0.f;
        }

        if (laneIdx < params.mTopK)
        {
            PackedScoreIdx<OutputT> packedScore{
                static_cast<OutputT>(finalWeight), static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
            if (params.mPtrTopKPacked != nullptr)
            {
                params.mPtrTopKPacked[tokenIdx * params.mTopK + laneIdx] = packedScore;
            }
            if (params.mPtrTopKWeights != nullptr)
            {
                params.mPtrTopKWeights[tokenIdx * params.mTopK + laneIdx] = static_cast<OutputT>(finalWeight);
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int constexpr getMaxNumExperts(int32_t numExperts)
{
    if (numExperts <= NumExperts128)
    {
        return NumExperts128;
    }
    if (numExperts <= NumExperts256)
    {
        return NumExperts256;
    }
    TLLM_LOG_ERROR("Unsupported numExperts");
    return 0;
}

#define LAUNCH_ROUTING_MINIMAX_NUM_EXPERTS(                                                                            \
    data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, numExperts)                                     \
    if (data.mDtypeExpW == tg::Dtype::Fp32)                                                                            \
    {                                                                                                                  \
        LAUNCH_TILEN(data, coopLaunch, LAUNCH_ESC(float, float, numExperts, MaxSupportedTopExperts), kernel,           \
            numBlocks, numThreads, smemSize, stream);                                                                  \
    }                                                                                                                  \
    else if (data.mDtypeExpW == tg::Dtype::Bfloat16)                                                                   \
    {                                                                                                                  \
        LAUNCH_TILEN(data, coopLaunch, LAUNCH_ESC(float, __nv_bfloat16, numExperts, MaxSupportedTopExperts), kernel,   \
            numBlocks, numThreads, smemSize, stream);                                                                  \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported dtypeExpW");                                                                       \
    }

#define LAUNCH_ROUTING_MINIMAX(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream)                      \
    if (data.mNumExperts <= NumExperts128)                                                                             \
    {                                                                                                                  \
        LAUNCH_ROUTING_MINIMAX_NUM_EXPERTS(                                                                            \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, NumExperts128);                         \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumExperts256)                                                                        \
    {                                                                                                                  \
        LAUNCH_ROUTING_MINIMAX_NUM_EXPERTS(                                                                            \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, NumExperts256);                         \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported numExperts");                                                                      \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr,
        "Routing kernel requires at least one input parameter");
    if (data.mPtrTopKIds != nullptr)
    {
        TLLM_CHECK_WITH_INFO(data.mPtrTopKWeights != nullptr,
            "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for MiniMax routing.");
    }
    if (data.mPtrScores != nullptr)
    {
        TLLM_CHECK_WITH_INFO(
            data.mPtrRoutingBias != nullptr, "MiniMax routing requires routing bias when routing from logits.");
    }
    if (data.mPtrExpandedIdxToPermutedIdx != nullptr || data.mPtrPermutedIdxToExpandedIdx != nullptr
        || data.mPtrPermutedIdxToTokenIdx != nullptr)
    {
        TLLM_CHECK_WITH_INFO(
            (data.mPtrTopKPacked != nullptr || data.mPtrTopKIds != nullptr) && data.mPtrPermutedIdxSize != nullptr,
            "If permuted index is required, `mPtrTopKPacked` or `mPtrTopKIds` is also required");
    }

    TLLM_CHECK_WITH_INFO(data.mTopK > 0 && data.mTopK <= MaxSupportedTopExperts,
        "MiniMax routing expects 0 < topK <= %d, got %d", MaxSupportedTopExperts, data.mTopK);
    TLLM_CHECK_WITH_INFO(data.mNumExperts <= MaxSupportedExperts,
        "MiniMax routing expects #experts %d to be no more than %d", data.mNumExperts, MaxSupportedExperts);
    TLLM_CHECK_WITH_INFO(
        data.mNumExperts % 4 == 0, "MiniMax routing expects #experts %d to be a multiple of 4.", data.mNumExperts);

    int const numThreadsHist = getMaxNumExperts(data.mNumExperts);
    if (data.mPtrScores != nullptr && data.mPtrTopKIds == nullptr)
    {
        uint32_t constexpr maxNumBlocks = 1024;
        int const warpsPerBlock = numThreadsHist / WarpSize;
        int const numBlocksScore
            = std::max(1, std::min<int32_t>(tensorrt_llm::common::divUp(data.mNumTokens, warpsPerBlock), maxNumBlocks));
        LAUNCH_ROUTING_MINIMAX(data, false, routingIndicesHistogramScoresKernel, numBlocksScore, numThreadsHist,
            /*smemSize=*/0, stream);
    }

    if (data.mPtrPermutedIdxSize != nullptr)
    {
        uint32_t constexpr maxNumBlocks = 1024;
        uint32_t const expandedIdxSize = data.mNumTokens * data.mTopK;
        uint32_t const histogramEltsPerBlock = 8 * numThreadsHist;
        uint32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;
        int const numBlocksHistogram
            = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
        int const numBlocksOffsets
            = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

        if (data.mPtrTopKIds != nullptr)
        {
            LAUNCH_ROUTING_MINIMAX(data, false, routingInitExpertCounts,
                (2 * data.mNumExperts - 1) / numThreadsHist + 1, numThreadsHist, /*smemSize=*/0, stream);
        }
        LAUNCH_ROUTING_MINIMAX(
            data, false, routingIndicesHistogramKernel, numBlocksHistogram, numThreadsHist, /*smemSize=*/0, stream);
        LAUNCH_ROUTING_MINIMAX(
            data, false, routingIndicesOffsetsKernel, numBlocksOffsets, numThreadsHist, /*smemSize=*/0, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingMiniMax
} // namespace moe::dev::routing

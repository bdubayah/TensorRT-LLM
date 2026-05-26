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

#include <cstdint>
#include <iterator>

#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"

#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/kernels/kvCachePartialCopy.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"

namespace tr = tensorrt_llm::runtime;
namespace tk = tensorrt_llm::kernels;
namespace kvc = tensorrt_llm::executor::kv_cache;

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

namespace
{

void validateReplicatedHostOffloadMode(executor::KvCacheTransferMode mode)
{
    TLLM_CHECK_WITH_INFO(mode == executor::KvCacheTransferMode::DRAM,
        "Replicated TP MLA host offload only supports DRAM transfer mode, got %d", static_cast<int>(mode));
}

// Replicated TP MLA host-offload flow:
// - Offload: all TP ranks observe the same secondary block id, but only the deterministic owner rank writes the
//   compact host-offload block.
// - Onboard: the owner rank copies the compact host block back to its primary GPU block, then every TP rank enters a
//   NCCL broadcast on the broadcast stream so non-owners receive the owner's primary block before it is reused.

} // namespace

static bool gpuToFilePosix(tr::ITensor::SharedPtr const& srcPtr, std::string const& filename)
{
    int fd = ::open(filename.c_str(), O_CREAT | O_WRONLY, 0664);
    TLLM_CHECK_WITH_INFO(fd >= 0, "Failed to open '%s' for writing (POSIX fallback)", filename.c_str());

    ssize_t numBytes = static_cast<ssize_t>(srcPtr->getSizeInBytes());
    std::vector<uint8_t> hostBuffer(numBytes);

    cudaError_t cpyErr = cudaMemcpy(hostBuffer.data(), srcPtr->data(), numBytes, cudaMemcpyDeviceToHost);
    TLLM_CHECK_WITH_INFO(cpyErr == cudaSuccess, "cudaMemcpy to host failed, error=%d", cpyErr);

    ssize_t written = ::write(fd, hostBuffer.data(), numBytes);
    TLLM_CHECK_WITH_INFO(written >= 0, "POSIX write error=%zd", written);

    TLLM_LOG_DEBUG("Wrote %zd bytes to %s (POSIX fallback)", written, filename.c_str());

    ::close(fd);
    return true;
}

static bool fileToGpuPosix(tr::ITensor::SharedPtr const& dstPtr, std::string const& filename)
{
    int fd = ::open(filename.c_str(), O_RDONLY);
    TLLM_CHECK_WITH_INFO(fd >= 0, "Failed to open '%s' for reading (POSIX fallback)", filename.c_str());

    ssize_t numBytes = static_cast<ssize_t>(dstPtr->getSizeInBytes());
    std::vector<uint8_t> hostBuffer(numBytes);

    ssize_t bytesRead = ::read(fd, hostBuffer.data(), numBytes);
    TLLM_CHECK_WITH_INFO(bytesRead >= 0, "POSIX read error=%zd", bytesRead);

    TLLM_LOG_DEBUG("Read %zd bytes from %s (POSIX fallback)", bytesRead, filename.c_str());

    cudaError_t cpyErr = cudaMemcpy(dstPtr->data(), hostBuffer.data(), numBytes, cudaMemcpyHostToDevice);
    TLLM_CHECK_WITH_INFO(cpyErr == cudaSuccess, "cudaMemcpy to device failed, error=%d", cpyErr);

    ::close(fd);
    return true;
}

KVCacheTransferManager::KVCacheTransferManager(tr::BufferManager const& bufferManager,
    std::shared_ptr<kvc::BaseLoopbackAgent> loopbackAgent, bool enableTpMlaReplicatedHostOffload,
    std::set<int> tpGroupRanks, std::optional<TpHostOffloadTopology> tpHostOffloadTopology, int worldRank)
    : mBufferManager{bufferManager}
    , mOnboardManager(std::make_shared<tr::CudaStream>())
    , mOffloadManager(std::make_shared<tr::CudaStream>())
    , mBroadcastStream(enableTpMlaReplicatedHostOffload ? std::make_shared<tr::CudaStream>() : nullptr)
    , mEnableTpMlaReplicatedHostOffload{enableTpMlaReplicatedHostOffload}
    , mTpGroupRanks{std::move(tpGroupRanks)}
    , mTpHostOffloadTopology{std::move(tpHostOffloadTopology)}
    , mWorldRank{worldRank}
    , mLoopbackAgent{loopbackAgent}
{
    if (mEnableTpMlaReplicatedHostOffload)
    {
        TLLM_CHECK_WITH_INFO(
            mTpHostOffloadTopology.has_value(), "Replicated TP MLA host offload requires an ownership topology.");
        TLLM_CHECK_WITH_INFO(
            mTpGroupRanks.count(mWorldRank) == 1, "Rank %d is not a member of the TP group.", mWorldRank);
    }
    TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
    TLLM_CHECK(mDeviceId != -1);
}

std::size_t KVCacheTransferManager::PendingTransferKeyHash::operator()(PendingTransferKey const& key) const
{
    auto const offset = static_cast<std::uint32_t>(key.offset);
    auto const encoded = (static_cast<std::uint64_t>(offset) << 1U) | static_cast<std::uint64_t>(key.isPrimary);
    return std::hash<std::uint64_t>{}(encoded);
}

tr::ITensor::SharedPtr KVCacheTransferManager::computeBlockPointer(
    BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools, size_t poolIdx)
{
    TLLM_CHECK_WITH_INFO(poolIdx < pools.size(), "Pool index %lu is out of bounds", poolIdx);
    auto const& pool = pools.at(poolIdx);
    auto ptr = block->isPrimary() ? pool.primaryPtr : pool.secondaryPtr;
    TLLM_CHECK_WITH_INFO(ptr != nullptr, "Missing %s pool pointer for block %d",
        block->isPrimary() ? "primary" : "secondary", block->getBlockId());
    // getMemoryPoolBlockIndex() is the logical block index within the selected pool level.
    // Replicated TP MLA host offload remaps secondary blocks to the owner's compact local storage.
    auto blockOffset = static_cast<runtime::SizeType32>(block->getMemoryPoolBlockIndex());
    if (mEnableTpMlaReplicatedHostOffload && !block->isPrimary())
    {
        auto const blockMapping = blockMappingForSecondaryBlock(block);
        TLLM_CHECK_WITH_INFO(blockMapping.ownerRank == mWorldRank,
            "Rank %d attempted to access secondary block %d owned by rank %d", mWorldRank, blockOffset,
            blockMapping.ownerRank);
        blockOffset = blockMapping.ownerLocalBlockIdx;
    }
    tr::ITensor::SharedPtr blockTensor{tr::ITensor::slice(ptr, blockOffset, 1)};
    return blockTensor;
}

KVCacheTransferManager::PendingTransferKey KVCacheTransferManager::computePendingTransferKey(BlockPtr const& block)
{
    return PendingTransferKey{block->getMemoryPoolBlockIndex(), block->isPrimary()};
}

TpHostOffloadBlockMapping KVCacheTransferManager::blockMappingForSecondaryBlock(BlockPtr const& block) const
{
    TLLM_CHECK_WITH_INFO(!block->isPrimary(), "Secondary block mapping is only valid for secondary blocks.");
    TLLM_CHECK_WITH_INFO(
        mTpHostOffloadTopology.has_value(), "Missing ownership topology for replicated TP MLA host offload.");
    return mTpHostOffloadTopology->blockMapping(block->getMemoryPoolBlockIndex());
}

int KVCacheTransferManager::tpGroupRankForWorldRank(int worldRank) const
{
    auto const rootIt = mTpGroupRanks.find(worldRank);
    TLLM_CHECK_WITH_INFO(rootIt != mTpGroupRanks.end(), "Rank %d is not a member of the TP group.", worldRank);
    return static_cast<int>(std::distance(mTpGroupRanks.begin(), rootIt));
}

void KVCacheTransferManager::waitForPendingReads(
    PendingReadMap& pendingReads, PendingTransferKey const& key, tr::CudaStream const& stream, bool eraseAfterWait)
{
    auto pendingReadItr = pendingReads.find(key);
    if (pendingReadItr != pendingReads.end())
    {
        for (auto const& pendingRead : pendingReadItr->second)
        {
            stream.wait(pendingRead);
        }
        if (eraseAfterWait)
        {
            pendingReads.erase(pendingReadItr);
        }
    }
}

void KVCacheTransferManager::waitForPendingTransfer(
    PendingWriteMap& pendingTransfers, PendingTransferKey const& key, tr::CudaStream const& stream,
    bool eraseAfterWait)
{
    auto pendingTransferItr = pendingTransfers.find(key);
    if (pendingTransferItr != pendingTransfers.end())
    {
        stream.wait(pendingTransferItr->second);
        if (eraseAfterWait)
        {
            pendingTransfers.erase(pendingTransferItr);
        }
    }
}

void KVCacheTransferManager::waitForPendingRead(
    PendingTransferKey const& key, tr::CudaStream const& stream, bool eraseAfterWait)
{
    waitForPendingReads(mPendingReads, key, stream, eraseAfterWait);
}

void KVCacheTransferManager::waitForPendingWrite(
    PendingTransferKey const& key, tr::CudaStream const& stream, bool eraseAfterWait)
{
    waitForPendingTransfer(mPendingWrites, key, stream, eraseAfterWait);
}

void KVCacheTransferManager::recordPendingTransfer(
    PendingWriteMap& pendingTransfers, PendingTransferKey const& key, tr::CudaStream const& stream)
{
    auto [pendingTransferItr, inserted] = pendingTransfers.emplace(key, tr::CudaEvent());
    TLLM_CHECK_WITH_INFO(inserted, "Previous pending write event still exists for the block.");
    stream.record(pendingTransferItr->second);
}

void KVCacheTransferManager::recordPendingRead(PendingTransferKey const& key, tr::CudaStream const& stream)
{
    auto& pendingReads = mPendingReads[key];
    pendingReads.emplace_back();
    stream.record(pendingReads.back());
}

void KVCacheTransferManager::recordPendingWrite(PendingTransferKey const& key, tr::CudaStream const& stream)
{
    recordPendingTransfer(mPendingWrites, key, stream);
}

void KVCacheTransferManager::broadcastBlock(
    BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools, int rootWorldRank)
{
    TLLM_CHECK_WITH_INFO(mEnableTpMlaReplicatedHostOffload,
        "broadcastBlock is only valid when replicated TP MLA host offload is enabled.");
    TLLM_CHECK_WITH_INFO(block->isPrimary(), "Replicated TP MLA host offload only broadcasts primary blocks.");
    TLLM_CHECK_WITH_INFO(
        mBroadcastStream != nullptr, "Missing NCCL broadcast stream for replicated TP MLA host offload.");

    // getComm(), getDtypeMap(), and the NCCL calls below are only available in multi-device builds.
#if ENABLE_MULTI_DEVICE
    TLLM_CHECK_WITH_INFO(mTpGroupRanks.size() > 1, "Replicated TP MLA host offload requires tp_size > 1.");
    auto ncclComm = ::tensorrt_llm::getComm(mTpGroupRanks);
    auto* dtypeMap = ::tensorrt_llm::getDtypeMap();
    auto const stream = mBroadcastStream->get();
    auto const rootTpRank = tpGroupRankForWorldRank(rootWorldRank);

    NCCLCHECK_THROW(ncclGroupStart());
    for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
    {
        auto blockPtr = computeBlockPointer(block, pools, poolIdx);
        auto const dtype = blockPtr->getDataType();
        auto const dtypeIt = dtypeMap->find(dtype);
        TLLM_CHECK_WITH_INFO(
            dtypeIt != dtypeMap->end(), "Unsupported NCCL broadcast dtype %d", static_cast<int>(dtype));
        NCCLCHECK_THROW(ncclBroadcast(
            blockPtr->data(), blockPtr->data(), blockPtr->getSize(), dtypeIt->second, rootTpRank, *ncclComm, stream));
    }
    NCCLCHECK_THROW(ncclGroupEnd());
#else
    TLLM_THROW("Replicated TP MLA host offload requires TensorRT-LLM to be built with multi-device support.");
#endif
}

void KVCacheTransferManager::copyBlock(BlockPtr const& src, BlockPtr const& dst,
    std::vector<KVCacheBlockPool> const& pools, bool isOffload, int numTokensToCopy, executor::KvCacheTransferMode mode,
    std::string const& directory)
{
    TLLM_LOG_DEBUG("copyBlock entered: srcId=%d, dstId=%d, isOffload=%s, mode=%d", src->getBlockId(), dst->getBlockId(),
        (isOffload ? "true" : "false"), static_cast<int>(mode));

    if (mode == executor::KvCacheTransferMode::DRAM)
    {
        TLLM_LOG_DEBUG("Using DRAM-based copy (GPU <-> CPU) for this block.");

        // Iterate over all pools, partial-copy logic
        for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
        {
            auto srcPtr = computeBlockPointer(src, pools, poolIdx);
            auto dstPtr = computeBlockPointer(dst, pools, poolIdx);

            // Does it contain block scales?
            auto containsBlockScales = pools[poolIdx].containsBlockScales;

            // If no partial tokens or if the dataType is not supported for partial copy, copy entire block.
            // Note that nvfp4 kv cache SFs use an interleaved layout, so we need to copy the entire block.
            if (numTokensToCopy <= 0 || srcPtr->getDataType() == nvinfer1::DataType::kINT4
                || srcPtr->getDataType() == nvinfer1::DataType::kFP4 || containsBlockScales)
            {
                // For partial copy not implemented with these data types,
                // just do a full copy.
                (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
            }
            else
            {
                int const tokensPerBlock = pools[poolIdx].tokensPerBlock;
                if (numTokensToCopy >= tokensPerBlock)
                {
                    // If requested tokens >= entire block, just do a full copy.
                    (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
                }
                else
                {
                    auto stream = (isOffload ? mOffloadManager : mOnboardManager).getStream().get();
                    int const numLayers = pools[poolIdx].numLayers;
                    int const kvFactor = pools[poolIdx].kvFactor;
                    int const numHeads = pools[poolIdx].numKvHeads;
                    int const sizePerHead = pools[poolIdx].sizePerHead;
                    auto shape = srcPtr->getShape();

                    TLLM_CHECK_WITH_INFO(
                        shape.nbDims == 4, "Expected KVCache block to have 4 dims, got %d", shape.nbDims);

                    tk::kvCacheBlockPartialCopy(*dstPtr, *srcPtr, numLayers, numHeads, tokensPerBlock, sizePerHead,
                        numTokensToCopy, kvFactor, stream);
                }
            }
        }

        TLLM_LOG_DEBUG("copyBlock: DRAM mode complete. Returning...");
        return;
    }

    std::vector<kvc::FileDesc> fileBlobs;
    std::vector<kvc::MemoryDesc> memoryBlobs;

    for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
    {
        auto ptr = isOffload ? computeBlockPointer(src, pools, poolIdx) : computeBlockPointer(dst, pools, poolIdx);
        auto block_id = src->getBlockId();

        TLLM_CHECK_WITH_INFO(
            !directory.empty(), "Expected a directory path for KVCache offload, but none was provided.");

        int size = std::snprintf(nullptr, 0, "%s/block_%d_pool_%zu.bin", directory.c_str(), block_id, poolIdx);
        std::string filename;
        filename.resize(size + 1);
        std::snprintf(
            filename.data(), filename.size(), "%s/block_%d_pool_%zu.bin", directory.c_str(), block_id, poolIdx);

        if (mode == executor::KvCacheTransferMode::POSIX_DEBUG_FALLBACK)
        {
            TLLM_LOG_INFO("Forcing POSIX fallback for file: %s", filename.c_str());
            if (isOffload)
            {
                gpuToFilePosix(ptr, filename);
            }
            else
            {
                fileToGpuPosix(ptr, filename);
            }
            continue;
        }
        else if (mode == executor::KvCacheTransferMode::GDS)
        {

            int openFlags = isOffload ? (O_CREAT | O_WRONLY) : O_RDONLY;
            fileBlobs.emplace_back(filename, openFlags, 0664, ptr->getSizeInBytes());
            memoryBlobs.emplace_back(ptr->data(), ptr->getSizeInBytes(), mDeviceId);
        }
    }

    if (mode == executor::KvCacheTransferMode::GDS)
    {
        if (mLoopbackAgent == nullptr)
        {
            TLLM_LOG_DEBUG("KVCacheTransferManager: creating mLoopbackAgent lazily");
            kvc::BaseAgentConfig config{std::string("GDSAgent"), true, true};
            mLoopbackAgent = kvc::makeLoopbackAgent("nixl", &config);
        }

        kvc::FileDescs fileDescs(std::move(fileBlobs));
        kvc::MemoryDescs memoryDescs(kvc::MemoryType::kVRAM, memoryBlobs);

        mLoopbackAgent->executeLoopbackRequest(memoryDescs, fileDescs, isOffload);
    }
}

//
// Note about recording events to wait for cudaMemcpyAsync calls between blocks:
// The memory copy involves raw memory blocks, which are identified by the
// memory pool block index plus the primary/secondary memory level. Using
// getBlockId() when recording events is wrong. getBlockId() returns the
// logical block id, which has nothing to do with the raw memory block
// pointers involved in a cudaMemcpy.
//

//
// Notes about need for synchronization:
//
// Relying on decoder syncing GPU with CPU to ensure that blocks are ready
// for offload/onboard/partial copy is dangerous. We have an asynchronous decoder
// that may not synchronize or synchronize at a later point in the execution stream.
// To avoid synchronization issues caused by changes to decoder design we rely on
// KVCacheTransferManager::syncWithBufferManager() that ensures that internal copy streams
// will wait for prefill and decode kernels that have already been scheduled.
//
// Earlier versions of this code did not account for all possible cases where a new block copy
// needed to wait for a previously scheduled copy to finish. For instance, it is possible
// that two primary blocks are offloaded to the same secondary block in a single step,
// scheduling the second offloading without waiting for the first one to finish leads to
// a corrupted block after offloading. It is possible that partial reuse will copy
// from a block that is currently being onboarded, scheduling the partial copy without
// waiting for the onboarding to finish will lead to a corrupted block. To handle all
// possible cases needing synchronization we record separate events for reads and writes
// to a block. When a new block copy is scheduled, we wait for all writes to the source
// block and all reads and writes to a destination block.
//
// As before, syncTransfers() must be called after last call to KVCacheManager::addSequence.
// Failing to do so will lead to corrupted blocks eventually.
//

void KVCacheTransferManager::offload(BlockPtr const& block, BlockPtr const& offloadBlock,
    std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy, executor::KvCacheTransferMode mode,
    std::string const& directory)
{
    auto const sourceKey = computePendingTransferKey(block);
    auto const destinationKey = computePendingTransferKey(offloadBlock);

    if (mEnableTpMlaReplicatedHostOffload)
    {
        validateReplicatedHostOffloadMode(mode);
        auto const ownerRank = blockMappingForSecondaryBlock(offloadBlock).ownerRank;
        if (mWorldRank != ownerRank)
        {
            // This is the dedupe point for replicated TP MLA host offload: all ranks see the replicated secondary
            // block, but only the deterministic owner rank writes the compact host-offload slot.
            return;
        }
    }

    waitForPendingWrite(sourceKey, mOffloadManager.getStream(), false);
    waitForPendingRead(destinationKey, mOffloadManager.getStream(), true);
    waitForPendingWrite(destinationKey, mOffloadManager.getStream(), true);

    copyBlock(block, offloadBlock, pools, true /* isOffload */, numTokensToCopy, mode, directory);

    recordPendingRead(sourceKey, mOffloadManager.getStream());
    recordPendingWrite(destinationKey, mOffloadManager.getStream());
}

void KVCacheTransferManager::onboard(BlockPtr const& offloadedBlock, BlockPtr const& block,
    std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy, executor::KvCacheTransferMode mode,
    std::string const& directory)
{
    auto const sourceKey = computePendingTransferKey(offloadedBlock);
    auto const destinationKey = computePendingTransferKey(block);

    if (mEnableTpMlaReplicatedHostOffload)
    {
        validateReplicatedHostOffloadMode(mode);
        TLLM_CHECK_WITH_INFO(
            mBroadcastStream != nullptr, "Missing NCCL broadcast stream for replicated TP MLA host offload.");
        auto const ownerRank = blockMappingForSecondaryBlock(offloadedBlock).ownerRank;

        if (mWorldRank == ownerRank)
        {
            waitForPendingWrite(sourceKey, mOnboardManager.getStream(), false);
            waitForPendingRead(destinationKey, mOnboardManager.getStream(), true);
            waitForPendingWrite(destinationKey, mOnboardManager.getStream(), true);

            copyBlock(offloadedBlock, block, pools, false, numTokensToCopy, mode, directory);

            // The owner rank copies host KV to its primary GPU block first; the TP broadcast stream waits on that
            // copy event before broadcasting the primary block to peer ranks.
            tr::CudaEvent copyDone;
            mOnboardManager.getStream().record(copyDone);
            recordPendingRead(sourceKey, mOnboardManager.getStream());
            mBroadcastStream->wait(copyDone);
        }
        else
        {
            // Non-owner ranks do not perform the host-to-primary copy, so they normally have no pending event for this
            // destination. These waits are still needed when local prior broadcasts/writes touched the same block.
            waitForPendingRead(destinationKey, *mBroadcastStream, true);
            waitForPendingWrite(destinationKey, *mBroadcastStream, true);
        }

        // The broadcast is the cross-rank sync point: non-owner receives cannot complete until the owner has entered
        // the same NCCL operation after its host-to-primary copy event.
        broadcastBlock(block, pools, ownerRank);
        recordPendingWrite(destinationKey, *mBroadcastStream);
        return;
    }

    waitForPendingWrite(sourceKey, mOnboardManager.getStream(), false);
    waitForPendingRead(destinationKey, mOnboardManager.getStream(), true);
    waitForPendingWrite(destinationKey, mOnboardManager.getStream(), true);

    copyBlock(offloadedBlock, block, pools, false /* isOffload */, numTokensToCopy, mode, directory);

    recordPendingRead(sourceKey, mOnboardManager.getStream());
    recordPendingWrite(destinationKey, mOnboardManager.getStream());
}

void KVCacheTransferManager::syncWithBufferManager()
{
    tr::CudaEvent readyForTransfersEvent;
    mBufferManager.getStream().record(readyForTransfersEvent);
    mOffloadManager.getStream().wait(readyForTransfersEvent);
    mOnboardManager.getStream().wait(readyForTransfersEvent);
    if (mBroadcastStream != nullptr)
    {
        mBroadcastStream->wait(readyForTransfersEvent);
    }

    // Once we synchronize, clear our list of pending transfers.
    mPendingReads.clear();
    mPendingWrites.clear();
}

void KVCacheTransferManager::syncTransfers()
{
    tr::CudaEvent offloadEvent;
    mOffloadManager.getStream().record(offloadEvent);
    mBufferManager.getStream().wait(offloadEvent);

    tr::CudaEvent onboardEvent;
    mOnboardManager.getStream().record(onboardEvent);
    mBufferManager.getStream().wait(onboardEvent);

    if (mBroadcastStream != nullptr)
    {
        tr::CudaEvent broadcastEvent;
        mBroadcastStream->record(broadcastEvent);
        mBufferManager.getStream().wait(broadcastEvent);
    }

    // Once we synchronize, clear our list of pending transfers.
    mPendingReads.clear();
    mPendingWrites.clear();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager

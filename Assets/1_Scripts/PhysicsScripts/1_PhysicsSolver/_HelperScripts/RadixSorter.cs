using Unity.Collections;
using UnityEngine;
using CollisionHandler;
using Unity.Mathematics;
using Unity.Jobs;
using Unity.Burst;
using System.Runtime.CompilerServices;

namespace HelperScripts
{
#region Data structures
public struct HashCellEntry64
{
    public ulong key;
    public int cellIndex;

    public HashCellEntry64(ulong key, int cellIndex)
    {
        this.key = key;
        this.cellIndex = cellIndex;
    }
}
#endregion
public static class RadixSorter
{
#region Cell ranges sort
    [BurstCompile]
    public struct BuildHistogramsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<HashCellEntry64> cellEntries64;
        [NativeDisableParallelForRestriction]
        public NativeArray<int> histograms;

        [ReadOnly] public int shift;
        [ReadOnly] public int chunkSize;

        public void Execute(int chunkIndex)
        {
            int start = chunkIndex * chunkSize;
            int end = math.min(start + chunkSize, cellEntries64.Length);

            int histBase = chunkIndex * 256;

            for (int i = start; i < end; i++)
            {
                int bucket = (int)((cellEntries64[i].key >> shift) & 0xFFUL);
                histograms[histBase + bucket]++;
            }
        }
    }

    [BurstCompile]
    public struct CalculateOffsetsJob : IJob
    {
        public NativeArray<int> histograms;
        [WriteOnly] public NativeArray<int> histogramsOffsets;

        [ReadOnly] public int numberOfChunks;

        public void Execute()
        {
            int currentCount = 0;
            for (int i = 0; i < 256; i++)
            for (int c = 0; c < numberOfChunks; c++)
            {
                int index = c * 256 + i;
                histogramsOffsets[index] = currentCount;
                currentCount += histograms[index];

                // Reset now to use in the BuildHistogramsJob later
                histograms[index] = 0;
            }
        }
    }

    [BurstCompile]
    public struct PopulateAuxListJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<HashCellEntry64> cellEntries64;
        [NativeDisableParallelForRestriction]
        [WriteOnly] public NativeArray<HashCellEntry64> auxCellEntries64;
        [NativeDisableParallelForRestriction]
        public NativeArray<int> histogramsOffsets;

        [ReadOnly] public int shift;
        [ReadOnly] public int chunkSize;

        public void Execute(int chunkIndex)
        {
            int start = chunkIndex * chunkSize;
            int end = math.min(start + chunkSize, cellEntries64.Length);

            int histBase = chunkIndex * 256;

            for (int i = start; i < end; i++)
            {
                int bucket = (int)((cellEntries64[i].key >> shift) & 0xFFUL);
                auxCellEntries64[histogramsOffsets[histBase + bucket]++] = cellEntries64[i];
            }
        }
    }

    public static void SortCellEntries(
        ref JobHandle handle,
        int numberOfChunks,
        int numberOfCellEntries,
        NativeList<HashCellEntry64> cellEntries64,
        NativeList<HashCellEntry64> auxCellEntries64,
        NativeArray<int> histograms,
        NativeArray<int> histogramsOffsets
    )
    {
        int chunkSize = (numberOfCellEntries + numberOfChunks - 1) / numberOfChunks;

        var input = cellEntries64;
        var output = auxCellEntries64;

        for (int shift = 0; shift < 64; shift += 8)
        {
            var buildHistograms = new BuildHistogramsJob
            {
                cellEntries64 = input.AsArray(),
                histograms = histograms,
                shift = shift,
                chunkSize = chunkSize
            };
            handle = buildHistograms.Schedule(numberOfChunks, 1, handle);

            var calculateOffsets = new CalculateOffsetsJob
            {
                histograms = histograms,
                histogramsOffsets = histogramsOffsets,
                numberOfChunks = numberOfChunks
            };
            handle = calculateOffsets.Schedule(handle);

            var populateAuxList = new PopulateAuxListJob
            {
                cellEntries64 = input.AsArray(),
                auxCellEntries64 = output.AsArray(),
                histogramsOffsets = histogramsOffsets,
                shift = shift,
                chunkSize = chunkSize
            };
            handle = populateAuxList.Schedule(numberOfChunks, 1, handle);
            // handle.Complete();

            var aux = input;
            input = output;
            output = aux;
        }
    }
#endregion

#region Pair sort
    [BurstCompile]
    public struct BuildPairHistogramsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ulong> pairEntries64;
        [NativeDisableParallelForRestriction]
        public NativeArray<int> histograms;

        [ReadOnly] public int shift;
        [ReadOnly] public int chunkSize;

        public void Execute(int chunkIndex)
        {
            int start = chunkIndex * chunkSize;
            int end = math.min(start + chunkSize, pairEntries64.Length);

            int histBase = chunkIndex * 256;

            for (int i = start; i < end; i++)
            {
                int bucket = (int)((pairEntries64[i] >> shift) & 0xFFUL);
                histograms[histBase + bucket]++;
            }
        }
    }

    [BurstCompile]
    public struct PopulatePairAuxListJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ulong> pairEntries64;
        [NativeDisableParallelForRestriction]
        [WriteOnly] public NativeArray<ulong> auxPairEntries64;
        [NativeDisableParallelForRestriction]
        public NativeArray<int> histogramsOffsets;

        [ReadOnly] public int shift;
        [ReadOnly] public int chunkSize;

        public void Execute(int chunkIndex)
        {
            int start = chunkIndex * chunkSize;
            int end = math.min(start + chunkSize, pairEntries64.Length);

            int histBase = chunkIndex * 256;

            for (int i = start; i < end; i++)
            {
                int bucket = (int)((pairEntries64[i] >> shift) & 0xFFUL);
                auxPairEntries64[histogramsOffsets[histBase + bucket]++] = pairEntries64[i];
            }
        }
    }

    public static void SortPairEntries(
        ref JobHandle handle,
        int numberOfChunks,
        int numberOfPairs,
        NativeList<ulong> pairEntries64,
        NativeList<ulong> auxPairEntries64,
        NativeArray<int> histograms,
        NativeArray<int> histogramsOffsets
    )
    {
        int chunkSize = (numberOfPairs + numberOfChunks - 1) / numberOfChunks;

        var input = pairEntries64;
        var output = auxPairEntries64;

        for (int shift = 0; shift < 64; shift += 8)
        {
            var buildHistograms = new BuildPairHistogramsJob
            {
                pairEntries64 = input.AsArray(),
                histograms = histograms,
                shift = shift,
                chunkSize = chunkSize
            };
            handle = buildHistograms.Schedule(numberOfChunks, 1, handle);

            var calculateOffsets = new CalculateOffsetsJob
            {
                histograms = histograms,
                histogramsOffsets = histogramsOffsets,
                numberOfChunks = numberOfChunks
            };
            handle = calculateOffsets.Schedule(handle);

            var populateAuxList = new PopulatePairAuxListJob
            {
                pairEntries64 = input.AsArray(),
                auxPairEntries64 = output.AsArray(),
                histogramsOffsets = histogramsOffsets,
                shift = shift,
                chunkSize = chunkSize
            };
            handle = populateAuxList.Schedule(numberOfChunks, 1, handle);

            var aux = input;
            input = output;
            output = aux;
        }
    }
#endregion
}
}

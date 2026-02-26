using Unity.Mathematics;
using UnityEngine;

public static class HelperAlgorithms
{
    // Sorts keys + verts by key (LSD radix base 256),
    // then remaps triangle vertex indices, then sorts triangles by .w.
    public static void SortVertsAndRemapTris(
        int[] keys,
        float3[] verts,
        int4[] tris)
    {
        int n = keys.Length;
        if (n <= 1) return;

        // 1) Radix sort keys + verts, but also track permutation
        var keysTmp  = new int[n];
        var vertsTmp = new float3[n];
        var idx      = new int[n];     // initial indices 0..n-1
        var idxTmp   = new int[n];

        for (int i = 0; i < n; i++)
            idx[i] = i;

        const int RADIX = 256;
        int[] count = new int[RADIX];

        int[]   inK = keys;     float3[] inV  = verts;   int[]   inIdx = idx;
        int[]   outK = keysTmp; float3[] outV = vertsTmp; int[] outIdx = idxTmp;

        for (int pass = 0; pass < 4; pass++)
        {
            // Clear counts
            for (int i = 0; i < RADIX; i++) count[i] = 0;

            int shift = pass * 8;

            // Count digit
            for (int i = 0; i < n; i++)
            {
                int mapped = inK[i] ^ unchecked((int)0x80000000);
                int digit  = (mapped >> shift) & 0xFF;
                count[digit]++;
            }

            // Prefix sum -> starting offsets
            int sum = 0;
            for (int i = 0; i < RADIX; i++)
            {
                int c = count[i];
                count[i] = sum;
                sum += c;
            }

            // Scatter (stable), including idx[]
            for (int i = 0; i < n; i++)
            {
                int k      = inK[i];
                int mapped = k ^ unchecked((int)0x80000000);
                int digit  = (mapped >> shift) & 0xFF;

                int dst   = count[digit]++;
                outK[dst] = k;
                outV[dst] = inV[i];
                outIdx[dst] = inIdx[i];
            }

            // Swap buffers
            (inK, outK)     = (outK, inK);
            (inV, outV)     = (outV, inV);
            (inIdx, outIdx) = (outIdx, inIdx);
        }

        // After 4 passes (even), sorted data ends back in original arrays:
        // keys = inK, verts = inV, idx = inIdx

        // 2) Build old->new map from idx[]
        // idx[newIndex] = oldIndex  =>  oldToNew[oldIndex] = newIndex
        int[] oldToNew = new int[n];
        for (int newIndex = 0; newIndex < n; newIndex++)
        {
            int oldIndex = inIdx[newIndex];
            oldToNew[oldIndex] = newIndex;
        }

        // 3) Remap triangle vertex indices
        for (int i = 0; i < tris.Length; i++)
        {
            int4 t = tris[i];
            t.x = oldToNew[t.x];
            t.y = oldToNew[t.y];
            t.z = oldToNew[t.z];
            tris[i] = t;
        }

        // 4) Optionally sort triangles by .w
        SortInt4ByW(tris);
    }

    // As before: radix-like sort of int4[] by .w using base 256
    public static void SortInt4ByW(int4[] tris)
    {
        int n = tris.Length;
        if (n <= 1) return;

        var tmp = new int4[n];
        const int RADIX = 256;
        int[] count = new int[RADIX];

        int4[] inArr  = tris;
        int4[] outArr = tmp;

        for (int pass = 0; pass < 4; pass++)
        {
            for (int i = 0; i < RADIX; i++) count[i] = 0;
            int shift = pass * 8;

            // Count
            for (int i = 0; i < n; i++)
            {
                int mapped = inArr[i].w ^ unchecked((int)0x80000000);
                int digit  = (mapped >> shift) & 0xFF;
                count[digit]++;
            }

            // Prefix sum
            int sum = 0;
            for (int i = 0; i < RADIX; i++)
            {
                int c = count[i];
                count[i] = sum;
                sum += c;
            }

            // Scatter
            for (int i = 0; i < n; i++)
            {
                int4 tri   = inArr[i];
                int mapped = tri.w ^ unchecked((int)0x80000000);
                int digit  = (mapped >> shift) & 0xFF;

                int dst     = count[digit]++;
                outArr[dst] = tri;
            }

            (inArr, outArr) = (outArr, inArr);
        }
        // 4 passes => even => result in tris
    }
}

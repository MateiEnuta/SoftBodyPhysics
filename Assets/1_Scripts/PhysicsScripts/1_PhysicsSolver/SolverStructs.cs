using Unity.Mathematics;
using UnityEngine;

namespace SolverStructs
{
    public struct Particle
    {
        public float3 currentPosition;
        public float3 previousPosition;
        public float invMass;
    }

    public struct LatticeCell
    {
        public int v0, v1, v2, v3, v4, v5, v6, v7;
    }

    public struct DynamicVertex
    {
        public float4 bary;
        public byte tet;

        public DynamicVertex(float4 bary, byte tet)
        {
            this.bary = bary;
            this.tet = tet;
        }
    }

    public struct CollisionWraper
    {
        public int cellIdx;
        public int2 colVertsRange;
        public int2 colTrisRange;
        public int2 colEdgesRange;

        public CollisionWraper(int cellIdx, int2 colVertsRange, int2 colTrisRange, int2 colEdgesRange)
        {
            this.cellIdx = cellIdx;
            this.colVertsRange = colVertsRange;
            this.colTrisRange = colTrisRange;
            this.colEdgesRange = colEdgesRange;
        }

        public CollisionWraper(CollisionWraper wraperCopy, int colVertsDiff, int colTriesDiff, int colEdgesDiff)
        {
            cellIdx = wraperCopy.cellIdx;
            colVertsRange = wraperCopy.colVertsRange + new int2(colVertsDiff);
            colTrisRange = wraperCopy.colTrisRange + new int2(colTriesDiff);
            colEdgesRange = wraperCopy.colEdgesRange + new int2(colEdgesDiff);
        }
    }

    public struct BodyWraper
    {
        public LatticePreprocessor latticePreprocessor;
        public int2 particlesRange;
        public int2[] cellRanges;
        public int2 colVertsRange;
        public int2 colTrisRange;
        public int2 colEdgesRange;
        public int2 colWrapersRange;
        public int2 visVerticesRange;
        public int2 visWrapersRange;

        public BodyWraper(LatticePreprocessor latticePreprocessor)
        {
            this.latticePreprocessor = latticePreprocessor;
            cellRanges = new int2[8];
            particlesRange = colVertsRange = colTrisRange = 
            colWrapersRange = colEdgesRange = visVerticesRange = 
            visWrapersRange = int2.zero;
        }
    }
}

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
    }
}

using SolverStructs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct BasicPlaneColJob : IJobParallelFor
{
    [ReadOnly] public float                        yPlanePos;
    [ReadOnly] public NativeArray<LatticeCell>     latticeCells;
    [ReadOnly] public NativeArray<DynamicVertex>   collisionVertices;
    [ReadOnly] public NativeArray<CollisionWraper> collisionWrapers;
    [NativeDisableParallelForRestriction] public NativeArray<Particle> particles;

    private static readonly int4[] s_CellTets =
    {
        new int4(0, 3, 6, 5),
        new int4(0, 5, 6, 4),
        new int4(0, 6, 3, 2),
        new int4(5, 3, 6, 7),
        new int4(0, 3, 5, 1)
    };

    public void Execute(int index)
    {
        LatticeCell currentCell = latticeCells[collisionWrapers[index].cellIdx];

        for (int i = collisionWrapers[index].colVertsRange.x; i < collisionWrapers[index].colVertsRange.y; i++)
        {
            int4 tet = s_CellTets[collisionVertices[i].tet];
            float4 bary = collisionVertices[i].bary;
            tet = new int4(
                GetCellVertIndex(currentCell, tet.x),
                GetCellVertIndex(currentCell, tet.y),
                GetCellVertIndex(currentCell, tet.z),
                GetCellVertIndex(currentCell, tet.w)
            );

            float3 colParticlePos = 
                bary.x * particles[tet.x].currentPosition + 
                bary.y * particles[tet.y].currentPosition +
                bary.z * particles[tet.z].currentPosition +
                bary.w * particles[tet.w].currentPosition;
            if (colParticlePos.y >= yPlanePos) continue;

            float3 correction = new float3(0, yPlanePos - colParticlePos.y, 0);

            // For equalizing the correction
            float denom = bary.x*bary.x + bary.y*bary.y + bary.z*bary.z + bary.w*bary.w;
            if (denom < 1e-08f) continue;
            correction /= denom;

            // Update actual particles positions
            Particle p0 = particles[tet.x];
            p0.currentPosition += bary.x * correction;
            particles[tet.x] = p0;

            Particle p1 = particles[tet.y];
            p1.currentPosition += bary.y * correction;
            particles[tet.y] = p1;

            Particle p2 = particles[tet.z];
            p2.currentPosition += bary.z * correction;
            particles[tet.z] = p2;

            Particle p3 = particles[tet.w];
            p3.currentPosition += bary.w * correction;
            particles[tet.w] = p3;
        }
    }

    private static int GetCellVertIndex(LatticeCell c, int localIndex)
    {
        switch (localIndex)
        {
            case 0: return c.v0;
            case 1: return c.v1;
            case 2: return c.v2;
            case 3: return c.v3;
            case 4: return c.v4;
            case 5: return c.v5;
            case 6: return c.v6;
            case 7: return c.v7;
            default: return -1;
        }
    }
}

using System.Collections;
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
        Particle p1 = particles[currentCell.v0];
        Particle p2 = particles[currentCell.v1];
        Particle p3 = particles[currentCell.v2];
        Particle p4 = particles[currentCell.v3];
        Particle p5 = particles[currentCell.v4];
        Particle p6 = particles[currentCell.v5];
        Particle p7 = particles[currentCell.v6];
        Particle p8 = particles[currentCell.v7];

        for (int i = collisionWrapers[index].colVertsRange.x; i < collisionWrapers[index].colVertsRange.y; i++)
        {
            int4 tet = s_CellTets[collisionVertices[i].tet];
            float4 bary = collisionVertices[i].bary;

            float3 colParticlePos = 
                bary.x * GetCellVertIndex(tet.x, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8).currentPosition + 
                bary.y * GetCellVertIndex(tet.y, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8).currentPosition +
                bary.z * GetCellVertIndex(tet.z, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8).currentPosition +
                bary.w * GetCellVertIndex(tet.w, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8).currentPosition;
            if (colParticlePos.y >= yPlanePos) continue;

            float3 correction = new float3(0, yPlanePos - colParticlePos.y, 0);

            // For equalizing the correction
            float denom = bary.x*bary.x + bary.y*bary.y + bary.z*bary.z + bary.w*bary.w;
            if (denom < 1e-08f) continue;
            correction /= denom;

            // Update actual particles positions
            GetCellVertIndex(tet.x, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8).currentPosition += bary.x * correction;
            GetCellVertIndex(tet.y, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8).currentPosition += bary.y * correction;
            GetCellVertIndex(tet.z, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8).currentPosition += bary.z * correction;
            GetCellVertIndex(tet.w, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8).currentPosition += bary.w * correction;
        }

        particles[currentCell.v0] = p1;
        particles[currentCell.v1] = p2;
        particles[currentCell.v2] = p3;
        particles[currentCell.v3] = p4;
        particles[currentCell.v4] = p5;
        particles[currentCell.v5] = p6;
        particles[currentCell.v6] = p7;
        particles[currentCell.v7] = p8;
    }

    private static ref Particle GetCellVertIndex(int localIndex, ref Particle p1, ref Particle p2, ref Particle p3, 
        ref Particle p4, ref Particle p5, ref Particle p6, ref Particle p7, ref Particle p8)
    {
        switch (localIndex)
        {
            case 0: return ref p1;
            case 1: return ref p2;
            case 2: return ref p3;
            case 3: return ref p4;
            case 4: return ref p5;
            case 5: return ref p6;
            case 6: return ref p7;
            case 7: return ref p8;
            default: return ref p1;
        }
    }
}

using SolverStructs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct VolConsJob : IJobParallelFor
{
    [ReadOnly] public float dtSq, invDt, invDtSq;
    [ReadOnly] public float damping;
    [ReadOnly] public float bigTetVol, smallTetsVol;
    [ReadOnly] public NativeSlice<LatticeCell> latticeCells;
    [ReadOnly] public NativeArray<int>         latticeCellBodyRef;
    [ReadOnly] public NativeArray<float>       bodyVolConsCompliance;
    [NativeDisableParallelForRestriction]
    public NativeArray<Particle> particles;


    static readonly int4[] s_CellTets = new[]
    {
        new int4(0, 3, 6, 5),       // new int4(0, 1, 3, 7),
        new int4(0, 5, 6, 4),       // new int4(0, 3, 2, 7),
        new int4(0, 6, 3, 2),       // new int4(0, 2, 6, 7),
        new int4(5, 3, 6, 7),       // new int4(0, 6, 4, 7),
        new int4(0, 3, 5, 1)        // new int4(0, 4, 5, 7),
                                    // new int4(0, 5, 1, 7)
    };


    public void Execute(int index)
    {
        LatticeCell cell = latticeCells[index];
        float cellCompliance = bodyVolConsCompliance[latticeCellBodyRef[index]];

        Particle p0 = particles[cell.v0];
        Particle p1 = particles[cell.v1];
        Particle p2 = particles[cell.v2];
        Particle p3 = particles[cell.v3];
        Particle p4 = particles[cell.v4];
        Particle p5 = particles[cell.v5];
        Particle p6 = particles[cell.v6];
        Particle p7 = particles[cell.v7];

        for (int i = 0; i < 5; i++)
        {
            int4 currentTet = s_CellTets[i];
            float initialTetVolume = i == 0 ? bigTetVol : smallTetsVol;

            SolveVolumeConstraint(
                ref GetCellVertIndex(currentTet.x, ref p0, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7),
                ref GetCellVertIndex(currentTet.y, ref p0, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7),
                ref GetCellVertIndex(currentTet.z, ref p0, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7),
                ref GetCellVertIndex(currentTet.w, ref p0, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7), 
                initialTetVolume, cellCompliance
            );
        }

        particles[cell.v0] = p0;
        particles[cell.v1] = p1;
        particles[cell.v2] = p2;
        particles[cell.v3] = p3;
        particles[cell.v4] = p4;
        particles[cell.v5] = p5;
        particles[cell.v6] = p6;
        particles[cell.v7] = p7;
    }

    private ref Particle GetCellVertIndex(int localIndex, ref Particle p1, ref Particle p2, ref Particle p3, 
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

    private void SolveVolumeConstraint(ref Particle p1, ref Particle p2, ref Particle p3, ref Particle p4, 
        float initialVolume, float compliance)
    {
        // Compute signed volume
        float3 v1 = p2.currentPosition - p1.currentPosition;
        float3 v2 = p3.currentPosition - p1.currentPosition;
        float3 v3 = p4.currentPosition - p1.currentPosition;
        float3 cross23 = math.cross(v2, v3);
        float currentVolume = math.dot(v1, cross23) / 6f;

        float C = currentVolume - initialVolume;

        // Gradients for tetrahedral volume constraint
        float3 cross31 = math.cross(v3, v1);
        float3 cross12 = math.cross(v1, v2);

        float3 grad1 = -(cross23 + cross31 + cross12) / 6f;
        float3 grad2 = cross23 / 6f;
        float3 grad3 = cross31 / 6f;
        float3 grad4 = cross12 / 6f;

        // Weighted gradient magnitudes
        float sumWeightedGradients = 0f;
        sumWeightedGradients += p1.invMass * math.lengthsq(grad1);
        sumWeightedGradients += p2.invMass * math.lengthsq(grad2);
        sumWeightedGradients += p3.invMass * math.lengthsq(grad3);
        sumWeightedGradients += p4.invMass * math.lengthsq(grad4);

        float alphaTilde = compliance * invDtSq;
        sumWeightedGradients += alphaTilde;

        if (math.abs(sumWeightedGradients) < 1e-10f) return;

        // -------------------------
        // XPBD damping term
        // -------------------------
        float dampingTerm = 0f;
        if (damping > 0f)
        {
            float betaTilde = damping * dtSq;
            float gamma = alphaTilde * betaTilde * invDt;

            float3 vel1 = p1.currentPosition - p1.previousPosition;
            float3 vel2 = p2.currentPosition - p2.previousPosition;
            float3 vel3 = p3.currentPosition - p3.previousPosition;
            float3 vel4 = p4.currentPosition - p4.previousPosition;

            float velocitySum =
                math.dot(grad1, vel1) +
                math.dot(grad2, vel2) +
                math.dot(grad3, vel3) +
                math.dot(grad4, vel4);

            dampingTerm = gamma * velocitySum;
            sumWeightedGradients *= 1f + gamma;
        }

        // Compute constraint correction
        float deltaLambda = -(C + dampingTerm) / sumWeightedGradients;

        // Apply position correction to all tetra points
        p1.currentPosition += p1.invMass * deltaLambda * grad1;
        p2.currentPosition += p2.invMass * deltaLambda * grad2;
        p3.currentPosition += p3.invMass * deltaLambda * grad3;
        p4.currentPosition += p4.invMass * deltaLambda * grad4;
    }
}
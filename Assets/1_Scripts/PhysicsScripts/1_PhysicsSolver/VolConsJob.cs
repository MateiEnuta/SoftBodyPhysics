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

    public void Execute(int index)
    {
        float cellCompliance = bodyVolConsCompliance[latticeCellBodyRef[index]];

        for (int i = 0; i < 5; i++)
        {
            int4 currentTet = GetCellTet(latticeCells[index], i);
            float initialTetVolume = i == 0 ? bigTetVol : smallTetsVol;

            SolveVolumeConstraint(currentTet, initialTetVolume, cellCompliance);
        }
    }

    static int4 GetCellTet(LatticeCell c, int tetIndex)
    {
        switch (tetIndex)
        {
            case 0: return new int4(c.v0, c.v3, c.v6, c.v5);
            case 1: return new int4(c.v0, c.v5, c.v6, c.v4);
            case 2: return new int4(c.v0, c.v6, c.v3, c.v2);
            case 3: return new int4(c.v5, c.v3, c.v6, c.v7);
            default:return new int4(c.v0, c.v3, c.v5, c.v1);
        }
    }

    private void SolveVolumeConstraint(int4 tet, float initialVolume, float compliance)
    {
        // Load tetra particles
        Particle p1 = particles[tet.x];
        Particle p2 = particles[tet.y];
        Particle p3 = particles[tet.z];
        Particle p4 = particles[tet.w];

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

        // Write back
        particles[tet.x] = p1;
        particles[tet.y] = p2;
        particles[tet.z] = p3;
        particles[tet.w] = p4;
    }
}
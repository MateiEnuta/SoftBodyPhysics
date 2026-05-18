using SolverStructs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct DistConsJob : IJobParallelFor
{
    [ReadOnly] public float dtSq, invDt, invDtSq;
    [ReadOnly] public float damping;
    [ReadOnly] public float edgeLen, faceDiagLen, bodyDiagLen;
    [ReadOnly] public NativeSlice<LatticeCell> latticeCells;
    [ReadOnly] public NativeArray<int>         latticeCellBodyRef;
    [ReadOnly] public NativeArray<float>       bodyDistConsCompliance;
    [NativeDisableParallelForRestriction]
    public NativeArray<Particle> particles;


    public void Execute(int index)
    {
        float cellCompliance = bodyDistConsCompliance[latticeCellBodyRef[index]];

        for (int i = 0; i < 7; i++)
        for (int j = i+1; j < 8; j++)
        {
            int e1 = GetCellVertex(latticeCells[index], i);
            int e2 = GetCellVertex(latticeCells[index], j);

            SolveDistanceConstraint(e1, e2, GetRestLength(i, j), cellCompliance);
        }
    }

    static int GetCellVertex(LatticeCell c, int i)
    {
        switch (i)
        {
            case 0: return c.v0;
            case 1: return c.v1;
            case 2: return c.v2;
            case 3: return c.v3;
            case 4: return c.v4;
            case 5: return c.v5;
            case 6: return c.v6;
            default:return c.v7;
        }
    }

    private float GetRestLength(int a, int b)
    {
        int diff = a ^ b;
        int axisCount = (diff & 1) + ((diff >> 1) & 1) + ((diff >> 2) & 1);

        return axisCount == 1 ? edgeLen :
            axisCount == 2 ? faceDiagLen :
                                bodyDiagLen;
    }

    private void SolveDistanceConstraint(int e1Idx, int e2Idx, float initialLength, float compliance)
    {
        Particle e1 = particles[e1Idx];
        Particle e2 = particles[e2Idx];

        // Calculate current length
        float3 dir = e2.currentPosition - e1.currentPosition;
        float currentLength = math.length(dir);
        if (currentLength < 1e-11) return;

        float C = currentLength - initialLength;

        float3 n = dir / currentLength;
        float3 grad1 = -n;
        float3 grad2 = n;

        // Sum is just the inv masses and the compliance term
        float sumWeightedGradients = e1.invMass + e2.invMass;
        float alphaTilde = compliance * invDtSq;
        sumWeightedGradients += alphaTilde;
        if (math.abs(sumWeightedGradients) < 1e-10f) return;

        float dampingTerm = 0f;
        if (damping > 0f)
        {
            float betaTilde = damping * dtSq;
            float gamma = alphaTilde * betaTilde * invDt;

            // Constraint-space velocity
            float3 v1 = e1.currentPosition - e1.previousPosition;
            float3 v2 = e2.currentPosition - e2.previousPosition;
            float relativeVelocity = math.dot(v2 - v1, grad2);

            dampingTerm = gamma * relativeVelocity;

            // Modify denominator for stable XPBD damping
            sumWeightedGradients *= 1f + gamma;
        }

        // XPBD correction
        float deltaLambda = -(C + dampingTerm) / sumWeightedGradients;

        // Apply positional correction
        e1.currentPosition += e1.invMass * deltaLambda * grad1;
        e2.currentPosition += e2.invMass * deltaLambda * grad2;

        // Write back
        particles[e1Idx] = e1;
        particles[e2Idx] = e2;
    }
}
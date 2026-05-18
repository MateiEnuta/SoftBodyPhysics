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
        LatticeCell cell = latticeCells[index];
        float cellCompliance = bodyDistConsCompliance[latticeCellBodyRef[index]];

        Particle p0 = particles[cell.v0];
        Particle p1 = particles[cell.v1];
        Particle p2 = particles[cell.v2];
        Particle p3 = particles[cell.v3];
        Particle p4 = particles[cell.v4];
        Particle p5 = particles[cell.v5];
        Particle p6 = particles[cell.v6];
        Particle p7 = particles[cell.v7];

        for (int a = 0; a < 7; a++)
        for (int b = a + 1; b < 8; b++)
        {
            SolveDistanceConstraint(
                ref GetCellVertIndex(a, ref p0, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7), 
                ref GetCellVertIndex(b, ref p0, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7), 
                GetRestLength(a, b), 
                cellCompliance
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

    private float GetRestLength(int a, int b)
    {
        int diff = a ^ b;
        int axisCount = (diff & 1) + ((diff >> 1) & 1) + ((diff >> 2) & 1);

        return axisCount == 1 ? edgeLen :
            axisCount == 2 ? faceDiagLen :
                                bodyDiagLen;
    }

    private void SolveDistanceConstraint(ref Particle e1, ref Particle e2, float initialLength, float compliance)
    {
        // Calculate current length
        float3 dir = e2.currentPosition - e1.currentPosition;
        float currentLength = math.length(dir);
        if (currentLength < 1e-11) return;

        // Sum is just the inv masses and the compliance term
        float sumWeightedGradients = e1.invMass + e2.invMass;
        float alphaTilde = compliance * invDtSq;
        sumWeightedGradients += alphaTilde;
        if (math.abs(sumWeightedGradients) < 1e-10f) return;

        float C = currentLength - initialLength;
        float3 n = dir / currentLength;

        float dampingTerm = 0f;
        if (damping > 0f)
        {
            float betaTilde = damping * dtSq;
            float gamma = alphaTilde * betaTilde * invDt;

            // Constraint-space velocity
            float3 v1 = e1.currentPosition - e1.previousPosition;
            float3 v2 = e2.currentPosition - e2.previousPosition;
            float relativeVelocity = math.dot(v2 - v1, n);

            dampingTerm = gamma * relativeVelocity;

            // Modify denominator for stable XPBD damping
            sumWeightedGradients *= 1f + gamma;
        }

        // XPBD correction
        float deltaLambda = -(C + dampingTerm) / sumWeightedGradients;

        // Apply positional correction
        e1.currentPosition += e1.invMass * deltaLambda * -n;
        e2.currentPosition += e2.invMass * deltaLambda * n;
    }
}
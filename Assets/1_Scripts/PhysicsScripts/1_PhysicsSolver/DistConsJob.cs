using SolverStructs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.SocialPlatforms;

[BurstCompile]
public struct DistConsJob : IJobParallelFor
{
    [ReadOnly] public float invDt, invDtSq, dtSq;
    [ReadOnly] public float damping;
    [ReadOnly] public float edgeLen, faceDiagLen, bodyDiagLen;
    [ReadOnly] public NativeSlice<int2>   latticeCells;
    [ReadOnly] public NativeArray<int2>   bodyParticleResolution;
    [ReadOnly] public NativeArray<float>  bodyDistConsCompliance;
    [NativeDisableParallelForRestriction]
    public NativeArray<Particle> particles;


    public void Execute(int index)
    {
        int2 cell = latticeCells[index];
        int2 res = bodyParticleResolution[cell.x];
        float cellCompliance = bodyDistConsCompliance[cell.x];

        int sx = res.x;
        int sxy = res.x * res.y;

        Particle p0 = particles[cell.y];
        Particle p1 = particles[cell.y + 1];
        Particle p2 = particles[cell.y + sx];
        Particle p3 = particles[cell.y + sx + 1];
        Particle p4 = particles[cell.y + sxy];
        Particle p5 = particles[cell.y + sxy + 1];
        Particle p6 = particles[cell.y + sxy + sx];
        Particle p7 = particles[cell.y + sxy + sx + 1];

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

        particles[cell.y] = p0;
        particles[cell.y + 1] = p1;
        particles[cell.y + sx] = p2;
        particles[cell.y + sx + 1] = p3;
        particles[cell.y + sxy] = p4;
        particles[cell.y + sxy + 1] = p5;
        particles[cell.y + sxy + sx] = p6;
        particles[cell.y + sxy + sx + 1] = p7;
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

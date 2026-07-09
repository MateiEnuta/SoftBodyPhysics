using SolverStructs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct ShapeMatchingJob : IJobParallelFor
{
    [ReadOnly] public float dt;
    [ReadOnly] public float damping;
    [ReadOnly] public int channelStart;
    [ReadOnly] public float halfD;
    [ReadOnly] public float3x3 AqqInverse;
    [ReadOnly] public NativeArray<LatticeCell> latticeCells;
    [ReadOnly] public NativeArray<int>         latticeCellBodyRef;
    [ReadOnly] public NativeArray<float>       bodyShpMatchCompliance;
    [NativeDisableParallelForRestriction] public NativeArray<Particle> particles;
    
    public void Execute(int index)
    {
        index += channelStart;

        LatticeCell cell = latticeCells[index];
        int v0 = cell.v0; int v1 = cell.v1; int v2 = cell.v2; int v3 = cell.v3;
        int v4 = cell.v4; int v5 = cell.v5; int v6 = cell.v6; int v7 = cell.v7;

        // 1. Current COM
        float3 currentCOM = float3.zero;
        currentCOM += particles[v0].currentPosition;
        currentCOM += particles[v1].currentPosition;
        currentCOM += particles[v2].currentPosition;
        currentCOM += particles[v3].currentPosition;
        currentCOM += particles[v4].currentPosition;
        currentCOM += particles[v5].currentPosition;
        currentCOM += particles[v6].currentPosition;
        currentCOM += particles[v7].currentPosition;
        currentCOM *= 1f/8f;

        // 2. Build A_pq = sum p_i * q_i^T
        float3x3 Apq = ShapeMatchingMath.Zero3x3();
        Apq += ShapeMatchingMath.OuterProduct(particles[v0].currentPosition - currentCOM, new float3(-halfD, -halfD, -halfD));
        Apq += ShapeMatchingMath.OuterProduct(particles[v1].currentPosition - currentCOM, new float3( halfD, -halfD, -halfD));
        Apq += ShapeMatchingMath.OuterProduct(particles[v2].currentPosition - currentCOM, new float3(-halfD,  halfD, -halfD));
        Apq += ShapeMatchingMath.OuterProduct(particles[v3].currentPosition - currentCOM, new float3( halfD,  halfD, -halfD));
        Apq += ShapeMatchingMath.OuterProduct(particles[v4].currentPosition - currentCOM, new float3(-halfD, -halfD,  halfD));
        Apq += ShapeMatchingMath.OuterProduct(particles[v5].currentPosition - currentCOM, new float3( halfD, -halfD,  halfD));
        Apq += ShapeMatchingMath.OuterProduct(particles[v6].currentPosition - currentCOM, new float3(-halfD,  halfD,  halfD));
        Apq += ShapeMatchingMath.OuterProduct(particles[v7].currentPosition - currentCOM, new float3( halfD,  halfD,  halfD));


        // 3. Extract target T from A
        float3x3 A = math.mul(Apq, AqqInverse);
        float3x3 R = ShapeMatchingMath.ExtractRotation(A);

        int bodyRef = latticeCellBodyRef[index];
        float compliance = bodyShpMatchCompliance[bodyRef];
        float3x3 T = compliance * A + (1f - compliance) * R;

        // 4. Solve the constraint
        float3 goalV0 = math.mul(T, new float3(-halfD, -halfD, -halfD)) + currentCOM;
        float3 goalV1 = math.mul(T, new float3( halfD, -halfD, -halfD)) + currentCOM;
        float3 goalV2 = math.mul(T, new float3(-halfD,  halfD, -halfD)) + currentCOM;
        float3 goalV3 = math.mul(T, new float3( halfD,  halfD, -halfD)) + currentCOM;
        float3 goalV4 = math.mul(T, new float3(-halfD, -halfD,  halfD)) + currentCOM;
        float3 goalV5 = math.mul(T, new float3( halfD, -halfD,  halfD)) + currentCOM;
        float3 goalV6 = math.mul(T, new float3(-halfD,  halfD,  halfD)) + currentCOM;
        float3 goalV7 = math.mul(T, new float3( halfD,  halfD,  halfD)) + currentCOM;


        UpdateParticlePos(v0, goalV0);
        UpdateParticlePos(v1, goalV1);
        UpdateParticlePos(v2, goalV2);
        UpdateParticlePos(v3, goalV3);
        UpdateParticlePos(v4, goalV4);
        UpdateParticlePos(v5, goalV5);
        UpdateParticlePos(v6, goalV6);
        UpdateParticlePos(v7, goalV7);
    }

    private void UpdateParticlePos(int particleIdx, float3 goalPos)
    {
        var p = particles[particleIdx];
        p.currentPosition = goalPos;
        particles[particleIdx] = p;
    }
}


public static class ShapeMatchingMath
{
    public static float3x3 ExtractRotation(float3x3 A, int iterations = 10)
    {
        float3x3 R = A;

        for (int i = 0; i < iterations; i++)
        {
            float det = Determinant(R);
            if (math.abs(det) < 1e-8f)
            {
                // Degenerate case: fall back to identity or previous valid rotation
                return float3x3.identity;
            }

            float3x3 RInvT = math.transpose(math.inverse(R));
            R = Mul(0.5f, Add(R, RInvT));
        }

        // Optional: enforce proper rotation (determinant > 0)
        if (Determinant(R) < 0.0f)
        {
            // Flip one axis to avoid reflection
            R.c2 = -R.c2;
        }

        return R;
    }

    public static float3x3 Add(float3x3 a, float3x3 b)
    {
        return new float3x3(
            a.c0 + b.c0,
            a.c1 + b.c1,
            a.c2 + b.c2
        );
    }

    public static float3x3 Mul(float s, float3x3 m)
    {
        return new float3x3(
            m.c0 * s,
            m.c1 * s,
            m.c2 * s
        );
    }

    public static float Determinant(float3x3 m)
    {
        return math.dot(m.c0, math.cross(m.c1, m.c2));
    }

    // Helper funcitons
    public static float3x3 Zero3x3()
    {
        return new float3x3(
            0f, 0f, 0f,
            0f, 0f, 0f,
            0f, 0f, 0f
        );
    }
    public static float3x3 OuterProduct(float3 a, float3 b)
    {
        return new float3x3(
            a * b.x,
            a * b.y,
            a * b.z
        );
    }
}
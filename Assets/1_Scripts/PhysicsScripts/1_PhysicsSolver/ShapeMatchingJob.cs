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
    [ReadOnly] public NativeArray<LatticeCell> latticeCells;
    [ReadOnly] public NativeArray<int>         latticeCellBodyRef;
    [ReadOnly] public NativeArray<float>       bodyDeviConsCompliance;
    [ReadOnly] public NativeArray<float>       bodyVolConsCompliance;
    [NativeDisableParallelForRestriction] public NativeArray<Particle> particles;
    
    public void Execute(int index)
    {
        index += channelStart;

        LatticeCell cell = latticeCells[index];

        int v0 = cell.v0; int v1 = cell.v1; int v2 = cell.v2; int v3 = cell.v3;
        int v4 = cell.v4; int v5 = cell.v5; int v6 = cell.v6; int v7 = cell.v7;
        float3 initPosV0 = new float3(-halfD, -halfD, -halfD);
        float3 initPosV1 = new float3( halfD, -halfD, -halfD);
        float3 initPosV2 = new float3(-halfD,  halfD, -halfD);
        float3 initPosV3 = new float3( halfD,  halfD, -halfD);
        float3 initPosV4 = new float3(-halfD, -halfD,  halfD);
        float3 initPosV5 = new float3( halfD, -halfD,  halfD);
        float3 initPosV6 = new float3(-halfD,  halfD,  halfD);
        float3 initPosV7 = new float3( halfD,  halfD,  halfD);

        float deviatoricCompliance = bodyDeviConsCompliance[latticeCellBodyRef[index]];
        float volumeCompliance = bodyVolConsCompliance[latticeCellBodyRef[index]];

        SolveTetNeoHookean(
            v0, v3, v6, v5,
            initPosV0,
            initPosV3,
            initPosV6,
            initPosV5,
            deviatoricCompliance,
            volumeCompliance
        );

        SolveTetNeoHookean(
            v0, v5, v6, v4,
            initPosV0,
            initPosV5,
            initPosV6,
            initPosV4,
            deviatoricCompliance,
            volumeCompliance
        );

        SolveTetNeoHookean(
            v0, v6, v3, v2,
            initPosV0,
            initPosV6,
            initPosV3,
            initPosV2,
            deviatoricCompliance,
            volumeCompliance
        );

        SolveTetNeoHookean(
            v5, v3, v6, v7,
            initPosV5,
            initPosV3,
            initPosV6,
            initPosV7,
            deviatoricCompliance,
            volumeCompliance
        );

        SolveTetNeoHookean(
            v0, v3, v5, v1,
            initPosV0,
            initPosV3,
            initPosV5,
            initPosV1,
            deviatoricCompliance,
            volumeCompliance
        );
    }

    private void SolveTetNeoHookean(
        int v0, int v1, int v2, int v3,
        float3 initialX0,
        float3 initialX1,
        float3 initialX2,
        float3 initialX3,
        float deviatoricCompliance,
        float volumeCompliance
    )
    {
        float3x3 invDs = math.inverse(new float3x3(
            initialX0 - initialX3,
            initialX1 - initialX3,
            initialX2 - initialX3
        ));

        SolveDeviatoricConstraint(v0, v1, v2, v3, deviatoricCompliance, invDs);
        SolveVolumeConstraint(v0, v1, v2, v3, volumeCompliance, invDs);
    }

    private void SolveVolumeConstraint(
        int v0, int v1, int v2, int v3,
        float volumeCompliance,
        float3x3 invDs
    )
    {
        float3x3 X = new float3x3(
            particles[v0].currentPosition - particles[v3].currentPosition,
            particles[v1].currentPosition - particles[v3].currentPosition,
            particles[v2].currentPosition - particles[v3].currentPosition
        );
        float3x3 F = math.mul(X, invDs);

        float3 f1 = F.c0;
        float3 f2 = F.c1;
        float3 f3 = F.c2;

        float C = math.determinant(F) - 1f;

        float3x3 H = new float3x3(
            math.cross(f2, f3),
            math.cross(f3, f1),
            math.cross(f1, f2)
        );

        float3x3 G = math.mul(H, math.transpose(invDs));

        float3 g0 = G.c0;
        float3 g1 = G.c1;
        float3 g2 = G.c2;
        float3 g3 = -(g0 + g1 + g2);

        SolveScalarXPBD(v0, v1, v2, v3, C, g0, g1, g2, g3, volumeCompliance);
    }

    private void SolveDeviatoricConstraint(
        int v0, int v1, int v2, int v3,
        float deviatoricCompliance,
        float3x3 invDs
    )
    {
        float3x3 X = new float3x3(
            particles[v0].currentPosition - particles[v3].currentPosition,
            particles[v1].currentPosition - particles[v3].currentPosition,
            particles[v2].currentPosition - particles[v3].currentPosition
        );
        float3x3 F = math.mul(X, invDs);

        float3 f1 = F.c0;
        float3 f2 = F.c1;
        float3 f3 = F.c2;

        float r = math.sqrt(
            math.lengthsq(f1) +
            math.lengthsq(f2) +
            math.lengthsq(f3));
        r = math.max(r, 1e-8f);

        float C = r - math.sqrt(3f);

        float3x3 H = new float3x3(
            f1 / r,
            f2 / r,
            f3 / r
        );

        float3x3 G = math.mul(H, math.transpose(invDs));

        float3 g0 = G.c0;
        float3 g1 = G.c1;
        float3 g2 = G.c2;
        float3 g3 = -(g0 + g1 + g2);

        SolveScalarXPBD(v0, v1, v2, v3, C, g0, g1, g2, g3, deviatoricCompliance);
    }

    private void SolveScalarXPBD(
        int v0, int v1, int v2, int v3,
        float C,
        float3 g0,
        float3 g1,
        float3 g2,
        float3 g3,
        float compliance
    )
    {
        float w0 = particles[v0].invMass;
        float w1 = particles[v1].invMass;
        float w2 = particles[v2].invMass;
        float w3 = particles[v3].invMass;

        float gradTerm =
            w0 * math.lengthsq(g0) +
            w1 * math.lengthsq(g1) +
            w2 * math.lengthsq(g2) +
            w3 * math.lengthsq(g3);

        float alpha = compliance / (dt * dt);

        if (gradTerm + alpha < 1e-8f)
            return;

        float denom = gradTerm + alpha;
        float dampingTerm = 0f;

        if (damping > 0f)
        {
            float3 vel0 = (particles[v0].currentPosition - particles[v0].previousPosition) / dt;
            float3 vel1 = (particles[v1].currentPosition - particles[v1].previousPosition) / dt;
            float3 vel2 = (particles[v2].currentPosition - particles[v2].previousPosition) / dt;
            float3 vel3 = (particles[v3].currentPosition - particles[v3].previousPosition) / dt;

            float constraintVelocity =
                math.dot(g0, vel0) +
                math.dot(g1, vel1) +
                math.dot(g2, vel2) +
                math.dot(g3, vel3);

            dampingTerm = damping * constraintVelocity;
            denom += damping * gradTerm;
        }

        float deltaLambda = -(C + dampingTerm) / denom;

        UpdateParticlePos(v0, w0 * deltaLambda * g0);
        UpdateParticlePos(v1, w1 * deltaLambda * g1);
        UpdateParticlePos(v2, w2 * deltaLambda * g2);
        UpdateParticlePos(v3, w3 * deltaLambda * g3);
    }

    private void UpdateParticlePos(int particleIdx, float3 delta)
    {
        var p = particles[particleIdx];
        p.currentPosition += delta;
        particles[particleIdx] = p;
    }
}
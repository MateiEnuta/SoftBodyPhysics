using SolverStructs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct DistConsJob : IJobParallelFor
{
    [ReadOnly] public float dt;
    [ReadOnly] public float damping;
    [ReadOnly] public int channelStart;
    [ReadOnly] public float latticeDimensions;
    [ReadOnly] public NativeArray<LatticeCell> latticeCells;
    [ReadOnly] public NativeArray<int>         latticeCellBodyRef;
    [ReadOnly] public NativeArray<float>       bodyDistConsCompliance;
    [ReadOnly] public NativeArray<float>       bodyVolConsCompliance;
    [NativeDisableParallelForRestriction] public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        index += channelStart;

        LatticeCell cell = latticeCells[index];
        int bodyRef = latticeCellBodyRef[index];

        float distCompliance = bodyDistConsCompliance[bodyRef];
        float volCompliance  = bodyVolConsCompliance[bodyRef];

        float dtSq = dt * dt;
        float alphaDist = distCompliance / dtSq;
        float alphaVol  = volCompliance / dtSq;

        float gammaDist = 0f;
        float gammaVol  = 0f;
        bool useDamping = damping > 0f;
        if (useDamping)
        {
            float betaTilde = damping * dtSq;
            gammaDist = alphaDist * betaTilde / dt;
            gammaVol  = alphaVol  * betaTilde / dt;
        }

        int v0 = cell.v0; int v1 = cell.v1; int v2 = cell.v2; int v3 = cell.v3;
        int v4 = cell.v4; int v5 = cell.v5; int v6 = cell.v6; int v7 = cell.v7;

        float dx = latticeDimensions, dy = latticeDimensions, dz = latticeDimensions;
        float dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;

        float lenX   = dx;
        float lenY   = dy;
        float lenZ   = dz;
        float lenXY  = math.sqrt(dx2 + dy2);
        float lenXZ  = math.sqrt(dx2 + dz2);
        float lenYZ  = math.sqrt(dy2 + dz2);
        float lenXYZ = math.sqrt(dx2 + dy2 + dz2);

        SolveDistanceConstraint(v0, v1, lenX,   alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v0, v2, lenY,   alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v0, v3, lenXY,  alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v0, v4, lenZ,   alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v0, v5, lenXZ,  alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v0, v6, lenYZ,  alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v0, v7, lenXYZ, alphaDist, gammaDist, useDamping);

        SolveDistanceConstraint(v1, v2, lenXY,  alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v1, v3, lenY,   alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v1, v4, lenXZ,  alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v1, v5, lenZ,   alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v1, v6, lenXYZ, alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v1, v7, lenYZ,  alphaDist, gammaDist, useDamping);

        SolveDistanceConstraint(v2, v3, lenX,   alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v2, v4, lenYZ,  alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v2, v5, lenXYZ, alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v2, v6, lenZ,   alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v2, v7, lenXZ,  alphaDist, gammaDist, useDamping);

        SolveDistanceConstraint(v3, v4, lenXYZ, alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v3, v5, lenYZ,  alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v3, v6, lenXZ,  alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v3, v7, lenZ,   alphaDist, gammaDist, useDamping);

        SolveDistanceConstraint(v4, v5, lenX,   alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v4, v6, lenY,   alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v4, v7, lenXY,  alphaDist, gammaDist, useDamping);

        SolveDistanceConstraint(v5, v6, lenXY,  alphaDist, gammaDist, useDamping);
        SolveDistanceConstraint(v5, v7, lenY,   alphaDist, gammaDist, useDamping);

        SolveDistanceConstraint(v6, v7, lenX,   alphaDist, gammaDist, useDamping);

        float cellVolume = dx * dy * dz;
        float bigTetVol = cellVolume * (1f / 3f);
        float smallTetVol = cellVolume * (1f / 6f);

        SolveVolumeConstraint(new int4(v0, v3, v6, v5), bigTetVol,   alphaVol, gammaVol, useDamping);
        SolveVolumeConstraint(new int4(v0, v5, v6, v4), smallTetVol, alphaVol, gammaVol, useDamping);
        SolveVolumeConstraint(new int4(v0, v6, v3, v2), smallTetVol, alphaVol, gammaVol, useDamping);
        SolveVolumeConstraint(new int4(v5, v3, v6, v7), smallTetVol, alphaVol, gammaVol, useDamping);
        SolveVolumeConstraint(new int4(v0, v3, v5, v1), smallTetVol, alphaVol, gammaVol, useDamping);
    }

    private void SolveDistanceConstraint(int e1Idx, int e2Idx, float initialLength, float alphaTilde, float gamma, bool useDamping)
    {
        Particle e1 = particles[e1Idx];
        Particle e2 = particles[e2Idx];

        float3 dir = e2.currentPosition - e1.currentPosition;
        float currentLengthSq = math.lengthsq(dir);
        if (currentLengthSq < 1e-22f) return;

        float currentLength = math.sqrt(currentLengthSq);
        float C = currentLength - initialLength;

        float invLength = 1f / currentLength;
        float3 n = dir * invLength;

        float sumWeightedGradients = e1.invMass + e2.invMass + alphaTilde;
        if (sumWeightedGradients < 1e-10f) return;

        float dampingTerm = 0f;
        if (useDamping)
        {
            float3 v1 = e1.currentPosition - e1.previousPosition;
            float3 v2 = e2.currentPosition - e2.previousPosition;
            float relativeVelocity = math.dot(v2 - v1, n);

            dampingTerm = gamma * relativeVelocity;
            sumWeightedGradients *= 1f + gamma;
        }

        float deltaLambda = -(C + dampingTerm) / sumWeightedGradients;
        float3 corr = deltaLambda * n;

        e1.currentPosition -= e1.invMass * corr;
        e2.currentPosition += e2.invMass * corr;

        particles[e1Idx] = e1;
        particles[e2Idx] = e2;
    }

    private void SolveVolumeConstraint(int4 tet, float initialVolume, float alphaTilde, float gamma, bool useDamping)
    {
        Particle p1 = particles[tet.x];
        Particle p2 = particles[tet.y];
        Particle p3 = particles[tet.z];
        Particle p4 = particles[tet.w];

        float3 v1 = p2.currentPosition - p1.currentPosition;
        float3 v2 = p3.currentPosition - p1.currentPosition;
        float3 v3 = p4.currentPosition - p1.currentPosition;

        float3 cross23 = math.cross(v2, v3);
        float currentVolume = math.dot(v1, cross23) * (1f / 6f);
        float C = currentVolume - initialVolume;

        float3 cross31 = math.cross(v3, v1);
        float3 cross12 = math.cross(v1, v2);

        const float oneSixth = 1f / 6f;
        float3 grad1 = -(cross23 + cross31 + cross12) * oneSixth;
        float3 grad2 = cross23 * oneSixth;
        float3 grad3 = cross31 * oneSixth;
        float3 grad4 = cross12 * oneSixth;

        float sumWeightedGradients =
            p1.invMass * math.lengthsq(grad1) +
            p2.invMass * math.lengthsq(grad2) +
            p3.invMass * math.lengthsq(grad3) +
            p4.invMass * math.lengthsq(grad4) +
            alphaTilde;

        if (sumWeightedGradients < 1e-10f) return;

        float dampingTerm = 0f;
        if (useDamping)
        {
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

        float deltaLambda = -(C + dampingTerm) / sumWeightedGradients;

        p1.currentPosition += p1.invMass * deltaLambda * grad1;
        p2.currentPosition += p2.invMass * deltaLambda * grad2;
        p3.currentPosition += p3.invMass * deltaLambda * grad3;
        p4.currentPosition += p4.invMass * deltaLambda * grad4;

        particles[tet.x] = p1;
        particles[tet.y] = p2;
        particles[tet.z] = p3;
        particles[tet.w] = p4;
    }
}
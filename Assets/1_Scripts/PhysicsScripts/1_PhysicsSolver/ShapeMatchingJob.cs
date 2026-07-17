using System.Runtime.CompilerServices;
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
        // SolveShapeMatching(index);

        LatticeCell cell = latticeCells[index];
        int v0 = cell.v0; int v1 = cell.v1; int v2 = cell.v2; int v3 = cell.v3;
        int v4 = cell.v4; int v5 = cell.v5; int v6 = cell.v6; int v7 = cell.v7;
        
        float3 q0 = new float3(-halfD, -halfD, -halfD);
        float3 q1 = new float3( halfD, -halfD, -halfD);
        float3 q2 = new float3(-halfD,  halfD, -halfD);
        float3 q3 = new float3( halfD,  halfD, -halfD);
        float3 q4 = new float3(-halfD, -halfD,  halfD);
        float3 q5 = new float3( halfD, -halfD,  halfD);
        float3 q6 = new float3(-halfD,  halfD,  halfD);
        float3 q7 = new float3( halfD,  halfD,  halfD);

        float3 x0 = particles[v0].currentPosition;
        float3 x1 = particles[v1].currentPosition;
        float3 x2 = particles[v2].currentPosition;
        float3 x3 = particles[v3].currentPosition;
        float3 x4 = particles[v4].currentPosition;
        float3 x5 = particles[v5].currentPosition;
        float3 x6 = particles[v6].currentPosition;
        float3 x7 = particles[v7].currentPosition;

        float3 prevX0 = particles[v0].previousPosition;
        float3 prevX1 = particles[v1].previousPosition;
        float3 prevX2 = particles[v2].previousPosition;
        float3 prevX3 = particles[v3].previousPosition;
        float3 prevX4 = particles[v4].previousPosition;
        float3 prevX5 = particles[v5].previousPosition;
        float3 prevX6 = particles[v6].previousPosition;
        float3 prevX7 = particles[v7].previousPosition;

        float wX0 = particles[v0].invMass;
        float wX1 = particles[v1].invMass;
        float wX2 = particles[v2].invMass;
        float wX3 = particles[v3].invMass;
        float wX4 = particles[v4].invMass;
        float wX5 = particles[v5].invMass;
        float wX6 = particles[v6].invMass;
        float wX7 = particles[v7].invMass;

        float dt2 = dt*dt;
        float alphaStretch = bodyDeviConsCompliance[latticeCellBodyRef[index]] / dt2;
        float alphaShear = bodyDeviConsCompliance[latticeCellBodyRef[index]] *2f / dt2;
        float alphaVolume = bodyVolConsCompliance[latticeCellBodyRef[index]] / dt2;

        SolveTetNeoHookean(
            q0, q3, q6, q5,
            ref x0, ref x3, ref x6, ref x5,
            prevX0, prevX3, prevX6, prevX5,
               wX0,    wX3,    wX6,    wX5,
            alphaStretch,
            alphaShear,
            alphaVolume
        );

        SolveTetNeoHookean(
            q0, q5, q6, q4,
            ref x0, ref x5, ref x6, ref x4,
            prevX0, prevX5, prevX6, prevX4,
               wX0,    wX5,    wX6,    wX4,
            alphaStretch,
            alphaShear,
            alphaVolume
        );

        SolveTetNeoHookean(
            q0, q6, q3, q2,
            ref x0, ref x6, ref x3, ref x2,
            prevX0, prevX6, prevX3, prevX2,
               wX0,    wX6,    wX3,    wX2,
            alphaStretch,
            alphaShear,
            alphaVolume
        );

        SolveTetNeoHookean(
            q5, q3, q6, q7,
            ref x5, ref x3, ref x6, ref x7,
            prevX5, prevX3, prevX6, prevX7,
               wX5,    wX3,    wX6,    wX7,
            alphaStretch,
            alphaShear,
            alphaVolume
        );

        SolveTetNeoHookean(
            q0, q3, q5, q1,
            ref x0, ref x3, ref x5, ref x1,
            prevX0, prevX3, prevX5, prevX1,
               wX0,    wX3,    wX5,    wX1,
            alphaStretch,
            alphaShear,
            alphaVolume
        );

        Particle p;
        p = particles[v0];
        p.currentPosition = x0;
        particles[v0] = p;
        p = particles[v1];
        p.currentPosition = x1;
        particles[v1] = p;
        p = particles[v2];
        p.currentPosition = x2;
        particles[v2] = p;
        p = particles[v3];
        p.currentPosition = x3;
        particles[v3] = p;
        p = particles[v4];
        p.currentPosition = x4;
        particles[v4] = p;
        p = particles[v5];
        p.currentPosition = x5;
        particles[v5] = p;
        p = particles[v6];
        p.currentPosition = x6;
        particles[v6] = p;
        p = particles[v7];
        p.currentPosition = x7;
        particles[v7] = p;
    }

#region Basic Shape Matching
    private void SolveShapeMatching(int index)
    {
        LatticeCell cell = latticeCells[index];
        int v0 = cell.v0; int v1 = cell.v1; int v2 = cell.v2; int v3 = cell.v3;
        int v4 = cell.v4; int v5 = cell.v5; int v6 = cell.v6; int v7 = cell.v7;
        float3 q0 = new float3(-halfD, -halfD, -halfD);
        float3 q1 = new float3( halfD, -halfD, -halfD);
        float3 q2 = new float3(-halfD,  halfD, -halfD);
        float3 q3 = new float3( halfD,  halfD, -halfD);
        float3 q4 = new float3(-halfD, -halfD,  halfD);
        float3 q5 = new float3( halfD, -halfD,  halfD);
        float3 q6 = new float3(-halfD,  halfD,  halfD);
        float3 q7 = new float3( halfD,  halfD,  halfD);

        float3 x0 = particles[v0].currentPosition;
        float3 x1 = particles[v1].currentPosition;
        float3 x2 = particles[v2].currentPosition;
        float3 x3 = particles[v3].currentPosition;
        float3 x4 = particles[v4].currentPosition;
        float3 x5 = particles[v5].currentPosition;
        float3 x6 = particles[v6].currentPosition;
        float3 x7 = particles[v7].currentPosition;

        // 1. Current COM
        float3 currentCOM = (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7) * 0.125f;

        // 2. Build A_pq = sum p_i * q_i^T and A_qq = sum q_i * q_i^T
        float3x3 Apq = ShapeMatchingMath.Zero3x3();
        Apq += ShapeMatchingMath.OuterProduct(x0 - currentCOM, q0);
        Apq += ShapeMatchingMath.OuterProduct(x1 - currentCOM, q1);
        Apq += ShapeMatchingMath.OuterProduct(x2 - currentCOM, q2);
        Apq += ShapeMatchingMath.OuterProduct(x3 - currentCOM, q3);
        Apq += ShapeMatchingMath.OuterProduct(x4 - currentCOM, q4);
        Apq += ShapeMatchingMath.OuterProduct(x5 - currentCOM, q5);
        Apq += ShapeMatchingMath.OuterProduct(x6 - currentCOM, q6);
        Apq += ShapeMatchingMath.OuterProduct(x7 - currentCOM, q7);

        float3x3 Aqq = ShapeMatchingMath.Zero3x3();
        Aqq += ShapeMatchingMath.OuterProduct(q0, q0);
        Aqq += ShapeMatchingMath.OuterProduct(q1, q1);
        Aqq += ShapeMatchingMath.OuterProduct(q2, q2);
        Aqq += ShapeMatchingMath.OuterProduct(q3, q3);
        Aqq += ShapeMatchingMath.OuterProduct(q4, q4);
        Aqq += ShapeMatchingMath.OuterProduct(q5, q5);
        Aqq += ShapeMatchingMath.OuterProduct(q6, q6);
        Aqq += ShapeMatchingMath.OuterProduct(q7, q7);

        float detAqq = math.determinant(Aqq);
        if (math.abs(detAqq) < 1e-9f)
            return;

        float3x3 AqqInverse = math.inverse(Aqq);


        // 3. Extract target T from A
        float3x3 A = math.mul(Apq, AqqInverse);
        float3x3 R = ShapeMatchingMath.ExtractRotation(A);

        int bodyRef = latticeCellBodyRef[index];
        float compliance = 0f;
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


        SetParticlePos(v0, goalV0);
        SetParticlePos(v1, goalV1);
        SetParticlePos(v2, goalV2);
        SetParticlePos(v3, goalV3);
        SetParticlePos(v4, goalV4);
        SetParticlePos(v5, goalV5);
        SetParticlePos(v6, goalV6);
        SetParticlePos(v7, goalV7);
    }
#endregion

#region Neo-Hookean
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SolveTetNeoHookean(
        float3 q0, float3 q1, float3 q2, float3 q3,
        ref float3 x0, ref float3 x1,
        ref float3 x2, ref float3 x3,
        float3 prevX0, float3 prevX1,
        float3 prevX2, float3 prevX3,
        float wX0, float wX1,
        float wX2, float wX3,
        float alphaStretch,
        float alphaShear,
        float alphaVolume
    )
    {
        float3x3 Dm = new float3x3(
            q1 - q0,
            q2 - q0,
            q3 - q0
        );
        float detDm = math.determinant(Dm);
        if (math.abs(detDm) < 1e-9f)
            return;
        float3x3 invDm = math.inverse(Dm);

        float3 d0 = invDm.c0;
        float3 d1 = invDm.c1;
        float3 d2 = invDm.c2;

        float restJ;
        if (alphaStretch <= 0f)
        {
            restJ = 1f;
        }
        else
        {
            restJ = 1f + alphaVolume / alphaStretch;
        }

        float3x3 Ds = new float3x3(
            x1 - x0,
            x2 - x0,
            x3 - x0
        );
        float3x3 F = math.mul(Ds, invDm);

        float3 gD0 = 0, gD1 = 0, gD2 = 0, gD3 = 0;
        float3 gH0 = 0, gH1 = 0, gH2 = 0, gH3 = 0;
        float cD = 0f;
        float cH = 0f;

        SolveNeoHookeanDeviatoric(
            F.c0, F.c1, F.c2,
            d0, d1, d2,
            ref cD,
            ref gD0, ref gD1, ref gD2, ref gD3
        );

        SolveNeoHookeanHydrostatic(
            F.c0, F.c1, F.c2,
            d0, d1, d2,
            restJ,
            ref cH,
            ref gH0, ref gH1, ref gH2, ref gH3
        );

        float gammaDampingDev = alphaStretch * damping * dt;
        float gammaDampingVol = alphaVolume * damping * dt;

        SolveNeoHookeanCoupled(
            ref x0, ref x1, ref x2, ref x3,
            prevX0, prevX1, prevX2, prevX3,
            wX0, wX1, wX2, wX3,
            gD0, gD1, gD2, gD3,
            gH0, gH1, gH2, gH3,
            cD, cH,
            alphaStretch, alphaVolume,
            gammaDampingDev, gammaDampingVol
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SolveNeoHookeanCoupled(
        ref float3 x0, ref float3 x1, ref float3 x2, ref float3 x3,
        float3 prevX0, float3 prevX1, float3 prevX2, float3 prevX3,
        float w0, float w1, float w2, float w3,
        float3 gD0, float3 gD1, float3 gD2, float3 gD3,
        float3 gH0, float3 gH1, float3 gH2, float3 gH3,
        float cD, float cH,
        float alphaD, float alphaH,
        float gammaD, // = alphaD * damping * dt, precomputed like your old code
        float gammaH  // = alphaH * damping * dt
    )
    {
        float k11 =
            w0 * math.lengthsq(gD0) + w1 * math.lengthsq(gD1) +
            w2 * math.lengthsq(gD2) + w3 * math.lengthsq(gD3) + alphaD;

        float k22 =
            w0 * math.lengthsq(gH0) + w1 * math.lengthsq(gH1) +
            w2 * math.lengthsq(gH2) + w3 * math.lengthsq(gH3) + alphaH;

        float k12 =
            w0 * math.dot(gD0, gH0) + w1 * math.dot(gD1, gH1) +
            w2 * math.dot(gD2, gH2) + w3 * math.dot(gD3, gH3);

        // Velocities, exactly as in your original damping code: no /dt division,
        // the /dt is folded into gammaD/gammaH already (same convention you used).
        float3 vel0 = x0 - prevX0;
        float3 vel1 = x1 - prevX1;
        float3 vel2 = x2 - prevX2;
        float3 vel3 = x3 - prevX3;

        float velD =
            math.dot(gD0, vel0) + math.dot(gD1, vel1) +
            math.dot(gD2, vel2) + math.dot(gD3, vel3);

        float velH =
            math.dot(gH0, vel0) + math.dot(gH1, vel1) +
            math.dot(gH2, vel2) + math.dot(gH3, vel3);

        // Same shape as your original: C + gamma * velocitySum in the numerator,
        // (1 + gamma) scaling the stiffness in the denominator.
        float rhsD = -(cD + gammaD * velD);
        float rhsH = -(cH + gammaH * velH);

        float k11d = k11 * (1f + gammaD);
        float k22d = k22 * (1f + gammaH);
        // k12 left unscaled: the coupling comes purely from shared gradients,
        // damping is treated as an intrinsic per-constraint property.

        float det = k11d * k22d - k12 * k12;
        if (math.abs(det) < 1e-12f) return;

        float dLambdaD = (rhsD * k22d - rhsH * k12) / det;
        float dLambdaH = (rhsH * k11d - rhsD * k12) / det;

        x0 += w0 * (dLambdaD * gD0 + dLambdaH * gH0);
        x1 += w1 * (dLambdaD * gD1 + dLambdaH * gH1);
        x2 += w2 * (dLambdaD * gD2 + dLambdaH * gH2);
        x3 += w3 * (dLambdaD * gD3 + dLambdaH * gH3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SolveNeoHookeanDeviatoric(
        float3 f0, float3 f1, float3 f2,
        float3 d0, float3 d1, float3 d2,
        ref float C,
        ref float3 g0,
        ref float3 g1,
        ref float3 g2,
        ref float3 g3
    )
    {
        float normF2 =
            math.lengthsq(f0) +
            math.lengthsq(f1) +
            math.lengthsq(f2);

        if (normF2 < 1e-16f)
            return;

        float normF = math.sqrt(normF2);
        float invNormF = 1f / normF;

        C = normF - math.sqrt(3f);

        float3 p0 = f0 * invNormF;
        float3 p1 = f1 * invNormF;
        float3 p2 = f2 * invNormF;

        // F = Ds * invDm, so gradients wrt x1, x2, x3 use ROWS of invDm.
        g1 = d0.x * p0 + d1.x * p1 + d2.x * p2;
        g2 = d0.y * p0 + d1.y * p1 + d2.y * p2;
        g3 = d0.z * p0 + d1.z * p1 + d2.z * p2;
        g0 = -(g1 + g2 + g3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SolveNeoHookeanHydrostatic(
        float3 f0, float3 f1, float3 f2,
        float3 d0, float3 d1, float3 d2,
        float restJ,
        ref float C,
        ref float3 g0,
        ref float3 g1,
        ref float3 g2,
        ref float3 g3
    )
    {
        float3 cof0 = math.cross(f1, f2);
        float3 cof1 = math.cross(f2, f0);
        float3 cof2 = math.cross(f0, f1);

        float J = math.dot(f0, cof0);

        // C = J - restJ;
        C = J - 1f;

        g1 = d0.x * cof0 + d1.x * cof1 + d2.x * cof2;
        g2 = d0.y * cof0 + d1.y * cof1 + d2.y * cof2;
        g3 = d0.z * cof0 + d1.z * cof1 + d2.z * cof2;
        g0 = -(g1 + g2 + g3);
    }
#endregion

#region Strain-Based
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SolveTetStrainVolume(
        float3 q0, float3 q1, float3 q2, float3 q3,
        ref float3 x0, ref float3 x1,
        ref float3 x2, ref float3 x3,
        float3 prevX0, float3 prevX1,
        float3 prevX2, float3 prevX3,
        float wX0, float wX1,
        float wX2, float wX3,
        float alphaStretch,
        float alphaShear,
        float alphaVolume
    )
    {
        float3x3 Dm = new float3x3(
            q1 - q0,
            q2 - q0,
            q3 - q0
        );
        float detDm = math.determinant(Dm);
        if (math.abs(detDm) < 1e-9f)
            return;
        float3x3 invDm = math.inverse(Dm);

        float3x3 Ds = new float3x3(
            x1 - x0,
            x2 - x0,
            x3 - x0
        );
        float3x3 F = math.mul(Ds, invDm);

        float3 f0 = F.c0;
        float3 f1 = F.c1;
        float3 f2 = F.c2;

        float3 d0 = invDm.c0;
        float3 d1 = invDm.c1;
        float3 d2 = invDm.c2;

        SolveStretch(
            ref x0, ref x1, ref x2, ref x3,
            prevX0, prevX1, prevX2, prevX3,
            wX0, wX1, wX2, wX3,
            f0, d0, alphaStretch
        );
        SolveStretch(
            ref x0, ref x1, ref x2, ref x3,
            prevX0, prevX1, prevX2, prevX3,
            wX0, wX1, wX2, wX3,
            f1, d1, alphaStretch
        );
        SolveStretch(
            ref x0, ref x1, ref x2, ref x3,
            prevX0, prevX1, prevX2, prevX3,
            wX0, wX1, wX2, wX3,
            f2, d2, alphaStretch
        );

        SolveShear(
            ref x0, ref x1, ref x2, ref x3,
            prevX0, prevX1, prevX2, prevX3,
            wX0, wX1, wX2, wX3,
            f0, f1, d0, d1, alphaShear
        );
        SolveShear(
            ref x0, ref x1, ref x2, ref x3,
            prevX0, prevX1, prevX2, prevX3,
            wX0, wX1, wX2, wX3,
            f0, f2, d0, d2, alphaShear
        );
        SolveShear(
            ref x0, ref x1, ref x2, ref x3,
            prevX0, prevX1, prevX2, prevX3,
            wX0, wX1, wX2, wX3,
            f1, f2, d1, d2, alphaShear
        );

        Ds = new float3x3(
            x1 - x0,
            x2 - x0,
            x3 - x0
        );
        F = math.mul(Ds, invDm);

        SolveVolume(
            ref x0, ref x1, ref x2, ref x3,
            prevX0, prevX1, prevX2, prevX3,
            wX0, wX1, wX2, wX3,
            F, d0, d1, d2,
            alphaVolume
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SolveStretch(
        ref float3 x0, ref float3 x1,
        ref float3 x2, ref float3 x3,
        float3 prevX0, float3 prevX1,
        float3 prevX2, float3 prevX3,
        float w0, float w1, float w2, float w3,
        float3 f,
        float3 d,
        float alpha
    )
    {
        float lenSq = math.lengthsq(f);
        if (lenSq < 1e-16f)
            return;

        float len = math.sqrt(lenSq);
        float invLen = 1f / len;

        float C = len - 1f;
        float3 n = f * invLen;

        float3 g1 = d.x * n;
        float3 g2 = d.y * n;
        float3 g3 = d.z * n;
        float3 g0 = -(g1 + g2 + g3);

        SolveScalarXPBD(
            ref x0, ref x1, ref x2, ref x3,
            prevX0, prevX1, prevX2, prevX3,
            w0, w1, w2, w3,
            C, g0, g1, g2, g3, alpha
        );
    }

    private void SolveShear(
        ref float3 x0, ref float3 x1,
        ref float3 x2, ref float3 x3,
        float3 prevX0, float3 prevX1,
        float3 prevX2, float3 prevX3,
        float w0, float w1, float w2, float w3,
        float3 fi,
        float3 fj,
        float3 di,
        float3 dj,
        float alpha
    )
    {
        float C = math.dot(fi, fj);

        float3 g1 = di.x * fj + dj.x * fi;
        float3 g2 = di.y * fj + dj.y * fi;
        float3 g3 = di.z * fj + dj.z * fi;
        float3 g0 = -(g1 + g2 + g3);

        SolveScalarXPBD(
            ref x0, ref x1, ref x2, ref x3,
            prevX0, prevX1, prevX2, prevX3,
            w0, w1, w2, w3,
            C, g0, g1, g2, g3, alpha
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SolveVolume(
        ref float3 x0, ref float3 x1,
        ref float3 x2, ref float3 x3,
        float3 prevX0, float3 prevX1,
        float3 prevX2, float3 prevX3,
        float w0, float w1, float w2, float w3,
        float3x3 F,
        float3 d0,
        float3 d1,
        float3 d2,
        float alpha
    )
    {
        float3 cof0 = math.cross(F.c1, F.c2);
        float3 cof1 = math.cross(F.c2, F.c0);
        float3 cof2 = math.cross(F.c0, F.c1);

        float C = math.dot(F.c0, cof0) - 1f;

        float3 g1 = d0.x * cof0 + d1.x * cof1 + d2.x * cof2;
        float3 g2 = d0.y * cof0 + d1.y * cof1 + d2.y * cof2;
        float3 g3 = d0.z * cof0 + d1.z * cof1 + d2.z * cof2;
        float3 g0 = -(g1 + g2 + g3);

        SolveScalarXPBD(
            ref x0, ref x1, ref x2, ref x3,
            prevX0, prevX1, prevX2, prevX3,
            w0, w1, w2, w3,
            C, g0, g1, g2, g3, alpha
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SolveScalarXPBD(
        ref float3 x0, ref float3 x1,
        ref float3 x2, ref float3 x3,
        float3 prevX0, float3 prevX1,
        float3 prevX2, float3 prevX3,
        float w0, float w1, float w2, float w3,
        float C,
        float3 g0, float3 g1,
        float3 g2, float3 g3,
        float alpha
    )
    {
        float denom =
            w0 * math.lengthsq(g0) +
            w1 * math.lengthsq(g1) +
            w2 * math.lengthsq(g2) +
            w3 * math.lengthsq(g3) +
            alpha;
        if (denom < 1e-8f)
            return;

        float dampingTerm = 0f;
        if (damping > 0f)
        {
            float betaTilde = damping * dt * dt;
            float gamma = alpha * betaTilde / dt;


            float3 vel1 = x0 - prevX0;
            float3 vel2 = x1 - prevX1;
            float3 vel3 = x2 - prevX2;
            float3 vel4 = x3 - prevX3;


            float velocitySum =
            math.dot(g0, vel1) +
            math.dot(g1, vel2) +
            math.dot(g2, vel3) +
            math.dot(g3, vel4);


            dampingTerm = gamma * velocitySum;
            denom *= 1f + gamma;
        }

        float deltaLambda = -(C + dampingTerm) / denom;

        x0 += w0 * deltaLambda * g0;
        x1 += w1 * deltaLambda * g1;
        x2 += w2 * deltaLambda * g2;
        x3 += w3 * deltaLambda * g3;
    }
#endregion

#region Helper Functions
    private void UpdateParticlePos(int particleIdx, float3 delta)
    {
        var p = particles[particleIdx];
        p.currentPosition += delta;
        particles[particleIdx] = p;
    }

    private void SetParticlePos(int particleIdx, float3 goal)
    {
        var p = particles[particleIdx];
        p.currentPosition = goal;
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
#endregion

using System.Runtime.CompilerServices;
using SolverStructs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct TetDampingJob : IJobParallelFor
{
    [ReadOnly] public float dt;
    [ReadOnly] public float damping;
    [ReadOnly] public int channelStart;
    [ReadOnly] public NativeArray<LatticeCell> latticeCells;
    [NativeDisableParallelForRestriction] public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        index += channelStart;

        LatticeCell cell = latticeCells[index];
        int v0 = cell.v0; int v1 = cell.v1; int v2 = cell.v2; int v3 = cell.v3;
        int v4 = cell.v4; int v5 = cell.v5; int v6 = cell.v6; int v7 = cell.v7;

        float3 x0 = particles[v0].currentPosition;
        float3 x1 = particles[v1].currentPosition;
        float3 x2 = particles[v2].currentPosition;
        float3 x3 = particles[v3].currentPosition;
        float3 x4 = particles[v4].currentPosition;
        float3 x5 = particles[v5].currentPosition;
        float3 x6 = particles[v6].currentPosition;
        float3 x7 = particles[v7].currentPosition;

        // Fresh copies, untouched by any tet's damping within this cell.
        float3 origPrevX0 = particles[v0].previousPosition;
        float3 origPrevX1 = particles[v1].previousPosition;
        float3 origPrevX2 = particles[v2].previousPosition;
        float3 origPrevX3 = particles[v3].previousPosition;
        float3 origPrevX4 = particles[v4].previousPosition;
        float3 origPrevX5 = particles[v5].previousPosition;
        float3 origPrevX6 = particles[v6].previousPosition;
        float3 origPrevX7 = particles[v7].previousPosition;

        float w0 = particles[v0].invMass;
        float w1 = particles[v1].invMass;
        float w2 = particles[v2].invMass;
        float w3 = particles[v3].invMass;
        float w4 = particles[v4].invMass;
        float w5 = particles[v5].invMass;
        float w6 = particles[v6].invMass;
        float w7 = particles[v7].invMass;

        ApplyCellInternalDamping(x0, x1, x2, x3, x4, x5, x6, x7, 
            ref origPrevX0, ref origPrevX1, ref origPrevX2, ref origPrevX3, ref origPrevX4, ref origPrevX5, ref origPrevX6, ref origPrevX7, 
            w0, w1, w2, w3, w4, w5, w6, w7, damping, dt
        );

        // Accumulate NEW previousPosition contributions and a count per vertex.
        // float3 accum0 = 0, accum1 = 0, accum2 = 0, accum3 = 0;
        // float3 accum4 = 0, accum5 = 0, accum6 = 0, accum7 = 0;
        // int n0 = 0, n1 = 0, n2 = 0, n3 = 0, n4 = 0, n5 = 0, n6 = 0, n7 = 0;

        // RunTet(x0, x3, x6, x5, origPrevX0, origPrevX3, origPrevX6, origPrevX5,
        //     m0, m3, m6, m5, ref accum0, ref accum3, ref accum6, ref accum5,
        //     ref n0, ref n3, ref n6, ref n5);

        // RunTet(x0, x5, x6, x4, origPrevX0, origPrevX5, origPrevX6, origPrevX4,
        //     m0, m5, m6, m4, ref accum0, ref accum5, ref accum6, ref accum4,
        //     ref n0, ref n5, ref n6, ref n4);

        // RunTet(x0, x6, x3, x2, origPrevX0, origPrevX6, origPrevX3, origPrevX2,
        //     m0, m6, m3, m2, ref accum0, ref accum6, ref accum3, ref accum2,
        //     ref n0, ref n6, ref n3, ref n2);

        // RunTet(x5, x3, x6, x7, origPrevX5, origPrevX3, origPrevX6, origPrevX7,
        //     m5, m3, m6, m7, ref accum5, ref accum3, ref accum6, ref accum7,
        //     ref n5, ref n3, ref n6, ref n7);

        // RunTet(x0, x3, x5, x1, origPrevX0, origPrevX3, origPrevX5, origPrevX1,
        //     m0, m3, m5, m1, ref accum0, ref accum3, ref accum5, ref accum1,
        //     ref n0, ref n3, ref n5, ref n1);

        SetPreviousParticlePos(v0, origPrevX0);
        SetPreviousParticlePos(v1, origPrevX1);
        SetPreviousParticlePos(v2, origPrevX2);
        SetPreviousParticlePos(v3, origPrevX3);
        SetPreviousParticlePos(v4, origPrevX4);
        SetPreviousParticlePos(v5, origPrevX5);
        SetPreviousParticlePos(v6, origPrevX6);
        SetPreviousParticlePos(v7, origPrevX7);
    }

#region Per cell approach
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float3x3 InertiaContribution(float3 r, float mass)
    {
        float3x3 outer = new float3x3(
            r * r.x,
            r * r.y,
            r * r.z
        );

        return mass * (float3x3.identity * math.lengthsq(r) - outer);
    }

    private static void ApplyCellInternalDamping(
        float3 x0, float3 x1, float3 x2, float3 x3,
        float3 x4, float3 x5, float3 x6, float3 x7,
        ref float3 prevX0, ref float3 prevX1,
        ref float3 prevX2, ref float3 prevX3,
        ref float3 prevX4, ref float3 prevX5,
        ref float3 prevX6, ref float3 prevX7,
        float invMass0, float invMass1,
        float invMass2, float invMass3,
        float invMass4, float invMass5,
        float invMass6, float invMass7,
        float dampingPerSecond,
        float dt)
    {
        // Static particles contribute no dynamic mass and are not modified.
        float m0 = invMass0 > 0f ? 1f / invMass0 : 0f;
        float m1 = invMass1 > 0f ? 1f / invMass1 : 0f;
        float m2 = invMass2 > 0f ? 1f / invMass2 : 0f;
        float m3 = invMass3 > 0f ? 1f / invMass3 : 0f;
        float m4 = invMass4 > 0f ? 1f / invMass4 : 0f;
        float m5 = invMass5 > 0f ? 1f / invMass5 : 0f;
        float m6 = invMass6 > 0f ? 1f / invMass6 : 0f;
        float m7 = invMass7 > 0f ? 1f / invMass7 : 0f;

        float massSum = m0 + m1 + m2 + m3 + m4 + m5 + m6 + m7;
        if (massSum < 1e-10f)
            return;

        // u = h * v: implicit displacement encoded by current/previous positions.
        float3 u0 = x0 - prevX0;
        float3 u1 = x1 - prevX1;
        float3 u2 = x2 - prevX2;
        float3 u3 = x3 - prevX3;
        float3 u4 = x4 - prevX4;
        float3 u5 = x5 - prevX5;
        float3 u6 = x6 - prevX6;
        float3 u7 = x7 - prevX7;

        // Center of mass and center-of-mass displacement.
        float3 xcm =
            (m0 * x0 + m1 * x1 + m2 * x2 + m3 * x3 +
            m4 * x4 + m5 * x5 + m6 * x6 + m7 * x7) / massSum;

        float3 ucm =
            (m0 * u0 + m1 * u1 + m2 * u2 + m3 * u3 +
            m4 * u4 + m5 * u5 + m6 * u6 + m7 * u7) / massSum;

        float3 r0 = x0 - xcm;
        float3 r1 = x1 - xcm;
        float3 r2 = x2 - xcm;
        float3 r3 = x3 - xcm;
        float3 r4 = x4 - xcm;
        float3 r5 = x5 - xcm;
        float3 r6 = x6 - xcm;
        float3 r7 = x7 - xcm;

        // Angular momentum, with displacement u = h*v instead of velocity.
        float3 angularDisplacement =
            math.cross(r0, m0 * u0) +
            math.cross(r1, m1 * u1) +
            math.cross(r2, m2 * u2) +
            math.cross(r3, m3 * u3) +
            math.cross(r4, m4 * u4) +
            math.cross(r5, m5 * u5) +
            math.cross(r6, m6 * u6) +
            math.cross(r7, m7 * u7);

        float3x3 inertia =
            InertiaContribution(r0, m0) +
            InertiaContribution(r1, m1) +
            InertiaContribution(r2, m2) +
            InertiaContribution(r3, m3) +
            InertiaContribution(r4, m4) +
            InertiaContribution(r5, m5) +
            InertiaContribution(r6, m6) +
            InertiaContribution(r7, m7);

        float detI = math.determinant(inertia);
        if (!math.isfinite(detI) || math.abs(detI) < 1e-10f)
            return;

        // This is h * omega, not omega itself, because u = h*v.
        float3 angularStep = math.mul(math.inverse(inertia), angularDisplacement);

        // Best rigid displacement field for every particle.
        float3 rigidU0 = ucm + math.cross(angularStep, r0);
        float3 rigidU1 = ucm + math.cross(angularStep, r1);
        float3 rigidU2 = ucm + math.cross(angularStep, r2);
        float3 rigidU3 = ucm + math.cross(angularStep, r3);
        float3 rigidU4 = ucm + math.cross(angularStep, r4);
        float3 rigidU5 = ucm + math.cross(angularStep, r5);
        float3 rigidU6 = ucm + math.cross(angularStep, r6);
        float3 rigidU7 = ucm + math.cross(angularStep, r7);

        // Frame/substep-independent damping fraction.
        //
        // 0       = no internal damping.
        // 1-3     = mild settling.
        // 5-12    = strong rubber/jelly settling.
        // 20+     = aggressively removes deformation velocity.
        float blend = 1f - math.exp(-math.max(0f, dampingPerSecond) * dt);

        // Blend actual implicit displacement toward rigid displacement.
        // This preserves COM translation and the cell's rigid rotation.
        float3 dampedU0 = math.lerp(u0, rigidU0, blend);
        float3 dampedU1 = math.lerp(u1, rigidU1, blend);
        float3 dampedU2 = math.lerp(u2, rigidU2, blend);
        float3 dampedU3 = math.lerp(u3, rigidU3, blend);
        float3 dampedU4 = math.lerp(u4, rigidU4, blend);
        float3 dampedU5 = math.lerp(u5, rigidU5, blend);
        float3 dampedU6 = math.lerp(u6, rigidU6, blend);
        float3 dampedU7 = math.lerp(u7, rigidU7, blend);

        if (invMass0 > 0f) prevX0 = x0 - dampedU0;
        if (invMass1 > 0f) prevX1 = x1 - dampedU1;
        if (invMass2 > 0f) prevX2 = x2 - dampedU2;
        if (invMass3 > 0f) prevX3 = x3 - dampedU3;
        if (invMass4 > 0f) prevX4 = x4 - dampedU4;
        if (invMass5 > 0f) prevX5 = x5 - dampedU5;
        if (invMass6 > 0f) prevX6 = x6 - dampedU6;
        if (invMass7 > 0f) prevX7 = x7 - dampedU7;
    }
#endregion

    private void SetPreviousParticlePos(int particleIdx, float3 goal)
    {
        var p = particles[particleIdx];
        p.previousPosition = goal;
        particles[particleIdx] = p;
    }

    void RunTet(
        float3 tx0, float3 tx1, float3 tx2, float3 tx3,
        float3 tp0, float3 tp1, float3 tp2, float3 tp3,
        float tm0, float tm1, float tm2, float tm3,
        ref float3 a0, ref float3 a1, ref float3 a2, ref float3 a3,
        ref int c0, ref int c1, ref int c2, ref int c3)
    {
        ApplyGroupDampingPositional(
            tx0, tx1, tx2, tx3,
            ref tp0, ref tp1, ref tp2, ref tp3,
            tm0, tm1, tm2, tm3,
            damping, dt
        );
        a0 += tp0; a1 += tp1; a2 += tp2; a3 += tp3;
        c0++; c1++; c2++; c3++;
    }

    private void ApplyGroupDampingPositional(
        float3 x0, float3 x1, float3 x2, float3 x3,
        ref float3 prevX0, ref float3 prevX1,
        ref float3 prevX2, ref float3 prevX3,
        float m0, float m1, float m2, float m3,
        float dampingCoefficient,
        float h
    )
    {
        float massSum = m0 + m1 + m2 + m3;
        if (massSum < 1e-10f) return;

        float3 xcm = (m0 * x0 + m1 * x1 + m2 * x2 + m3 * x3) / massSum;

        // Mass-weighted average of (x - prevX) stands in for vcm * h.
        float3 pcm =
            (m0 * prevX0 + m1 * prevX1 + m2 * prevX2 + m3 * prevX3) / massSum;

        float3 r0 = x0 - xcm, r1 = x1 - xcm, r2 = x2 - xcm, r3 = x3 - xcm;
        float3 d0 = (x0 - prevX0) - (xcm - pcm);
        float3 d1 = (x1 - prevX1) - (xcm - pcm);
        float3 d2 = (x2 - prevX2) - (xcm - pcm);
        float3 d3 = (x3 - prevX3) - (xcm - pcm);

        float3 L =
            math.cross(r0, m0 * d0) + math.cross(r1, m1 * d1) +
            math.cross(r2, m2 * d2) + math.cross(r3, m3 * d3);

        float3x3 InertiaFromR(float3 r, float m)
        {
            float3x3 outer = new float3x3(r * r.x, r * r.y, r * r.z);
            return m * (float3x3.identity * math.lengthsq(r) - outer);
        }

        float3x3 I =
            InertiaFromR(r0, m0) + InertiaFromR(r1, m1) +
            InertiaFromR(r2, m2) + InertiaFromR(r3, m3);

        float detI = math.determinant(I);
        if (math.abs(detI) < 1e-9f) return;

        float3 omega = math.mul(math.inverse(I), L); // already divided by h implicitly below

        float3 rigidDisp0 = (xcm - pcm) + math.cross(omega, r0);
        float3 rigidDisp1 = (xcm - pcm) + math.cross(omega, r1);
        float3 rigidDisp2 = (xcm - pcm) + math.cross(omega, r2);
        float3 rigidDisp3 = (xcm - pcm) + math.cross(omega, r3);

        float blend = math.min(dampingCoefficient * h, 1f);

        prevX0 = x0 - ((x0 - prevX0) + blend * (rigidDisp0 - (x0 - prevX0)));
        prevX1 = x1 - ((x1 - prevX1) + blend * (rigidDisp1 - (x1 - prevX1)));
        prevX2 = x2 - ((x2 - prevX2) + blend * (rigidDisp2 - (x2 - prevX2)));
        prevX3 = x3 - ((x3 - prevX3) + blend * (rigidDisp3 - (x3 - prevX3)));
    }
}

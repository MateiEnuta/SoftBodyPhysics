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

        float m0 = 1f / particles[v0].invMass;
        float m1 = 1f / particles[v1].invMass;
        float m2 = 1f / particles[v2].invMass;
        float m3 = 1f / particles[v3].invMass;
        float m4 = 1f / particles[v4].invMass;
        float m5 = 1f / particles[v5].invMass;
        float m6 = 1f / particles[v6].invMass;
        float m7 = 1f / particles[v7].invMass;

        // Accumulate NEW previousPosition contributions and a count per vertex.
        float3 accum0 = 0, accum1 = 0, accum2 = 0, accum3 = 0;
        float3 accum4 = 0, accum5 = 0, accum6 = 0, accum7 = 0;
        int n0 = 0, n1 = 0, n2 = 0, n3 = 0, n4 = 0, n5 = 0, n6 = 0, n7 = 0;

        RunTet(x0, x3, x6, x5, origPrevX0, origPrevX3, origPrevX6, origPrevX5,
            m0, m3, m6, m5, ref accum0, ref accum3, ref accum6, ref accum5,
            ref n0, ref n3, ref n6, ref n5);

        RunTet(x0, x5, x6, x4, origPrevX0, origPrevX5, origPrevX6, origPrevX4,
            m0, m5, m6, m4, ref accum0, ref accum5, ref accum6, ref accum4,
            ref n0, ref n5, ref n6, ref n4);

        RunTet(x0, x6, x3, x2, origPrevX0, origPrevX6, origPrevX3, origPrevX2,
            m0, m6, m3, m2, ref accum0, ref accum6, ref accum3, ref accum2,
            ref n0, ref n6, ref n3, ref n2);

        RunTet(x5, x3, x6, x7, origPrevX5, origPrevX3, origPrevX6, origPrevX7,
            m5, m3, m6, m7, ref accum5, ref accum3, ref accum6, ref accum7,
            ref n5, ref n3, ref n6, ref n7);

        RunTet(x0, x3, x5, x1, origPrevX0, origPrevX3, origPrevX5, origPrevX1,
            m0, m3, m5, m1, ref accum0, ref accum3, ref accum5, ref accum1,
            ref n0, ref n3, ref n5, ref n1);

        SetPreviousParticlePos(v0, accum0 / math.max(n0, 1));
        SetPreviousParticlePos(v1, accum1 / math.max(n1, 1));
        SetPreviousParticlePos(v2, accum2 / math.max(n2, 1));
        SetPreviousParticlePos(v3, accum3 / math.max(n3, 1));
        SetPreviousParticlePos(v4, accum4 / math.max(n4, 1));
        SetPreviousParticlePos(v5, accum5 / math.max(n5, 1));
        SetPreviousParticlePos(v6, accum6 / math.max(n6, 1));
        SetPreviousParticlePos(v7, accum7 / math.max(n7, 1));
    }

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

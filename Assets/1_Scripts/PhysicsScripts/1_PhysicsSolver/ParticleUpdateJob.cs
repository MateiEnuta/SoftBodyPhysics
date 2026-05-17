using SolverStructs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct ParticleUpdateJob : IJobParallelFor
{
    [ReadOnly] public float3 gravityDt;
    [ReadOnly] public float dt, invDt;
    [ReadOnly] public NativeArray<int> particleMask;
    [NativeDisableParallelForRestriction]
    public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        Particle p = particles[particleMask[index]];

        // Semi-implicit Euler: update velocity first
        float3 vel = (p.currentPosition - p.previousPosition) * invDt + gravityDt;

        p.previousPosition = p.currentPosition;
        p.currentPosition += vel * dt;

        particles[particleMask[index]] = p;
    }
}

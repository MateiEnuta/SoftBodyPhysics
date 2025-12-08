using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.XR;

#region Jobs
[BurstCompile]
public struct PositionUpdate : IJobParallelFor
{
    [ReadOnly] public float3 gravity;
    [ReadOnly] public float dt;
    public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        Particle currentParticle = particles[index];

        currentParticle.velocity += gravity * dt;
        currentParticle.previousPosition = currentParticle.position;
        currentParticle.position += currentParticle.velocity * dt;

        particles[index] = currentParticle;
    }
}

[BurstCompile]
public struct VelocityUpdate : IJobParallelFor
{
    [ReadOnly] public float dt;
    public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        Particle currentParticle = particles[index];
        
        currentParticle.velocity = (currentParticle.position - currentParticle.previousPosition) / dt;

        particles[index] = currentParticle;
    }
}

[BurstCompile]
public struct VisualVerticesUpdate : IJobParallelFor
{
    [ReadOnly] public NativeArray<Particle> particles;
    [ReadOnly] public NativeArray<VertexTriLinear> verticesTriLinear;
    public NativeArray<float3> verticesPositions;

    public void Execute(int index)
    {
        // Store the lattice corners
        int p0 = (int)verticesTriLinear[index].idx03.x;
        int p1 = (int)verticesTriLinear[index].idx03.y;
        int p2 = (int)verticesTriLinear[index].idx03.z;
        int p3 = (int)verticesTriLinear[index].idx03.w;
        int p4 = (int)verticesTriLinear[index].idx47.x;
        int p5 = (int)verticesTriLinear[index].idx47.y;
        int p6 = (int)verticesTriLinear[index].idx47.z;
        int p7 = (int)verticesTriLinear[index].idx47.w;

        // Interpolate on the x axis
        float3 x1 = math.lerp(particles[p0].position, particles[p1].position, verticesTriLinear[index].triLinearValues.x);
        float3 x2 = math.lerp(particles[p3].position, particles[p2].position, verticesTriLinear[index].triLinearValues.x);
        float3 x3 = math.lerp(particles[p4].position, particles[p5].position, verticesTriLinear[index].triLinearValues.x);
        float3 x4 = math.lerp(particles[p7].position, particles[p6].position, verticesTriLinear[index].triLinearValues.x);

        // Interpolate on the y axis
        float3 y1 = math.lerp(x1, x2, verticesTriLinear[index].triLinearValues.y);
        float3 y2 = math.lerp(x3, x4, verticesTriLinear[index].triLinearValues.y);

        // Interpolate on the z axis
        float3 z1 = math.lerp(y1, y2, verticesTriLinear[index].triLinearValues.z);

        verticesPositions[index] = z1;
    }
}

// Constraint jobs

[BurstCompile]
public struct DistanceConstraintsUpdate : IJobParallelFor
{
    [ReadOnly] public float dt;
    [ReadOnly] public float damping;
    [ReadOnly] public NativeArray<uint2> distanceConGroups;
    [NativeDisableParallelForRestriction]public NativeArray<DistanceConstraint> distanceConstraints;
    [NativeDisableParallelForRestriction]public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        int2 startEnd = (int2)distanceConGroups[index];
        for (int i = startEnd.x; i < startEnd.y; i++)
        {
            DistanceConstraint constraint = distanceConstraints[i];

            Particle p1 = particles[(int)constraint.particlesIdx.x];
            Particle p2 = particles[(int)constraint.particlesIdx.y];

            float3 dir = p2.position - p1.position;
            float currentLength = math.length(dir);
            if (currentLength < 1e-10f) continue;

            float C = currentLength - constraint.restLength;

            float3 n = dir / currentLength;
            float3 grad1 = -n;
            float3 grad2 = n;

            float sumWeightedGradients = p1.invMass + p2.invMass;

            float alphaTilde = constraint.compliance / (dt * dt);
            sumWeightedGradients += alphaTilde;
            if (math.abs(sumWeightedGradients) < 1e-10f) continue;

            float dampingTerm = 0f;
            if (damping > 0f)
            {
                float betaTilde = damping * dt * dt;
                float gamma = alphaTilde * betaTilde / dt;

                float3 v1 = p1.position - p1.previousPosition;
                float3 v2 = p2.position - p2.previousPosition;
                float relativeVelocity = math.dot(v2 - v1, grad2);

                dampingTerm = gamma * relativeVelocity;
                sumWeightedGradients *= 1f + gamma;
            }

            float deltaLambda = -(C + dampingTerm) / sumWeightedGradients;

            p1.position += p1.invMass * deltaLambda * grad1;
            p2.position += p2.invMass * deltaLambda * grad2;

            particles[(int)constraint.particlesIdx.x] = p1;
            particles[(int)constraint.particlesIdx.y] = p2;
        }
    }
}

[BurstCompile]
public struct VolumeConstraintsUpdate : IJobParallelFor
{
    [ReadOnly] public float dt;
    [ReadOnly] public float damping;
    [NativeDisableParallelForRestriction]public NativeArray<VolumeConstraint> volumeConstraints;
    [NativeDisableParallelForRestriction]public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        for (int i = index * 5; i < index * 5 + 5; i++)
        {
            VolumeConstraint constraint = volumeConstraints[i];

            Particle p1 = particles[(int)constraint.particlesIdx.x];
            Particle p2 = particles[(int)constraint.particlesIdx.y];
            Particle p3 = particles[(int)constraint.particlesIdx.z];
            Particle p4 = particles[(int)constraint.particlesIdx.w];

            float3 v1 = p2.position - p1.position;
            float3 v2 = p3.position - p1.position;
            float3 v3 = p4.position - p1.position;
            float3 cross23 = math.cross(v2, v3);
            float currentVolume = math.dot(v1, cross23) / 6f;

            float C = currentVolume - constraint.restVolume;

            float3 cross31 = math.cross(v3, v1);
            float3 cross12 = math.cross(v1, v2);

            float3 grad1 = -(cross23 + cross31 + cross12) / 6f;
            float3 grad2 = cross23 / 6f;
            float3 grad3 = cross31 / 6f;
            float3 grad4 = cross12 / 6f;

            float sumWeightedGradients = 0f;
            sumWeightedGradients += p1.invMass * math.lengthsq(grad1);
            sumWeightedGradients += p2.invMass * math.lengthsq(grad2);
            sumWeightedGradients += p3.invMass * math.lengthsq(grad3);
            sumWeightedGradients += p4.invMass * math.lengthsq(grad4);

            float alphaTilde = constraint.compliance / (dt * dt);
            sumWeightedGradients += alphaTilde;
            if (math.abs(sumWeightedGradients) < 1e-10f) continue;

            float dampingTerm = 0f;
            if (damping > 0f)
            {
                float betaTilde = damping * dt * dt;
                float gamma = alphaTilde * betaTilde / dt;

                float3 vel1 = p1.position - p1.previousPosition;
                float3 vel2 = p2.position - p2.previousPosition;
                float3 vel3 = p3.position - p3.previousPosition;
                float3 vel4 = p4.position - p4.previousPosition;

                float velocitySum = math.dot(grad1, vel1) + 
                    math.dot(grad2, vel2) + math.dot(grad3, vel3)
                    + math.dot(grad4, vel4);
                dampingTerm = gamma * velocitySum;

                sumWeightedGradients *= 1f + gamma;
            }

            float deltaLambda = -(C + dampingTerm) / sumWeightedGradients;

            p1.position += p1.invMass * deltaLambda * grad1;
            p2.position += p2.invMass * deltaLambda * grad2;
            p3.position += p3.invMass * deltaLambda * grad3;
            p4.position += p4.invMass * deltaLambda * grad4;

            particles[(int)constraint.particlesIdx.x] = p1;
            particles[(int)constraint.particlesIdx.y] = p2;
            particles[(int)constraint.particlesIdx.z] = p3;
            particles[(int)constraint.particlesIdx.w] = p4;
        }
    }
}

[BurstCompile]
public struct CrudeCollision : IJobParallelFor
{
    public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        Particle particle = particles[index];

        if (particle.position.y < 0) particle.position.y = 0;

        particles[index] = particle;
    }
}
#endregion

#region Variables
public struct SoftBodyData
{
    public ObjectInstanceManager instanceManager;
    public int verticesStartIndex;

    public SoftBodyData(ObjectInstanceManager instanceManager, int verticesStartIndex)
    {
        this.instanceManager = instanceManager;
        this.verticesStartIndex = verticesStartIndex;
    }
}

public struct Particle
{
    public float3 position, previousPosition, velocity;
    public float invMass;

    public Particle(float3 position, float invMass)
    {
        this.position = position;
        this.invMass = invMass;

        previousPosition = velocity = float3.zero;
    }
}

public struct DistanceConstraint
{
    public uint2 particlesIdx;
    public float restLength, compliance;

    public DistanceConstraint(DistanceConstraintRef distanceConstraintRef, uint particlesOffset, float compliance)
    {
        particlesIdx = distanceConstraintRef.pidx + new uint2(particlesOffset, particlesOffset);
        restLength = distanceConstraintRef.restLenght;
        this.compliance = compliance;
    }
}

public struct VolumeConstraint
{
    public uint4 particlesIdx;
    public float restVolume, compliance;

    public VolumeConstraint(VolumeConstraintRef volumeConstraintRef, uint particlesOffset, float compliance)
    {
        particlesIdx = volumeConstraintRef.pidx + new uint4(particlesOffset, particlesOffset, particlesOffset, particlesOffset);
        restVolume = volumeConstraintRef.restVolume;
        this.compliance = compliance;
    }
}

public class JobPhysicsSolver : MonoBehaviour
{
    // Silver settings
    [Range(0, 100)][SerializeField] private int substeps = 5;
    [SerializeField] private float damping = 0;

    // Soft bodies data
    private List<SoftBodyData> softBodies;

    // Particles data
    private NativeList<Particle> particles;

    // Mesh vertices data
    private NativeList<float3> verticesPositions;
    private NativeList<VertexTriLinear> verticesTriLinear;

    // Constraints data
    private NativeList<uint2>[] distanceConGroups;
    private NativeList<DistanceConstraint>[] distanceConstraints;
    private NativeList<VolumeConstraint>[] volumeConstraints;

    // Debug
    private bool initialised = false;
#endregion

#region Simulation
    int particleCount = 0;
    int[] distanceLengths;
    int[] volumeLengths;
    private void FixedUpdate()
    {
        if (!initialised) return;
        
        JobHandle handle = default;
        float dt = Time.fixedDeltaTime / substeps;
        particleCount = particles.Length;
        if (distanceLengths == null) distanceLengths = volumeLengths = new int[8];
        for (int i = 0; i < 8; i++)
        {
            distanceLengths[i] = distanceConGroups[i].Length;
            volumeLengths[i] = volumeConstraints[i].Length / 5;
        }

        for (int i = 0; i < substeps; i++)
        {
            UpdatePositions(ref handle, dt);
            UpdateConstraints(ref handle, dt);
            UpdateVelocities(ref handle, dt);
            JobHandle.ScheduleBatchedJobs();
        }

        UpdateVisualMeshes(ref handle);
    }

    private void UpdatePositions(ref JobHandle handle, float dt)
    {
        PositionUpdate positionUpdate = new PositionUpdate
        {
            gravity = Physics.gravity,
            dt = dt,
            particles = particles.AsArray()
        };
        handle = positionUpdate.Schedule(particleCount, 64, handle);
    }

    private void UpdateConstraints(ref JobHandle handle, float dt)
    {
        for (int i = 0; i < 8; i++)
        {
            DistanceConstraintsUpdate distanceConstraintsUpdate = new DistanceConstraintsUpdate
            {
                dt = dt,
                damping = damping,
                distanceConGroups = distanceConGroups[i].AsArray(),
                distanceConstraints = distanceConstraints[i].AsArray(),
                particles = particles.AsArray()
            };
            handle = distanceConstraintsUpdate.Schedule(distanceLengths[i], 1, handle);
        }

        for (int i = 0; i < 8; i++)
        {
            VolumeConstraintsUpdate volumeConstraintsUpdate = new VolumeConstraintsUpdate
            {
                dt = dt,
                damping = damping,
                volumeConstraints = volumeConstraints[i].AsArray(),
                particles = particles.AsArray()
            };
            handle = volumeConstraintsUpdate.Schedule(volumeLengths[i], 1, handle);
        }

        CrudeCollision crudeCollision = new CrudeCollision
        {
            particles = particles.AsArray()  
        };
        handle = crudeCollision.Schedule(particleCount, 64, handle);
    }

    private void UpdateVelocities(ref JobHandle handle, float dt)
    {
        VelocityUpdate velocityUpdate = new VelocityUpdate
        {
            dt = dt,
            particles = particles.AsArray()  
        };
        handle = velocityUpdate.Schedule(particleCount, 10, handle);
    }

    private void UpdateVisualMeshes(ref JobHandle handle)
    {
        VisualVerticesUpdate visualVerticesUpdate = new VisualVerticesUpdate
        {
            particles = particles.AsArray(),
            verticesTriLinear = verticesTriLinear.AsArray(),
            verticesPositions = verticesPositions.AsArray()
        };
        handle = visualVerticesUpdate.Schedule(verticesPositions.Length, 64, handle);
        handle.Complete();

        for (int i = 0; i < softBodies.Count; i++)
        {
            softBodies[i].instanceManager.UpdateMesh(verticesPositions.AsArray(), softBodies[i].verticesStartIndex);
        }
    }
#endregion

#region  Object addition
    public void AddSoftBody(ObjectInstanceManager objectManager, float bodyMass, float bodyDisCompliance, float bodyVolCompliance, 
        List<float3> bodyParticles, List<uint2>[] bodyDisConGroups, List<DistanceConstraintRef>[] bodyDisConstraints, 
        List<VolumeConstraintRef>[] bodyVolConstraints, VertexTriLinear[] bodyVerticesTriLinear)
    {
        if (!initialised) InitialiseSolver();

        // Soft body data addition
        int verticesOffset = verticesPositions.Length;
        softBodies.Add(new SoftBodyData(objectManager, verticesOffset));

        // Particles addition
        uint particlesOffset = (uint)particles.Length;
        float particleInvMass = 1f / (bodyMass / bodyParticles.Count);
        for (int i = 0; i < bodyParticles.Count; i++) particles.Add(new Particle(bodyParticles[i], particleInvMass));

        // Mesh vertices data addition
        for (int i = 0; i < bodyVerticesTriLinear.Length; i++)
        {
            verticesPositions.Add(float3.zero);
            verticesTriLinear.Add(new VertexTriLinear(
                bodyVerticesTriLinear[i].idx03 + new uint4(particlesOffset, particlesOffset, particlesOffset, particlesOffset), 
                bodyVerticesTriLinear[i].idx47 + new uint4(particlesOffset, particlesOffset, particlesOffset, particlesOffset), 
                bodyVerticesTriLinear[i].triLinearValues
            ));
        }

        // Constraints addition
        for (int k = 0; k < 8; k++)
        {
            for (int i = 0; i < bodyDisConGroups[k].Count; i++) distanceConGroups[k].Add(bodyDisConGroups[k][i]);
            for (int i = 0; i < bodyDisConstraints[k].Count; i++) distanceConstraints[k].Add(
                new DistanceConstraint(bodyDisConstraints[k][i], particlesOffset, bodyDisCompliance)
            );
            for (int i = 0; i < bodyVolConstraints[k].Count; i++) volumeConstraints[k].Add(
                new VolumeConstraint(bodyVolConstraints[k][i], particlesOffset, bodyVolCompliance)
            );
        }
    }
#endregion

#region Data managment
    public void InitialiseSolver()
    {
        initialised = true;

        // Soft bodies data initialisation
        softBodies = new List<SoftBodyData>();

        // Particle data initialisation
        particles = new NativeList<Particle>(Allocator.Persistent);

        // Mesh vertices data initialisation
        verticesPositions = new NativeList<float3>(Allocator.Persistent);
        verticesTriLinear = new NativeList<VertexTriLinear>(Allocator.Persistent);

        // Constraint data initialisation
        distanceConGroups = new NativeList<uint2>[8];
        distanceConstraints = new NativeList<DistanceConstraint>[8];
        volumeConstraints = new NativeList<VolumeConstraint>[8];
        for (int i = 0; i < 8; i++)
        {
            distanceConGroups[i] = new NativeList<uint2>(Allocator.Persistent);
            distanceConstraints[i] = new NativeList<DistanceConstraint>(Allocator.Persistent);
            volumeConstraints[i] = new NativeList<VolumeConstraint>(Allocator.Persistent);
        }
    }

    private void OnDestroy()
    {
        if (!initialised) return;

        softBodies = null;

        particles.Dispose();

        verticesPositions.Dispose();
        verticesTriLinear.Dispose();
        
        for (int i = 0; i < 8; i++)
        {
            distanceConGroups[i].Dispose();
            distanceConstraints[i].Dispose();
            volumeConstraints[i].Dispose();
        }
        distanceConGroups = null;
        distanceConstraints = null;
        volumeConstraints = null;
    }
#endregion
}

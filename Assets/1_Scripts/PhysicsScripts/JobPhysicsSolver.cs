using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.XR;

#region Jobs
// -----------------------------------------------------------------------------
// POSITION UPDATE JOB
// Integrates particle positions using explicit Euler (XPBD-style).
// Stores previousPosition so velocities can be recomputed later.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct PositionUpdate : IJobParallelFor
{
    [ReadOnly] public float3 gravity;
    [ReadOnly] public float dt;
    public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        // Read particle
        Particle currentParticle = particles[index];

        // Semi-implicit Euler: update velocity first
        currentParticle.velocity += gravity * dt;

        // Save previous position for velocity reconstruction
        currentParticle.previousPosition = currentParticle.position;

        // Integrate new position: x += v * dt
        currentParticle.position += currentParticle.velocity * dt;

        // Write back
        particles[index] = currentParticle;
    }
}

// -----------------------------------------------------------------------------
// VELOCITY UPDATE JOB
// Reconstructs velocities from positions after constraints have corrected them.
// XPBD keeps constraints stable by recomputing velocity from the correction.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct VelocityUpdate : IJobParallelFor
{
    [ReadOnly] public float dt;
    public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        Particle currentParticle = particles[index];

        // Velocity = (x_new - x_old) / dt
        currentParticle.velocity = 
            (currentParticle.position - currentParticle.previousPosition) / dt;

        particles[index] = currentParticle;
    }
}

// -----------------------------------------------------------------------------
// VERTEX INTERPOLATION JOB
// Computes mesh vertex positions by trilinearly interpolating 8 lattice nodes.
// Each visual vertex is a weighted blend of 8 particles (cube cell corners).
// -----------------------------------------------------------------------------
[BurstCompile]
public struct VisualVerticesUpdate : IJobParallelFor
{
    [ReadOnly] public NativeArray<Particle> particles;
    [ReadOnly] public NativeArray<VertexTriLinear> verticesTriLinear;
    public NativeArray<float3> verticesPositions;

    public void Execute(int index)
    {
        // Indices of the 8 lattice particles influencing this vertex
        int p0 = (int)verticesTriLinear[index].idx03.x;
        int p1 = (int)verticesTriLinear[index].idx03.y;
        int p2 = (int)verticesTriLinear[index].idx03.z;
        int p3 = (int)verticesTriLinear[index].idx03.w;
        int p4 = (int)verticesTriLinear[index].idx47.x;
        int p5 = (int)verticesTriLinear[index].idx47.y;
        int p6 = (int)verticesTriLinear[index].idx47.z;
        int p7 = (int)verticesTriLinear[index].idx47.w;

        // Trilinear interpolation along x → y → z
        float3 x1 = math.lerp(particles[p0].position, particles[p1].position, verticesTriLinear[index].triLinearValues.x);
        float3 x2 = math.lerp(particles[p3].position, particles[p2].position, verticesTriLinear[index].triLinearValues.x);
        float3 x3 = math.lerp(particles[p4].position, particles[p5].position, verticesTriLinear[index].triLinearValues.x);
        float3 x4 = math.lerp(particles[p7].position, particles[p6].position, verticesTriLinear[index].triLinearValues.x);

        float3 y1 = math.lerp(x1, x2, verticesTriLinear[index].triLinearValues.y);
        float3 y2 = math.lerp(x3, x4, verticesTriLinear[index].triLinearValues.y);

        // Final Z interpolation
        float3 z1 = math.lerp(y1, y2, verticesTriLinear[index].triLinearValues.z);

        verticesPositions[index] = z1;
    }
}

// -----------------------------------------------------------------------------
// DISTANCE CONSTRAINT JOB
// Solves batches of distance constraints for a given channel.
// Each channel stores ranges [start,end) for that body.
// XPBD formulation including optional damping.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct DistanceConstraintsUpdate : IJobParallelFor
{
    [ReadOnly] public float dt;
    [ReadOnly] public float damping;

    // Every element describes a range of constraints to process.
    [ReadOnly] public NativeArray<uint2> distanceConGroups;

    // Full list of actual constraints
    [NativeDisableParallelForRestriction] public NativeArray<DistanceConstraint> distanceConstraints;

    // Particle array (writes allowed)
    [NativeDisableParallelForRestriction] public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        // Range of constraints belonging to this job index
        int2 startEnd = (int2)distanceConGroups[index];

        // Process each constraint sequentially
        for (int i = startEnd.x; i < startEnd.y; i++)
        {
            DistanceConstraint constraint = distanceConstraints[i];

            // Load particle data
            Particle p1 = particles[(int)constraint.particlesIdx.x];
            Particle p2 = particles[(int)constraint.particlesIdx.y];

            float3 dir = p2.position - p1.position;
            float currentLength = math.length(dir);
            if (currentLength < 1e-10f) continue; // Prevent NaN

            float C = currentLength - constraint.restLength;

            // Gradients of the constraint
            float3 n = dir / currentLength;
            float3 grad1 = -n;
            float3 grad2 = n;

            // XPBD sum of w_i * |grad|^2  (for distance constraints = invMass1 + invMass2)
            float sumWeightedGradients = p1.invMass + p2.invMass;

            // Compliance term: α = compliance / dt²
            float alphaTilde = constraint.compliance / (dt * dt);
            sumWeightedGradients += alphaTilde;

            if (math.abs(sumWeightedGradients) < 1e-10f) continue;

            // -------------------------
            // Constraint damping (optional)
            // -------------------------
            float dampingTerm = 0f;
            if (damping > 0f)
            {
                float betaTilde = damping * dt * dt;
                float gamma = alphaTilde * betaTilde / dt;

                // Constraint-space velocity
                float3 v1 = p1.position - p1.previousPosition;
                float3 v2 = p2.position - p2.previousPosition;
                float relativeVelocity = math.dot(v2 - v1, grad2);

                dampingTerm = gamma * relativeVelocity;

                // Modify denominator for stable XPBD damping
                sumWeightedGradients *= 1f + gamma;
            }

            // XPBD correction
            float deltaLambda = -(C + dampingTerm) / sumWeightedGradients;

            // Apply positional correction
            p1.position += p1.invMass * deltaLambda * grad1;
            p2.position += p2.invMass * deltaLambda * grad2;

            // Write back
            particles[(int)constraint.particlesIdx.x] = p1;
            particles[(int)constraint.particlesIdx.y] = p2;
        }
    }
}

// -----------------------------------------------------------------------------
// VOLUME CONSTRAINT JOB
// Solves tetrahedral volume preservation constraints in batches of 5.
// Standard XPBD volume formulation with damping.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct VolumeConstraintsUpdate : IJobParallelFor
{
    [ReadOnly] public float dt;
    [ReadOnly] public float damping;

    [NativeDisableParallelForRestriction] public NativeArray<VolumeConstraint> volumeConstraints;
    [NativeDisableParallelForRestriction] public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        // Each tetra uses 5 constraints stored contiguously
        for (int i = index * 5; i < index * 5 + 5; i++)
        {
            VolumeConstraint constraint = volumeConstraints[i];

            // Load tetra particles
            Particle p1 = particles[(int)constraint.particlesIdx.x];
            Particle p2 = particles[(int)constraint.particlesIdx.y];
            Particle p3 = particles[(int)constraint.particlesIdx.z];
            Particle p4 = particles[(int)constraint.particlesIdx.w];

            // Compute signed volume
            float3 v1 = p2.position - p1.position;
            float3 v2 = p3.position - p1.position;
            float3 v3 = p4.position - p1.position;
            float3 cross23 = math.cross(v2, v3);
            float currentVolume = math.dot(v1, cross23) / 6f;

            float C = currentVolume - constraint.restVolume;

            // Gradients for tetrahedral volume constraint
            float3 cross31 = math.cross(v3, v1);
            float3 cross12 = math.cross(v1, v2);

            float3 grad1 = -(cross23 + cross31 + cross12) / 6f;
            float3 grad2 = cross23 / 6f;
            float3 grad3 = cross31 / 6f;
            float3 grad4 = cross12 / 6f;

            // Weighted gradient magnitudes
            float sumWeightedGradients = 0f;
            sumWeightedGradients += p1.invMass * math.lengthsq(grad1);
            sumWeightedGradients += p2.invMass * math.lengthsq(grad2);
            sumWeightedGradients += p3.invMass * math.lengthsq(grad3);
            sumWeightedGradients += p4.invMass * math.lengthsq(grad4);

            float alphaTilde = constraint.compliance / (dt * dt);
            sumWeightedGradients += alphaTilde;

            if (math.abs(sumWeightedGradients) < 1e-10f) continue;

            // -------------------------
            // XPBD damping term
            // -------------------------
            float dampingTerm = 0f;
            if (damping > 0f)
            {
                float betaTilde = damping * dt * dt;
                float gamma = alphaTilde * betaTilde / dt;

                float3 vel1 = p1.position - p1.previousPosition;
                float3 vel2 = p2.position - p2.previousPosition;
                float3 vel3 = p3.position - p3.previousPosition;
                float3 vel4 = p4.position - p4.previousPosition;

                float velocitySum =
                    math.dot(grad1, vel1) +
                    math.dot(grad2, vel2) +
                    math.dot(grad3, vel3) +
                    math.dot(grad4, vel4);

                dampingTerm = gamma * velocitySum;
                sumWeightedGradients *= 1f + gamma;
            }

            // Compute constraint correction
            float deltaLambda = -(C + dampingTerm) / sumWeightedGradients;

            // Apply position correction to all tetra points
            p1.position += p1.invMass * deltaLambda * grad1;
            p2.position += p2.invMass * deltaLambda * grad2;
            p3.position += p3.invMass * deltaLambda * grad3;
            p4.position += p4.invMass * deltaLambda * grad4;

            // Write back
            particles[(int)constraint.particlesIdx.x] = p1;
            particles[(int)constraint.particlesIdx.y] = p2;
            particles[(int)constraint.particlesIdx.z] = p3;
            particles[(int)constraint.particlesIdx.w] = p4;
        }
    }
}

// -----------------------------------------------------------------------------
// SIMPLE COLLISION TEST
// Keeps particles above y=0. Very crude but safe for testing.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct CrudeCollision : IJobParallelFor
{
    public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        Particle particle = particles[index];

        // Clamp to the plane at y = 0
        if (particle.position.y < 0)
            particle.position.y = 0;

        particles[index] = particle;
    }
}
#endregion

#region Variables
// General data each soft body
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

// Particle data
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

// Data for the distance constraint
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

// Data for the volume constraint
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

        // Substep delta time (XPBD requires small timesteps)
        float dt = Time.fixedDeltaTime / substeps;

        particleCount = particles.Length;

        // Cache per-channel list sizes (fixes safety issues when using NativeList.Length)
        if (distanceLengths == null) distanceLengths = volumeLengths = new int[8];
        for (int i = 0; i < 8; i++)
        {
            // Number of distance constraint groups in this solver batch
            distanceLengths[i] = distanceConGroups[i].Length;

            // Number of volumetric constraints (5 entries per tetra constraint)
            volumeLengths[i] = volumeConstraints[i].Length / 5;
        }

        // XPBD multi-step solver
        for (int i = 0; i < substeps; i++)
        {
            // Predict integration step
            UpdatePositions(ref handle, dt);

            // Solve all constraints
            UpdateConstraints(ref handle, dt);

            // Reconstruct velocities after corrections
            UpdateVelocities(ref handle, dt);

            JobHandle.ScheduleBatchedJobs();
        }

        // Copy simulated vertices into meshes
        UpdateVisualMeshes(ref handle);
    }

    private void UpdatePositions(ref JobHandle handle, float dt)
    {
        // Integrate particle positions using semi-implicit Euler
        PositionUpdate positionUpdate = new PositionUpdate
        {
            gravity = Physics.gravity,
            dt = dt,
            particles = particles.AsArray()
        };

        // Parallel integration
        handle = positionUpdate.Schedule(particleCount, 64, handle);
    }

    private void UpdateConstraints(ref JobHandle handle, float dt)
    {
        // Solve distance constraints across 8 solver batches
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

            // Each range corresponds to multiple constraints
            handle = distanceConstraintsUpdate.Schedule(distanceLengths[i], 1, handle);
        }

        // Solve volume constraints across the 8 solver batches
        for (int i = 0; i < 8; i++)
        {
            VolumeConstraintsUpdate volumeConstraintsUpdate = new VolumeConstraintsUpdate
            {
                dt = dt,
                damping = damping,
                volumeConstraints = volumeConstraints[i].AsArray(),
                particles = particles.AsArray()
            };

            // Each job handles one tetra group (5 entries)
            handle = volumeConstraintsUpdate.Schedule(volumeLengths[i], 1, handle);
        }

        // Very simple plane collision for starters
        CrudeCollision crudeCollision = new CrudeCollision
        {
            particles = particles.AsArray()
        };
        handle = crudeCollision.Schedule(particleCount, 64, handle);
    }

    private void UpdateVelocities(ref JobHandle handle, float dt)
    {
        // Recompute velocities from corrected positional updates
        VelocityUpdate velocityUpdate = new VelocityUpdate
        {
            dt = dt,
            particles = particles.AsArray()
        };

        handle = velocityUpdate.Schedule(particleCount, 10, handle);
    }

    private void UpdateVisualMeshes(ref JobHandle handle)
    {
        // Compute mesh vertices from lattice particles via trilinear interpolation
        VisualVerticesUpdate visualVerticesUpdate = new VisualVerticesUpdate
        {
            particles = particles.AsArray(),
            verticesTriLinear = verticesTriLinear.AsArray(),
            verticesPositions = verticesPositions.AsArray()
        };

        handle = visualVerticesUpdate.Schedule(verticesPositions.Length, 64, handle);

        // Must complete before sending results to the render thread
        handle.Complete();

        // Push updated vertex data into each soft body's mesh instance
        for (int i = 0; i < softBodies.Count; i++)
        {
            softBodies[i].instanceManager.UpdateMesh(verticesPositions.AsArray(),
                                                    softBodies[i].verticesStartIndex);
        }
    }
#endregion

#region Object addition
    public void AddSoftBody(
        ObjectInstanceManager objectManager,
        float bodyMass,
        float bodyDisCompliance,
        float bodyVolCompliance,
        List<float3> bodyParticles,
        List<uint2>[] bodyDisConGroups,
        List<DistanceConstraintRef>[] bodyDisConstraints,
        List<VolumeConstraintRef>[] bodyVolConstraints,
        VertexTriLinear[] bodyVerticesTriLinear)
    {
        if (!initialised) InitialiseSolver();

        // Register new soft body instance
        int verticesOffset = verticesPositions.Length;
        softBodies.Add(new SoftBodyData(objectManager, verticesOffset));

        // Add all particles belonging to the body
        uint particlesOffset = (uint)particles.Length;
        float particleInvMass = 1f / (bodyMass / bodyParticles.Count);

        for (int i = 0; i < bodyParticles.Count; i++)
            particles.Add(new Particle(bodyParticles[i], particleInvMass));

        // Add empty vertex entries + trilinear mapping
        for (int i = 0; i < bodyVerticesTriLinear.Length; i++)
        {
            verticesPositions.Add(float3.zero);

            // Remap indices by particle offset
            verticesTriLinear.Add(
                new VertexTriLinear(
                    bodyVerticesTriLinear[i].idx03 +
                    new uint4(particlesOffset),
                    bodyVerticesTriLinear[i].idx47 +
                    new uint4(particlesOffset),
                    bodyVerticesTriLinear[i].triLinearValues));
        }

        // Append all constraints into the solver's 8-channel structure
        for (int k = 0; k < 8; k++)
        {
            uint disConsOffset = (uint)distanceConstraints[k].Length;

            // Add constraint ranges
            for (int i = 0; i < bodyDisConGroups[k].Count; i++)
                distanceConGroups[k].Add(bodyDisConGroups[k][i] + new uint2(disConsOffset));

            // Add distance constraints
            for (int i = 0; i < bodyDisConstraints[k].Count; i++)
                distanceConstraints[k].Add(new DistanceConstraint(
                    bodyDisConstraints[k][i], particlesOffset, bodyDisCompliance));

            // Add volume constraints
            for (int i = 0; i < bodyVolConstraints[k].Count; i++)
                volumeConstraints[k].Add(new VolumeConstraint(
                    bodyVolConstraints[k][i], particlesOffset, bodyVolCompliance));
        }
    }
#endregion

#region Data management
    public void InitialiseSolver()
    {
        initialised = true;

        // List of all soft body instances
        softBodies = new List<SoftBodyData>();

        // Main particle storage
        particles = new NativeList<Particle>(Allocator.Persistent);

        // Storage for mesh vertices & their lattice mapping
        verticesPositions = new NativeList<float3>(Allocator.Persistent);
        verticesTriLinear = new NativeList<VertexTriLinear>(Allocator.Persistent);

        // Initialise per-channel constraint lists
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

        // Managed lists
        softBodies = null;

        // Dispose particle data
        particles.Dispose();

        // Dispose vertex data
        verticesPositions.Dispose();
        verticesTriLinear.Dispose();

        // Dispose constraint lists
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

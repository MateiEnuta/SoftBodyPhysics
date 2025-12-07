using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

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
    public float restLenght, compliance;

    public DistanceConstraint(DistanceConstraintRef distanceConstraintRef, uint particlesOffset, float compliance)
    {
        particlesIdx = distanceConstraintRef.pidx + new uint2(particlesOffset, particlesOffset);
        restLenght = distanceConstraintRef.restLenght;
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

    // Solver Settings
    private bool initialised = false;
#endregion

#region Simulation
    private void FixedUpdate()
    {
        if (!initialised) return;

        UpdateMeshes();
    }

    private void UpdateMeshes()
    {
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

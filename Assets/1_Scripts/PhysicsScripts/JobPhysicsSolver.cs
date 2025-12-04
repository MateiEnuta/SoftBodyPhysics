using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

#region Variables
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
    // Particle data
    private NativeList<Particle> particles;

    // Constraints data
    private NativeList<uint2>[] distanceConGroups;
    private NativeList<DistanceConstraint>[] distanceConstraints;
    private NativeList<VolumeConstraint>[] volumeConstraints;

    // Solver data
    private bool initialised = false;
#endregion

#region Simulation
    private void FixedUpdate()
    {
        if (!initialised) return;
    }
#endregion

#region  Object addition
    public void AddSoftBody(float bodyMass, float bodyDisCompliance, float bodyVolCompliance, List<float3> bodyParticles, List<uint2>[] bodyDisConGroups,
        List<DistanceConstraintRef>[] bodyDisConstraints, List<VolumeConstraintRef>[] bodyVolConstraints)
    {
        if (!initialised) InitialiseSolver();

        // Particle addition
        uint particlesOffset = (uint)particles.Length;
        float particleInvMass = 1f / (bodyMass / bodyParticles.Count);
        for (int i = 0; i < bodyParticles.Count; i++) particles.Add(new Particle(bodyParticles[i], particleInvMass));

        // Constraint addition
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

        // particle data initialisation
        particles = new NativeList<Particle>();

        // constraint data initialisation
        distanceConGroups = new NativeList<uint2>[8];
        distanceConstraints = new NativeList<DistanceConstraint>[8];
        volumeConstraints = new NativeList<VolumeConstraint>[8];
        for (int i = 0; i < 8; i++)
        {
            distanceConGroups[i] = new NativeList<uint2>();
            distanceConstraints[i] = new NativeList<DistanceConstraint>();
            volumeConstraints[i] = new NativeList<VolumeConstraint>();
        }
    }

    private void OnDestroy()
    {
        if (!initialised) return;

        particles.Dispose();
        
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

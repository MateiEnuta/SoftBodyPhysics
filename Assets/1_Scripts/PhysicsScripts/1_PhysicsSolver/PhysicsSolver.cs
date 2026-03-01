using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using SolverStructs;

public class PhysicsSolver : MonoBehaviour
{
    // All of the processed lattice bodies
    public List<LatticePreprocessor> latticeDatas;
    public float3 targetLatticeDimensions = new float3(1,1,1);

    private bool initialised = false;

    private NativeList<Particle> particles;

    public void ProcessLatticeBody(LatticeBody latticeBody)
    {
        // Try to initialise
        InitialiseLists();
        // Create a new pre processor and store it
        LatticePreprocessor currentPreProcessor = new LatticePreprocessor(latticeBody, targetLatticeDimensions);
        latticeDatas.Add(currentPreProcessor);

        // Add lattice body particles
        float particleInvMass = 1f / (latticeBody.mass / currentPreProcessor.activeParticles.Count);
        for (int i = 0; i < currentPreProcessor.activeParticles.Count; i++)
        {
            // Calculate the particle position in the lattice
            int3 particleCoords = currentPreProcessor.to3D_Particle(currentPreProcessor.activeParticles[i]);
            float3 posInLattice = currentPreProcessor.latticeNegativeCorner + particleCoords * currentPreProcessor.latticeDimensions;
            
            // Transform the position to world space
            float3 posInWorld = latticeBody.transform.TransformPoint(posInLattice);

            // Set the position with 0 velocity
            // Create new particle with set invMass
            Particle newParticle = new Particle
            {
                invMass = particleInvMass,
                currentPosition = posInWorld,
                previousPosition = posInWorld
            };

            // Add new particle
            particles.Add(newParticle);
        }
    }

    private void InitialiseLists()
    {
        // Only initialise once
        if (initialised) return;
        initialised = true;

        // Allocate all native lists
        particles = new NativeList<Particle>(Allocator.Persistent);
    }

    private void DisposeLists()
    {
        // Dispose of all native lists
        if (particles.IsCreated) particles.Dispose();
    }


    private void OnEnable()
    {
        InitialiseLists();
    }

    private void OnDisable()
    {
        DisposeLists();
    }
}

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

    private NativeList<Particle>    particles;
    private NativeList<LatticeCell> latticeCells;
    private int[]                   globalChlCount;

    private int[] initialChlCount = new int[8]; 
    public void ProcessLatticeBody(LatticeBody latticeBody)
    {
        // Try to initialise
        InitialiseLists();
        // Create a new pre processor and store it
        LatticePreprocessor currentPreProcessor = new LatticePreprocessor(latticeBody, targetLatticeDimensions);
        latticeDatas.Add(currentPreProcessor);

        // Store initial list lengths
        int initialParticleCount = particles.Length;
        for (int i = 0; i < 8; i++) initialChlCount[i] = currentPreProcessor.chlCount[i];

        // Add particles
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

        // Add lattice cells
        for (int i = 0; i < currentPreProcessor.finalLatticeCells.Length; i++)
        {
            // Find the current cell channel
            int currentChannel = 0, channelSum = 0;
            for (currentChannel = 0; currentChannel < 8; currentChannel++)
            {
                channelSum += currentPreProcessor.chlCount[currentChannel];
                if (i < channelSum) break;
            }

            // Update the global channel list and calculate the new cell position
            globalChlCount[currentChannel]++;
            int newCellPos = channelSum - currentPreProcessor.chlCount[currentChannel];
            
            // Generate new cell
            LatticeCell newCell = currentPreProcessor.finalLatticeCells[i];
            newCell.v0 += initialParticleCount;
            newCell.v1 += initialParticleCount;
            newCell.v2 += initialParticleCount;
            newCell.v3 += initialParticleCount;
            newCell.v4 += initialParticleCount;
            newCell.v5 += initialParticleCount;
            newCell.v6 += initialParticleCount;
            newCell.v7 += initialParticleCount;

            // Insert new cell
            InsertAt(ref latticeCells, newCellPos, newCell);
        }
    }

    private void InitialiseLists()
    {
        // Only initialise once
        if (initialised) return;
        initialised = true;

        // Allocate all native lists
        // Particles allocation
        particles = new NativeList<Particle>(Allocator.Persistent);

        // Cell data allocation
        globalChlCount = new int[8];
        for (int i = 0; i < 8; i++) globalChlCount[i] = 0;
        latticeCells = new NativeList<LatticeCell>(Allocator.Persistent);
    }

    private void DisposeLists()
    {
        // Dispose of all native lists
        if (particles.IsCreated) particles.Dispose();
        if (latticeCells.IsCreated) latticeCells.Dispose();
    }


    private void OnEnable()
    {
        InitialiseLists();
    }

    private void OnDisable()
    {
        DisposeLists();
    }

    public void InsertAt<T>(ref NativeList<T> list, int index, in T value)
        where T : unmanaged
    {
        // Append one element to grow length by 1
        list.Add(value); // value will be overwritten if index < Length-1

        // Shift elements [index..Length-2] one step to the right
        for (int i = list.Length - 1; i > index; i--)
        {
            list[i] = list[i - 1];
        }

        // Write the new value at index
        list[index] = value;
    }
}

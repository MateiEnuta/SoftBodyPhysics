using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using SolverStructs;
using System.Linq;

public class PhysicsSolver : MonoBehaviour
{
    // --------------Public data--------------
    public float3 targetLatticeDimensions = new float3(1,1,1);
    public bool debugLatticeCells = false, debugColMesh = false, debugVisVerts = false;

    // --------------Solver data--------------
    private bool initialised = false;
    // Lattice bodies data
    private List<BodyWraper> latticeBodies;

    // --------------Particle data--------------
    private NativeList<Particle>    particles;

    // --------------Cell data--------------
    private NativeList<LatticeCell> latticeCells;
    private List<int>               latticeColReferences;
    private List<int>               latticeVisReferences;
    private int[]                   globalChlCount;

    // --------------Collision data--------------
    private NativeList<DynamicVertex>   collisionVertices;
    private NativeList<int3>            collisionTriangles;
    private NativeList<int2>            collisionEdges;
    private NativeList<CollisionWraper> collisionWrapers;

    // --------------Visual data--------------
    private NativeList<DynamicVertex> visualVertices;
    private NativeList<int>           visualVerticesCells;


#region Initialisation

    public void ProcessLatticeBody(LatticeBody latticeBody)
    {
        InitialiseSolver();
        // Generate all lattice data
        LatticePreprocessor currentProcess = new LatticePreprocessor(latticeBody, targetLatticeDimensions);
        BodyWraper bodyWraper = new BodyWraper(currentProcess);

        // Store initial list lengths
        int initialParticleCount = particles.Length;
        int initialColVertCount = collisionVertices.Length;
        int initialColTriCount = collisionTriangles.Length;
        int initialColEdgeCount = collisionEdges.Length;
        int initialColWraperCount = collisionWrapers.Length;
        int initialVisVerticesCount = visualVertices.Length;


        // Add particles
        float particleInvMass = 1f / (latticeBody.mass / currentProcess.activeParticles.Count);
        for (int i = 0; i < currentProcess.activeParticles.Count; i++)
        {
            // Calculate the particle position in the lattice
            int3 particleCoords = currentProcess.to3D_Particle(currentProcess.activeParticles[i]);
            float3 posInLattice = currentProcess.latticeNegativeCorner + particleCoords * currentProcess.latticeDimensions;
            
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
        // Store particle range
        bodyWraper.particlesRange = new int2(initialParticleCount, initialParticleCount - particles.Length);


        // Add lattice cells
        int currentLatticeCell = 0;
        int insertionPos = 0; // where the next channel’s block starts globally
        for (int currentChl = 0; currentChl < 8; currentChl++)
        {
            int existingCount = globalChlCount[currentChl];  // how many cells of this color are already in latticeCells
            int addCount      = currentProcess.chlCount[currentChl];
            if (addCount == 0)
            {
                // Skip to next channel, but the insertion position must jump
                insertionPos += existingCount;
                continue;
            }

            // Ensure space at the end
            for (int i = 0; i < addCount; i++)
                latticeCells.Add(new LatticeCell());

            // Shift the existing tail (all cells after this channel’s current block)
            // from [insertionPos + existingCount .. oldLength-1] up by addCount
            for (int i = latticeCells.Length - 1; i >= insertionPos + existingCount + addCount; i--)
                latticeCells[i] = latticeCells[i - addCount];

            
            // Fill the gap [insertionPos + existingCount .. insertionPos + existingCount + addCount)
            int writeStart = insertionPos + existingCount;
            latticeColReferences.InsertRange(writeStart, Enumerable.Repeat(-1, addCount));
            latticeVisReferences.InsertRange(writeStart, Enumerable.Repeat(-1, addCount));

            for (int i = 0; i < addCount; i++)
            {
                LatticeCell newCell = currentProcess.finalLatticeCells[currentLatticeCell];
                newCell.v0 += initialParticleCount;
                newCell.v1 += initialParticleCount;
                newCell.v2 += initialParticleCount;
                newCell.v3 += initialParticleCount;
                newCell.v4 += initialParticleCount;
                newCell.v5 += initialParticleCount;
                newCell.v6 += initialParticleCount;
                newCell.v7 += initialParticleCount;

                latticeCells[writeStart + i] = newCell;
                latticeColReferences[writeStart + i] = currentProcess.cellToColCell[currentLatticeCell] + initialColWraperCount;
                latticeVisReferences[writeStart + i] = currentProcess.cellToVisCell[currentLatticeCell] + initialVisVerticesCount;
                // Increment the pre processor lattice index
                currentLatticeCell++;
            }

            // Store lattice channel range
            bodyWraper.cellRanges[currentChl] = new int2(existingCount, addCount);

            // Move insertionPos past this channel’s whole block (old + new)
            insertionPos += existingCount + addCount;
            globalChlCount[currentChl] += addCount;
        }


        // Add collision data
        for (int i = 0; i < currentProcess.collisionVertices.Count; i++)
            collisionVertices.Add(currentProcess.collisionVertices[i]);
        for (int i = 0; i < currentProcess.collisionTriangles.Count; i++)
            collisionTriangles.Add(currentProcess.collisionTriangles[i]);
        for (int i = 0; i < currentProcess.collisionEdges.Count; i++)
            collisionEdges.Add(currentProcess.collisionEdges[i]);
        for (int i = 0; i < currentProcess.collisionWrapers.Count; i++)
            collisionWrapers.Add(currentProcess.collisionWrapers[i]);
        // Store collision data range
        bodyWraper.colVertsRange = new int2(initialColVertCount, collisionVertices.Length);
        bodyWraper.colTrisRange = new int2(initialColTriCount, collisionTriangles.Length);
        bodyWraper.colEdgesRange = new int2(initialColEdgeCount, collisionEdges.Length);
        bodyWraper.colWrapersRange = new int2(initialColWraperCount, collisionWrapers.Length);

        // Add visual data
        for (int i = 0; i < currentProcess.visualVertices.Length; i++)
        {
            visualVertices.Add(currentProcess.visualVertices[i]);
            visualVerticesCells.Add(currentProcess.visualVerticesCells[i]);
        }
        bodyWraper.visVerticesRange = new int2(initialVisVerticesCount, visualVertices.Length);


        // Assign the new lattice body wraper
        latticeBodies.Add(bodyWraper);

        // At the end update all cell refrences to match the updated list
        UpdateCellVertReferences();
    }

    private void UpdateCellVertReferences()
    {
        for (int i = 0; i < latticeCells.Length; i++)
        {
            if (latticeColReferences[i] != -1) {
                CollisionWraper wraperCopy = collisionWrapers[latticeColReferences[i]];
                wraperCopy.cellIdx = i;
                collisionWrapers[latticeColReferences[i]] = wraperCopy;
            }
            if (latticeVisReferences[i] != -1) visualVerticesCells[latticeVisReferences[i]] = i;
        }
    }

    private void InitialiseSolver()
    {
        // Only initialise once
        if (initialised) return;
        initialised = true;

        latticeBodies = new List<BodyWraper>();

        // Allocate all native lists
        // Particles allocation
        particles = new NativeList<Particle>(Allocator.Persistent);

        // Cell data allocation
        latticeCells = new NativeList<LatticeCell>(Allocator.Persistent);
        globalChlCount = new int[8];
        for (int i = 0; i < 8; i++) globalChlCount[i] = 0;
        latticeColReferences = new List<int>();
        latticeVisReferences = new List<int>();

        // Collision data allocation
        collisionVertices = new NativeList<DynamicVertex>(Allocator.Persistent);
        collisionTriangles = new NativeList<int3>(Allocator.Persistent);
        collisionEdges = new NativeList<int2>(Allocator.Persistent);
        collisionWrapers = new NativeList<CollisionWraper>(Allocator.Persistent);

        // Visual data allocation
        visualVertices = new NativeList<DynamicVertex>(Allocator.Persistent);
        visualVerticesCells = new NativeList<int>(Allocator.Persistent);
    }

    private void DisposeLists()
    {
        // Dispose of all native lists
        if (particles.IsCreated) particles.Dispose();
        if (latticeCells.IsCreated) latticeCells.Dispose();
        if (collisionVertices.IsCreated) collisionVertices.Dispose();
        if (collisionTriangles.IsCreated) collisionTriangles.Dispose();
        if (collisionEdges.IsCreated) collisionEdges.Dispose();
        if (collisionWrapers.IsCreated) collisionWrapers.Dispose();
        if (visualVertices.IsCreated) visualVertices.Dispose();
        if (visualVerticesCells.IsCreated) visualVerticesCells.Dispose();
    }


    private void OnEnable()
    {
        InitialiseSolver();
    }

    private void OnDisable()
    {
        DisposeLists();
    }

#endregion

#region Debugging

    // 8 fixed colors for parity channels
    static readonly Color[] s_ChannelColors =
    {
        Color.red,
        Color.green,
        Color.blue,
        Color.yellow,
        Color.cyan,
        Color.magenta,
        new Color(1f, 0.5f, 0f),   // orange
        Color.white
    };

    // Tetra order, the same with pre processor
    static readonly int4[] s_CellTets = new[]
    {
        new int4(0, 3, 6, 5),       // new int4(0, 1, 3, 7),
        new int4(0, 5, 6, 4),       // new int4(0, 3, 2, 7),
        new int4(0, 6, 3, 2),       // new int4(0, 2, 6, 7),
        new int4(5, 3, 6, 7),       // new int4(0, 6, 4, 7),
        new int4(0, 3, 5, 1)        // new int4(0, 4, 5, 7),
                                    // new int4(0, 5, 1, 7)
    };

    private void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;
        if (!particles.IsCreated || !latticeCells.IsCreated) return;

        for (int i = 0; i < latticeCells.Length && debugLatticeCells; i++)
        {
            LatticeCell cell = latticeCells[i];
            int totalCount = 0;
            int channel = 0;
            for (channel = 0; channel < 8; channel++) 
            {
                totalCount += globalChlCount[channel];
                if (i < totalCount) break;
            }

            Gizmos.color = s_ChannelColors[channel];

            // Get corner positions
            float3 p0 = particles[cell.v0].currentPosition;
            float3 p1 = particles[cell.v1].currentPosition;
            float3 p2 = particles[cell.v2].currentPosition;
            float3 p3 = particles[cell.v3].currentPosition;
            float3 p4 = particles[cell.v4].currentPosition;
            float3 p5 = particles[cell.v5].currentPosition;
            float3 p6 = particles[cell.v6].currentPosition;
            float3 p7 = particles[cell.v7].currentPosition;

            // Draw cube edges (12 edges)
            // bottom square
            Gizmos.DrawLine(p0, p1);
            Gizmos.DrawLine(p1, p3);
            Gizmos.DrawLine(p3, p2);
            Gizmos.DrawLine(p2, p0);

            // top square
            Gizmos.DrawLine(p4, p5);
            Gizmos.DrawLine(p5, p7);
            Gizmos.DrawLine(p7, p6);
            Gizmos.DrawLine(p6, p4);

            // vertical edges
            Gizmos.DrawLine(p0, p4);
            Gizmos.DrawLine(p1, p5);
            Gizmos.DrawLine(p2, p6);
            Gizmos.DrawLine(p3, p7);
        }

        for (int i = 0; i < collisionWrapers.Length; i++)
        {
            
        }
    }

#endregion
}

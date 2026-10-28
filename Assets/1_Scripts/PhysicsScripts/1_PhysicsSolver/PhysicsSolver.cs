using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using SolverStructs;
using System.Linq;
using Unity.Jobs;
using CollisionHandler;

public class PhysicsSolver : MonoBehaviour
{
    // --------------Public data--------------
    public float targetLatticeDimensions = 0.25f;
    public float substeps = 10;
    public float3 gravity = Physics.gravity;
    public float damping = 0;
    public bool debugLatticeCells = false, debugColMesh = false, debugVisVerts = false;
    public float numberOfCells;

    // --------------Solver data--------------
    private bool initialised = false;

    // --------------Lattice bodies data--------------
    private NativeList<float>  bodyMu;
    private NativeList<float>  bodyLambda;
    private List<BodyWraper>   bodyWrapers;

    // --------------Particle data--------------
    private NativeList<Particle>    particles;

    // --------------Cell data--------------
    private NativeList<LatticeCell> latticeCells;
    private NativeList<int>         latticeCellBodyRef;
    private NativeList<int>         activeCollCellBodyRef;
    private NativeList<int>         latticeColReferences;
    private NativeList<int>         activeCollisionLattices;
    private List<int>               latticeVisReferences;
    private int[]                   globalChlCount;

    // --------------Collision data--------------
    private NativeList<DynamicVertex>   collisionVertices;
    private NativeList<int3>            collisionTriangles;
    private NativeList<int2>            collisionEdges;
    private NativeList<CollisionWraper> collisionWrapers;

    // --------------Visual data--------------
    private NativeList<DynamicVertex> visualVerticesData;
    private NativeList<int>           visualVerticesOffset;
    private NativeList<float3>        visualVertices;
    private NativeList<int3>          visualWrapers;


#region Simulation

    private void FixedUpdate()
    {
        if (!initialised) return;
        JobHandle handle = default;
        float dt = Time.fixedDeltaTime / substeps;

        int particleCount = particles.Length;
        int visualWraperCount = visualWrapers.Length;
        int collisionWraperCount = collisionWrapers.Length;

        float cellVolume = targetLatticeDimensions * targetLatticeDimensions * targetLatticeDimensions;
        float bigTetVol = cellVolume * (1f/3f);
        float smallTetsVol = cellVolume * (1f/6f);

        FullCollisionJob.RunCollision(
            targetLatticeDimensions,
            0.025f,
            4,
            ref handle,
            particles,
            latticeCells,
            activeCollisionLattices,
            activeCollCellBodyRef,
            latticeColReferences,
            collisionVertices,
            collisionTriangles,
            collisionEdges,
            collisionWrapers
        );

        // for (int i = 0; i < substeps; i++)
        // {
        //     ParticleUpdateJob particleUpdateJob = new ParticleUpdateJob
        //     {
        //         gravity = gravity,
        //         dt = dt, particles = particles.AsArray()
        //     };
        //     handle = particleUpdateJob.Schedule(particleCount, 32, handle);

        //     int channelStart = 0;
        //     for (int chl = 0; chl < 8; chl++)
        //     {
        //         ShapeMatchingJob shapeMatchingJob = new ShapeMatchingJob
        //         {
        //             dt = dt, damping = damping, channelStart = channelStart,
        //             halfD = targetLatticeDimensions * 0.5f,
        //             bigTetVol = bigTetVol, smallTetsVol = smallTetsVol,
        //             latticeCells = latticeCells.AsArray(),
        //             latticeCellBodyRef = latticeCellBodyRef.AsArray(),
        //             bodyMu = bodyMu.AsArray(),
        //             bodyLambda = bodyLambda.AsArray(),
        //             particles = particles.AsArray()
        //         };
        //         handle = shapeMatchingJob.Schedule(globalChlCount[chl], 1, handle);
        //         channelStart += globalChlCount[chl];
        //     }

        //     FullCollisionJob.RunCollision(
        //         dt,
        //         targetLatticeDimensions,
        //         ref handle,
        //         particles,
        //         latticeCells,
        //         collisionVertices,
        //         collisionTriangles,
        //         collisionEdges,
        //         collisionWrapers
        //     );

        //     BasicPlaneColJob basicPlaneColJob = new BasicPlaneColJob
        //     {
        //         yPlanePos = 0f, latticeCells = latticeCells.AsArray(),
        //         collisionVertices = collisionVertices.AsArray(), collisionWrapers = collisionWrapers.AsArray(),
        //         particles = particles.AsArray()
        //     };
        //     handle = basicPlaneColJob.Schedule(collisionWraperCount, 1, handle);

        //     // channelStart = 0;
        //     // for (int chl = 0; chl < 8; chl++)
        //     // {
        //     //     TetDampingJob tetDampingJob = new TetDampingJob
        //     //     {
        //     //         dt = dt,
        //     //         damping = damping,
        //     //         channelStart = channelStart,
        //     //         latticeCells = latticeCells.AsArray(),
        //     //         particles = particles.AsArray()
        //     //     };
        //     //     handle = tetDampingJob.Schedule(globalChlCount[chl], 1, handle);
        //     //     channelStart += globalChlCount[chl];
        //     // }
        // }

        VisVertsUpdateJob visVertsUpdateJob = new VisVertsUpdateJob
        {
            particles = particles.AsArray(), latticeCells = latticeCells.AsArray(),
            visualVerticesData = visualVerticesData.AsArray(), visualVerticesOffset = visualVerticesOffset.AsArray(),
            visualVertices = visualVertices.AsArray(), visualWrapers = visualWrapers.AsArray()
        };
        handle = visVertsUpdateJob.Schedule(visualWraperCount, 1, handle);
        handle.Complete();

        // Update visuals
        foreach (var body in bodyWrapers)
        {
            // Calculate the average position
            float3 averagePos = Vector3.zero;
            for (int i = body.particlesRange.x; i < body.particlesRange.y; i++) averagePos += particles[i].currentPosition;
            averagePos /= body.particlesRange.y - body.particlesRange.x;
            body.transform.position = averagePos;
            body.transform.rotation = quaternion.identity;

            for (int i = body.visVerticesRange.x; i < body.visVerticesRange.y; i++) visualVertices[i] -= averagePos;

            body.visualMesh.SetVertices(visualVertices.AsArray(), body.visVerticesRange.x, body.visVerticesRange.y - body.visVerticesRange.x);
            body.visualMesh.RecalculateBounds();
            body.visualMesh.RecalculateNormals();
        }
    }

#endregion


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
        int initialVisWrapersCount = visualWrapers.Length;


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
        bodyWraper.particlesRange = new int2(initialParticleCount, particles.Length);


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
            
            // Fill the gap [insertionPos + existingCount .. insertionPos + existingCount + addCount)
            int writeStart = insertionPos + existingCount;
            latticeCells.InsertRange(writeStart, addCount);
            latticeCellBodyRef.InsertRange(writeStart, addCount);
            latticeColReferences.InsertRange(writeStart, addCount);
            latticeVisReferences.InsertRange(writeStart, Enumerable.Repeat(-1, addCount));

            for (int i = 0; i < addCount; i++)
            {
                // Create new cell and offset it's particle references
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
                latticeCellBodyRef[writeStart + i] = bodyWrapers.Count;
                // NativeList.InsertRange leaves the slots uninitialised, so -1 is written explicitly
                latticeColReferences[writeStart + i] = currentProcess.cellToColCell[currentLatticeCell] != -1
                    ? currentProcess.cellToColCell[currentLatticeCell] + initialColWraperCount
                    : -1;
                if (currentProcess.cellToVisCell[currentLatticeCell] != -1)
                    latticeVisReferences[writeStart + i] = currentProcess.cellToVisCell[currentLatticeCell] + initialVisWrapersCount;
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
            collisionTriangles.Add(currentProcess.collisionTriangles[i] + new int3(initialColVertCount));
        for (int i = 0; i < currentProcess.collisionEdges.Count; i++)
            collisionEdges.Add(currentProcess.collisionEdges[i] + new int2(initialColVertCount));
        for (int i = 0; i < currentProcess.collisionWrapers.Count; i++)
            collisionWrapers.Add(new CollisionWraper(
                currentProcess.collisionWrapers[i], // Copy from the reference to the new wraper
                initialColVertCount, initialColTriCount, initialColEdgeCount
            ));
        // Store collision data range
        bodyWraper.colVertsRange = new int2(initialColVertCount, collisionVertices.Length);
        bodyWraper.colTrisRange = new int2(initialColTriCount, collisionTriangles.Length);
        bodyWraper.colEdgesRange = new int2(initialColEdgeCount, collisionEdges.Length);
        bodyWraper.colWrapersRange = new int2(initialColWraperCount, collisionWrapers.Length);

        // Add visual data
        for (int i = 0; i < currentProcess.visualVerticesData.Length; i++)
        {
            visualVerticesData.Add(currentProcess.visualVerticesData[i]);
            visualVerticesOffset.Add(currentProcess.visualVerticesOffset[i] + initialVisVerticesCount);
            visualVertices.Add(float3.zero);
        }
        for (int i = 0; i < currentProcess.visualVerticesWrapers.Length; i++)
            visualWrapers.Add(currentProcess.visualVerticesWrapers[i] + new int3(initialVisVerticesCount));
        bodyWraper.visVerticesRange = new int2(initialVisVerticesCount, visualVertices.Length);
        bodyWraper.visWrapersRange = new int2(initialVisWrapersCount, visualWrapers.Length);


        // Add the new body wraper and body data
        bodyWrapers.Add(bodyWraper);
        bodyMu.Add(latticeBody.mu);
        bodyLambda.Add(latticeBody.lambda);
        
        latticeBody.transform.localScale = Vector3.one;
        numberOfCells = latticeCells.Length;

        // Update the cell references
        UpdateCellVertReferences();
        // Testing and finally finding that bug [05.03.2026]
        // HashSet<int> used = new HashSet<int>();
        // for (int i = 0; i < latticeColReferences.Count; i++)
        // {
        //     if (latticeColReferences[i] != -1 && !used.Add(latticeColReferences[i]))
        //         Debug.Log("Duplicate collision wrapper reference: " + latticeColReferences[i] + " ref: " + i);
        // }
    }

    private void UpdateCellVertReferences()
    {
        activeCollisionLattices.Clear();
        activeCollCellBodyRef.Clear();
        for (int i = 0; i < latticeCells.Length; i++)
        {
            if (latticeColReferences[i] != -1) {
                CollisionWraper wraperCopy = collisionWrapers[latticeColReferences[i]];
                wraperCopy.cellIdx = i;
                collisionWrapers[latticeColReferences[i]] = wraperCopy;

                // Mark the current cell as having a collision mesh
                activeCollisionLattices.Add(i);
                activeCollCellBodyRef.Add(latticeCellBodyRef[i]);
            }
            if (latticeVisReferences[i] != -1) 
            {
                int3 wraperCopy = visualWrapers[latticeVisReferences[i]];
                wraperCopy.x = i;
                visualWrapers[latticeVisReferences[i]] = wraperCopy;
            }
        }
    }

    private void InitialiseSolver()
    {
        // Only initialise once
        if (initialised) return;
        initialised = true;
        numberOfCells = 0;


        // Body data
        bodyMu = new NativeList<float>(Allocator.Persistent);
        bodyLambda = new NativeList<float>(Allocator.Persistent);
        bodyWrapers = new List<BodyWraper>();

        // Allocate all native lists
        // Particles allocation
        particles = new NativeList<Particle>(Allocator.Persistent);

        // Cell data allocation
        latticeCells = new NativeList<LatticeCell>(Allocator.Persistent);
        latticeCellBodyRef = new NativeList<int>(Allocator.Persistent);
        activeCollCellBodyRef = new NativeList<int>(Allocator.Persistent);
        globalChlCount = new int[8];
        for (int i = 0; i < 8; i++) globalChlCount[i] = 0;
        latticeColReferences = new NativeList<int>(Allocator.Persistent);
        activeCollisionLattices = new NativeList<int>(Allocator.Persistent);
        latticeVisReferences = new List<int>();

        // Collision data allocation
        collisionVertices = new NativeList<DynamicVertex>(Allocator.Persistent);
        collisionTriangles = new NativeList<int3>(Allocator.Persistent);
        collisionEdges = new NativeList<int2>(Allocator.Persistent);
        collisionWrapers = new NativeList<CollisionWraper>(Allocator.Persistent);

        // Visual data allocation
        visualVerticesData = new NativeList<DynamicVertex>(Allocator.Persistent);
        visualVerticesOffset = new NativeList<int>(Allocator.Persistent);
        visualVertices = new NativeList<float3>(Allocator.Persistent);
        visualWrapers = new NativeList<int3>(Allocator.Persistent);
    }

    private void DisposeLists()
    {
        if (!initialised) return;
        initialised = false;

        // Dispose of all native lists
        bodyMu.Dispose();
        bodyLambda.Dispose();
        particles.Dispose();
        latticeCells.Dispose();
        latticeCellBodyRef.Dispose();
        activeCollCellBodyRef.Dispose();
        latticeColReferences.Dispose();
        activeCollisionLattices.Dispose();
        collisionVertices.Dispose();
        collisionTriangles.Dispose();
        collisionEdges.Dispose();
        collisionWrapers.Dispose();
        visualVerticesData.Dispose();
        visualVerticesOffset.Dispose();
        visualVertices.Dispose();
        visualWrapers.Dispose();
    }


    private void OnEnable()
    {
        InitialiseSolver();
    }

    private void OnDisable()
    {
        // TODO :
        // Make it so that the disable function would be called outside this script
        // so that if multiple solvers use the collision solver they would not break
        FullCollisionJob.Disable();

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
            int channel;
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

        for (int i = 0; i < collisionWrapers.Length && debugColMesh; i++)
        {
            LatticeCell cell = latticeCells[collisionWrapers[i].cellIdx];
            int totalCount = 0;
            int channel;
            for (channel = 0; channel < 8; channel++) 
            {
                totalCount += globalChlCount[channel];
                if (collisionWrapers[i].cellIdx < totalCount) break;
            }
            Gizmos.color = s_ChannelColors[channel];

            for (int v = collisionWrapers[i].colVertsRange.x; v < collisionWrapers[i].colVertsRange.y; v++)
            {
                Gizmos.DrawSphere(GetDynamicVertexPos(cell, collisionVertices[v]), 0.005f);
            }

            for (int e = collisionWrapers[i].colEdgesRange.x; e < collisionWrapers[i].colEdgesRange.y; e++)
            {
                int2 edge = collisionEdges[e];

                Gizmos.DrawLine(GetDynamicVertexPos(cell, collisionVertices[edge.x]), GetDynamicVertexPos(cell, collisionVertices[edge.y]));
            }
        }

        for (int i = 0; i < visualWrapers.Length && debugVisVerts; i++)
        {
            LatticeCell cell = latticeCells[visualWrapers[i].x];
            int totalCount = 0;
            int channel;
            for (channel = 0; channel < 8; channel++) 
            {
                totalCount += globalChlCount[channel];
                if (visualWrapers[i].x < totalCount) break;
            }
            Gizmos.color = s_ChannelColors[channel];

            for (int v = visualWrapers[i].y; v < visualWrapers[i].z; v++)
            {
                Gizmos.DrawSphere(GetDynamicVertexPos(cell, visualVerticesData[v]), 0.005f);
            }
        }
    }

    float3 GetDynamicVertexPos(LatticeCell cell, DynamicVertex vertex)
    {
        int4 tet = s_CellTets[vertex.tet];

        int GetCellVertIndex(LatticeCell c, int localIndex)
        {
            switch (localIndex)
            {
                case 0: return c.v0;
                case 1: return c.v1;
                case 2: return c.v2;
                case 3: return c.v3;
                case 4: return c.v4;
                case 5: return c.v5;
                case 6: return c.v6;
                case 7: return c.v7;
                default: return -1;
            }
        }

        int vi0 = GetCellVertIndex(cell, tet.x);
        int vi1 = GetCellVertIndex(cell, tet.y);
        int vi2 = GetCellVertIndex(cell, tet.z);
        int vi3 = GetCellVertIndex(cell, tet.w);

        float3 p0 = particles[vi0].currentPosition; // or wherever you store positions
        float3 p1 = particles[vi1].currentPosition;
        float3 p2 = particles[vi2].currentPosition;
        float3 p3 = particles[vi3].currentPosition;

        float4 w = vertex.bary; // barycentric weights (w.x..w.w)

        return  p0 * w.x +
                p1 * w.y +
                p2 * w.z +
                p3 * w.w;
    }

#endregion
}
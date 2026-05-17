using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using SolverStructs;
using System.Linq;
using Unity.Jobs;

public class PhysicsSolver : MonoBehaviour
{
    // --------------Public data--------------
    public float targetLatticeDimensions = 1;
    public float substeps = 10;
    public float3 gravity = Physics.gravity;
    public float damping = 0;
    public bool debugLatticeCells = false, debugColMesh = false, debugVisVerts = false;

    // --------------Solver data--------------
    private bool initialised = false;

    // --------------Lattice bodies data--------------
    private int                numberOfBodies;
    private NativeList<int2>   bodyParticleResolution;
    private NativeList<float>  bodyDistConsCompliance;
    private NativeList<float>  bodyVolConsCompliance;
    // Main thread
    private List<BodyWraper>   bodyWrapers;

    // --------------Particle data--------------
    private NativeList<Particle>    particles;
    private NativeList<int>         particleMask;

    // --------------Cell data--------------
    private NativeList<int2>    latticeCells;
    private NativeSlice<int2>[] cellChannelSlices;
    // Main thread
    private List<int>           latticeColReferences;
    private List<int>           latticeVisReferences;
    private int[]               globalChlCount;

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
        float dtSq = dt * dt;
        float invDt = 1f / dt;
        float invDtSq = invDt * invDt;

        // Particle precomputed data
        float3 gravityDt = gravity * dt;

        // Distance constraint precomputed data
        float edgeLen     = targetLatticeDimensions;
        float faceDiagLen = targetLatticeDimensions * 1.41421356f;
        float bodyDiagLen = targetLatticeDimensions * 1.7320508f;

        int particleCount = particleMask.Length;
        int visualWraperCount = visualWrapers.Length;
        int collisionWraperCount = collisionWrapers.Length;

        for (int i = 0; i < substeps; i++)
        {
            ParticleUpdateJob particleUpdateJob = new ParticleUpdateJob
            {
                gravityDt = gravityDt,
                dt = dt,
                invDt = invDt,
                particles = particles.AsArray(),
                particleMask = particleMask.AsArray()
            };
            handle = particleUpdateJob.Schedule(particleCount, 32, handle);

            for (int chl = 0; chl < 8; chl++)
            {
                DistConsJob distConsJob = new DistConsJob
                {
                    invDt = invDt,
                    invDtSq = invDtSq,
                    dtSq = dtSq,
                    damping = damping,
                    edgeLen = edgeLen,
                    faceDiagLen = faceDiagLen,
                    bodyDiagLen = bodyDiagLen,
                    latticeCells = cellChannelSlices[chl],
                    bodyParticleResolution = bodyParticleResolution.AsArray(),
                    bodyDistConsCompliance = bodyDistConsCompliance.AsArray(),
                    particles = particles.AsArray()
                };
                handle = distConsJob.Schedule(globalChlCount[chl], 1, handle);
            }

            // int channelStart = 0;
            // for (int chl = 0; chl < 8; chl++)
            // {
            //     VolConsJob volConsJob = new VolConsJob
            //     {
            //         dt = dt, damping = damping, channelStart = channelStart, latticeCells = latticeCells.AsArray(),
            //         latticeCellBodyRef = latticeCellBodyRef.AsArray(), bodyLatticeDimensions = bodyLatticeDimensions.AsArray(),
            //         bodyVolConsCompliance = bodyVolConsCompliance.AsArray(), particles = particles.AsArray()
            //     };
            //     handle = volConsJob.Schedule(globalChlCount[chl], 1, handle);

            //     channelStart += globalChlCount[chl];
            // }

            // BasicPlaneColJob basicPlaneColJob = new BasicPlaneColJob
            // {
            //     yPlanePos = 0f, latticeCells = latticeCells.AsArray(),
            //     collisionVertices = collisionVertices.AsArray(), collisionWrapers = collisionWrapers.AsArray(),
            //     particles = particles.AsArray()
            // };
            // handle = basicPlaneColJob.Schedule(collisionWraperCount, 4, handle);
            // handle.Complete();
        }

        // VisVertsUpdateJob visVertsUpdateJob = new VisVertsUpdateJob
        // {
        //     particles = particles.AsArray(), latticeCells = latticeCells.AsArray(),
        //     visualVerticesData = visualVerticesData.AsArray(), visualVerticesOffset = visualVerticesOffset.AsArray(),
        //     visualVertices = visualVertices.AsArray(), visualWrapers = visualWrapers.AsArray()
        // };
        // handle = visVertsUpdateJob.Schedule(visualWraperCount, 1, handle);
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
        LatticePreprocessor bodyData = new LatticePreprocessor(latticeBody, targetLatticeDimensions);

        // Store initial list lengths
        int initialParticleCount = particles.Length;

        int initialColVertCount = collisionVertices.Length;
        int initialColTriCount = collisionTriangles.Length;
        int initialColEdgeCount = collisionEdges.Length;
        int initialColWraperCount = collisionWrapers.Length;

        int initialVisVerticesCount = visualVertices.Length;
        int initialVisWrapersCount = visualWrapers.Length;


        // Add particles
        float particleInvMass = 1f / (latticeBody.mass / bodyData.activeParticlesCount);
        int totalParticleCount = bodyData.particleResolution.x * 
            bodyData.particleResolution.y * bodyData.particleResolution.z;
        // Add all particles, even unused ones
        for (int i = 0; i < totalParticleCount; i++) particles.Add(new Particle());
        // Update the used particles using a mask
        for (int i = 0; i < bodyData.activeParticlesCount; i++)
        {
            // Calculate the particle position in the lattice
            int3 particleCoords = bodyData.to3D_Particle(bodyData.activeParticlesMask[i]);
            float3 posInLattice = bodyData.latticeNegativeCorner + particleCoords * bodyData.latticeDimensions;
            
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
            int offsetParticleIdx = initialParticleCount + bodyData.activeParticlesMask[i];
            particleMask.Add(offsetParticleIdx);
            particles[offsetParticleIdx] = newParticle;
        }


        // Add lattice cells
        int currentLatticeCell = 0;
        int insertionPos = 0; // where the next channel’s block starts globally
        for (int currentChl = 0; currentChl < 8; currentChl++)
        {
            int existingCount = globalChlCount[currentChl];  // how many cells of this color are already in latticeCells
            int addCount      = bodyData.chlCount[currentChl];
            if (addCount == 0)
            {
                // Skip to next channel, but the insertion position must jump
                insertionPos += existingCount;
                continue;
            }
            
            // Fill the gap [insertionPos + existingCount .. insertionPos + existingCount + addCount)
            int writeStart = insertionPos + existingCount;
            latticeCells.InsertRange(writeStart, addCount);
            latticeColReferences.InsertRange(writeStart, Enumerable.Repeat(-1, addCount));
            latticeVisReferences.InsertRange(writeStart, Enumerable.Repeat(-1, addCount));

            for (int i = 0; i < addCount; i++)
            {
                latticeCells[writeStart + i] = new int2(numberOfBodies, 
                    initialParticleCount + bodyData.finalLatticeCells[currentLatticeCell]);

                if (bodyData.cellToColCell[currentLatticeCell] != -1)
                    latticeColReferences[writeStart + i] = bodyData.cellToColCell[currentLatticeCell] + initialColWraperCount;
                if (bodyData.cellToVisCell[currentLatticeCell] != -1)
                    latticeVisReferences[writeStart + i] = bodyData.cellToVisCell[currentLatticeCell] + initialVisWrapersCount;
                
                currentLatticeCell++;
            }

            // Move insertionPos past this channel’s whole block (old + new)
            insertionPos += existingCount + addCount;
            globalChlCount[currentChl] += addCount;
        }


        // Add collision data
        for (int i = 0; i < bodyData.collisionVertices.Count; i++)
            collisionVertices.Add(bodyData.collisionVertices[i]);
        for (int i = 0; i < bodyData.collisionTriangles.Count; i++)
            collisionTriangles.Add(bodyData.collisionTriangles[i] + new int3(initialColVertCount));
        for (int i = 0; i < bodyData.collisionEdges.Count; i++)
            collisionEdges.Add(bodyData.collisionEdges[i] + new int2(initialColVertCount));
        for (int i = 0; i < bodyData.collisionWrapers.Count; i++)
            collisionWrapers.Add(new CollisionWraper(
                bodyData.collisionWrapers[i], // Copy from the reference to the new wraper
                initialColVertCount, initialColTriCount, initialColEdgeCount
            ));

        // Add visual data
        for (int i = 0; i < bodyData.visualVerticesData.Length; i++)
        {
            visualVerticesData.Add(bodyData.visualVerticesData[i]);
            visualVerticesOffset.Add(bodyData.visualVerticesOffset[i] + initialVisVerticesCount);
            visualVertices.Add(float3.zero);
        }
        for (int i = 0; i < bodyData.visualVerticesWrapers.Length; i++)
            visualWrapers.Add(bodyData.visualVerticesWrapers[i] + new int3(initialVisVerticesCount));


        // Add the new body wraper and body data
        numberOfBodies++;
        bodyParticleResolution.Add(bodyData.particleResolution.xy);
        bodyDistConsCompliance.Add(latticeBody.distanceCompliance);
        bodyVolConsCompliance.Add(latticeBody.volumeCompliance);

        int channelStart = 0;
        for (int chl = 0; chl < 8; chl++)
        {
            cellChannelSlices[chl] = new NativeSlice<int2>(latticeCells.AsArray(), channelStart, globalChlCount[chl]);
            channelStart += globalChlCount[chl];
        }

        // Update the cell references
        UpdateCellReferences();
        // Testing and finally finding that bug [05.03.2026]
        // HashSet<int> used = new HashSet<int>();
        // for (int i = 0; i < latticeColReferences.Count; i++)
        // {
        //     if (latticeColReferences[i] != -1 && !used.Add(latticeColReferences[i]))
        //         Debug.Log("Duplicate collision wrapper reference: " + latticeColReferences[i] + " ref: " + i);
        // }
    }

    private void UpdateCellReferences()
    {
        for (int i = 0; i < latticeCells.Length; i++)
        {
            if (latticeColReferences[i] != -1) {
                CollisionWraper wraperCopy = collisionWrapers[latticeColReferences[i]];
                wraperCopy.cellIdx = i;
                collisionWrapers[latticeColReferences[i]] = wraperCopy;
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

        // Body data
        numberOfBodies = 0;
        bodyParticleResolution = new NativeList<int2>(Allocator.Persistent);
        bodyDistConsCompliance = new NativeList<float>(Allocator.Persistent);
        bodyVolConsCompliance = new NativeList<float>(Allocator.Persistent);
        bodyWrapers = new List<BodyWraper>();

        // Allocate all native lists
        // Particles allocation
        particles = new NativeList<Particle>(Allocator.Persistent);
        particleMask = new NativeList<int>(Allocator.Persistent);

        // Cell data allocation
        latticeCells = new NativeList<int2>(Allocator.Persistent);
        cellChannelSlices = new NativeSlice<int2>[8];
        globalChlCount = new int[8];
        latticeColReferences = new List<int>();
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
        // Dispose of all native lists
        if (!initialised) return;
        bodyParticleResolution.Dispose();
        bodyDistConsCompliance.Dispose();
        bodyVolConsCompliance.Dispose();
        particles.Dispose();
        particleMask.Dispose();
        latticeCells.Dispose();
        cellChannelSlices = null;
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
            int2 cell = latticeCells[i];
            int2 res = bodyParticleResolution[cell.x];

            // Color assigment botch
            int totalCount = 0;
            int channel;
            for (channel = 0; channel < 8; channel++) 
            {
                totalCount += globalChlCount[channel];
                if (i < totalCount) break;
            }
            Gizmos.color = s_ChannelColors[channel];

            // Get corner positions
            float3 p0 = particles[cell.y].currentPosition;
            float3 p1 = particles[cell.y + 1].currentPosition;
            float3 p2 = particles[cell.y + res.x].currentPosition;
            float3 p3 = particles[cell.y + res.x + 1].currentPosition;
            float3 p4 = particles[cell.y + res.x * res.y].currentPosition;
            float3 p5 = particles[cell.y + res.x * res.y + 1].currentPosition;
            float3 p6 = particles[cell.y + res.x * res.y + res.x].currentPosition;
            float3 p7 = particles[cell.y + res.x * res.y + res.x + 1].currentPosition;

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
            int2 cell = latticeCells[collisionWrapers[i].cellIdx];

            // Color assigment botch
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
            int2 cell = latticeCells[visualWrapers[i].x];

            // Color assigment botch
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

    float3 GetDynamicVertexPos(int2 cell, DynamicVertex vertex)
    {
        int4 tet = s_CellTets[vertex.tet];

        int GetCellVertIndex(int localIndex)
        {
            int2 res = bodyParticleResolution[cell.x];

            switch (localIndex)
            {
                case 0: return cell.y;
                case 1: return cell.y + 1;
                case 2: return cell.y + res.x;
                case 3: return cell.y + res.x + 1;
                case 4: return cell.y + res.x * res.y;
                case 5: return cell.y + res.x * res.y + 1;
                case 6: return cell.y + res.x * res.y + res.x;
                case 7: return cell.y + res.x * res.y + res.x + 1;
                default: return -1;
            }
        }

        int vi0 = GetCellVertIndex(tet.x);
        int vi1 = GetCellVertIndex(tet.y);
        int vi2 = GetCellVertIndex(tet.z);
        int vi3 = GetCellVertIndex(tet.w);

        float3 p0 = particles[vi0].currentPosition;
        float3 p1 = particles[vi1].currentPosition;
        float3 p2 = particles[vi2].currentPosition;
        float3 p3 = particles[vi3].currentPosition;

        float4 w = vertex.bary;
        return  p0 * w.x +
                p1 * w.y +
                p2 * w.z +
                p3 * w.w;
    }

#endregion
}

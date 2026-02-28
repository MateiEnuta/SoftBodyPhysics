using UnityEngine;
using System;
using Unity.Mathematics;
using System.Collections.Generic;
using SolverStructs;

readonly struct Vec3Key : IEquatable<Vec3Key>
{
    private readonly int x, y, z;

    public Vec3Key(Vector3 v, float tol)
    {
        float inv = 1f / tol;
        x = Mathf.RoundToInt(v.x * inv);
        y = Mathf.RoundToInt(v.y * inv);
        z = Mathf.RoundToInt(v.z * inv);
    }

    public bool Equals(Vec3Key other) => x == other.x && y == other.y && z == other.z;
    public override bool Equals(object obj) => obj is Vec3Key other && Equals(other);
    public override int GetHashCode()
    {
        unchecked
        {
            int h = 17;
            h = h * 31 + x;
            h = h * 31 + y;
            h = h * 31 + z;
            return h;
        }
    }
}

[Serializable]
public class LatticePreprocessor
{
    static readonly int4[] s_CellTets = new[]
    {
        new int4(0, 1, 3, 7),
        new int4(0, 3, 2, 7),
        new int4(0, 2, 6, 7),
        new int4(0, 6, 4, 7),
        new int4(0, 4, 5, 7),
    };

    public float3 targetLatticeDimensions = new float3(1,1,1);

    // --------------Lattice data--------------
    // Helper data
    private int[] latticeCells;
    private List<int> activeCells;
    private int[] particles;
    // Persistent data
    private float3 latticeNegativeCorner;
    private int3 latticeResolution;
    private float3 latticeDimensions;
    private List<int> activeParticles;

    // --------------Collision data--------------
    // Helper data
    private Dictionary<Vec3Key, int>[] cellColParticlesIdx;
    private List<float3>[]             cellColParticles;
    private List<int3>[]               cellColTriangles;
    private HashSet<int2>[]            cellColEdges;
    private int totalColParticles;
    private int totalColTriangles;
    private int totalColEdges;
    // Persistent data
    private DynamicVertex[] collisionVertices;
    private int3[]            collisionTriangles;
    private int2[]            collisionEdges;
    private CollisionWraper[] collisionWrapers;

    // --------------Visual data--------------

    public void ProcessLatticeBody(LatticeBody latticeBody)
    {
    #region Setup
        if (latticeBody.visualMesh == null) return;

        // Calculate the lattice parameters
        Bounds visualMeshBounds = latticeBody.visualMesh.bounds;
        // enlarge the bounds to avoid precision errors
        float EPSILON = 1e-4f;
        float3 boundsSize = (float3)visualMeshBounds.size + new float3(2 * EPSILON, 2 * EPSILON, 2 * EPSILON);
        latticeNegativeCorner = (float3)visualMeshBounds.min - new float3(EPSILON, EPSILON, EPSILON);
        float3 localTargetDimensions = latticeBody.transform.InverseTransformDirection(targetLatticeDimensions);

        // Prevent division by zero
        Vector3 safeTarget = new Vector3(
            Mathf.Max(localTargetDimensions.x, 0.1f),
            Mathf.Max(localTargetDimensions.y, 0.1f),
            Mathf.Max(localTargetDimensions.z, 0.1f));

        // Closest integer count per axis
        int nx = Mathf.Max(1, Mathf.RoundToInt(boundsSize.x / safeTarget.x));
        int ny = Mathf.Max(1, Mathf.RoundToInt(boundsSize.y / safeTarget.y));
        int nz = Mathf.Max(1, Mathf.RoundToInt(boundsSize.z / safeTarget.z));

        // Assign final lattice data
        latticeResolution = new int3(nx, ny, nz);
        latticeDimensions = new float3(
            boundsSize.x / nx,
            boundsSize.y / ny,
            boundsSize.z / nz
        );

        // Lattice setup
        int maxNumberOfCells = latticeResolution.x * latticeResolution.y * latticeResolution.z;
        latticeCells = new int[maxNumberOfCells];
        for (int i = 0; i < maxNumberOfCells; i++) latticeCells[i] = -1;
        activeCells = new List<int>(maxNumberOfCells / 2);
        // Collision data used pre cell
        cellColParticlesIdx = new Dictionary<Vec3Key, int>[maxNumberOfCells];
        cellColParticles = new List<float3>[maxNumberOfCells];
        cellColTriangles = new List<int3>[maxNumberOfCells];
        cellColEdges = new HashSet<int2>[maxNumberOfCells];
        totalColParticles = 0;
        totalColTriangles = 0;
        totalColEdges = 0;
        // Particle setup
        int maxNumberOfParticles = (latticeResolution.x + 1) * (latticeResolution.y + 1) * (latticeResolution.z + 1);
        particles = new int[maxNumberOfParticles];
        for (int i = 0; i < maxNumberOfParticles; i++) particles[i] = -1;
        activeParticles = new List<int>(maxNumberOfParticles / 2);
    #endregion
        

    #region Collision Mesh Creator
        // Try getting the collision mesh if there is one
        Mesh collisionMesh = latticeBody.collisionMesh == null ? latticeBody.visualMesh : latticeBody.collisionMesh;
        Vector3[] meshCollisionVertices = collisionMesh.vertices;
        int[] meshCollisionTriangles = collisionMesh.triangles;
        // Make sure the collision mesh is as small as possible
        MeshMath.RemoveDuplicateVertices(ref meshCollisionVertices, ref meshCollisionTriangles, 1e-06f);

        // Pre allocate data for the mesh cutter
        float3[] xSliceEdges = new float3[5];
        int xSliceEdgesCount;
        float3[] ySliceEdges = new float3[7];
        int ySliceEdgesCount;
        float3[] zSliceEdges = new float3[9];
        int zSliceEdgesCount;
        // Run through each ref triangle and cut it
        for (int t = 0; t < meshCollisionTriangles.Length; t+=3)
        {
            float3 t1 = meshCollisionVertices[meshCollisionTriangles[t]];
            float3 t2 = meshCollisionVertices[meshCollisionTriangles[t+1]];
            float3 t3 = meshCollisionVertices[meshCollisionTriangles[t+2]];

            // Calculate the min and max corners of the bounding box
            float3 minBB = new float3(
                t1.x < (t2.x < t3.x ? t2.x : t3.x) ? t1.x : (t2.x < t3.x ? t2.x : t3.x),
                t1.y < (t2.y < t3.y ? t2.y : t3.y) ? t1.y : (t2.y < t3.y ? t2.y : t3.y),
                t1.z < (t2.z < t3.z ? t2.z : t3.z) ? t1.z : (t2.z < t3.z ? t2.z : t3.z)
            );
            float3 maxBB = new float3(
                t1.x > (t2.x > t3.x ? t2.x : t3.x) ? t1.x : (t2.x > t3.x ? t2.x : t3.x),
                t1.y > (t2.y > t3.y ? t2.y : t3.y) ? t1.y : (t2.y > t3.y ? t2.y : t3.y),
                t1.z > (t2.z > t3.z ? t2.z : t3.z) ? t1.z : (t2.z > t3.z ? t2.z : t3.z)
            );

            // Calculates the min negative corner idxs
            float3 minPosInLattice = minBB - latticeNegativeCorner;
            int3 minNegativeCorner = (int3)(minPosInLattice / latticeDimensions);

            // Calculates the max negative corner idxs
            float3 maxPosInLattice = maxBB - latticeNegativeCorner;
            int3 maxNegativeCorner = (int3)(maxPosInLattice / latticeDimensions);

        #region Cell Cutter
            // Loop though each cell touched by the triangle bounding box
            // And cut the triangle for each on of them
            static void TryCutEdge(float3[] slicedEdges, ref int currentEdgeIdx,
                float3 v0, float3 v1,  float planeValueBack,  float planeValueFront, int axis)
            {
                // Tries to first add the current vertex then the cut edge if it exists
                if (v0[axis] - planeValueBack > -1e-06f &&
                    v0[axis] - planeValueFront < 1e-06f) slicedEdges[currentEdgeIdx++] = v0;
                // Test for plane intersections
                float3 edgeDir = v1 - v0;
                if (edgeDir[axis] != 0)
                {
                    float t1 = (planeValueBack - v0[axis]) / edgeDir[axis];
                    float t2 = (planeValueFront - v0[axis]) / edgeDir[axis];

                    // Swap to preserve clock wise orientation
                    if (t2 < t1) (t2, t1) = (t1, t2);

                    // Add them only if they are exactly between planes
                    if (t1 > 1e-06f && t1 - 1 < 1e-06f) 
                        slicedEdges[currentEdgeIdx++] = v0 + t1 * edgeDir;
                    if (t2 > 1e-06f && t2 - 1 < 1e-06f) 
                        slicedEdges[currentEdgeIdx++] = v0 + t2 * edgeDir;
                }
            }
            for (int x = minNegativeCorner.x; x <= maxNegativeCorner.x; x++)
            {
                float xPlaneValueBack = latticeNegativeCorner.x + latticeDimensions.x * x;
                float xPlaneValueFront = latticeNegativeCorner.x + latticeDimensions.x * (x + 1);
                xSliceEdgesCount = 0;

                // Test each triangle edge and
                // biuld the remaining tri array
                // v0-v1 edge
                TryCutEdge(xSliceEdges, ref xSliceEdgesCount, t1, t2, xPlaneValueBack, xPlaneValueFront, 0);
                // v1-v2 edge
                TryCutEdge(xSliceEdges, ref xSliceEdgesCount, t2, t3, xPlaneValueBack, xPlaneValueFront, 0);
                // v2-v0 edge
                TryCutEdge(xSliceEdges, ref xSliceEdgesCount, t3, t1, xPlaneValueBack, xPlaneValueFront, 0);

                for (int y = minNegativeCorner.y; y <= maxNegativeCorner.y; y++)
                {
                    float yPlaneValueBack = latticeNegativeCorner.y + latticeDimensions.y * y;
                    float yPlaneValueFront = latticeNegativeCorner.y + latticeDimensions.y * (y + 1);
                    ySliceEdgesCount = 0;

                    // Loop though each edge of the x slice
                    // cutting and adding it to the y slice list
                    for (int i = 0; i < xSliceEdgesCount; i++)
                    {
                        float3 e0 = xSliceEdges[i];
                        float3 e1 = i+1 == xSliceEdgesCount ? xSliceEdges[0] : xSliceEdges[i+1];
                        
                        // Cut if possible
                        TryCutEdge(ySliceEdges, ref ySliceEdgesCount, e0, e1, yPlaneValueBack, yPlaneValueFront, 1);
                    }

                    for (int z = minNegativeCorner.z; z <= maxNegativeCorner.z; z++)
                    {
                        float zPlaneValueBack = latticeNegativeCorner.z + latticeDimensions.z * z;
                        float zPlaneValueFront = latticeNegativeCorner.z + latticeDimensions.z * (z + 1);
                        zSliceEdgesCount = 0;

                        // Loop though each edge of the y slice
                        // cutting and adding it to the z slice list
                        for (int i = 0; i < ySliceEdgesCount; i++)
                        {
                            float3 e0 = ySliceEdges[i];
                            float3 e1 = i+1 == ySliceEdgesCount ? ySliceEdges[0] : ySliceEdges[i+1];

                            // Cut if possible
                            TryCutEdge(zSliceEdges, ref zSliceEdgesCount, e0, e1, zPlaneValueBack, zPlaneValueFront, 2);
                        }
                        // Discard it it doesn't make a triangle
                        if (zSliceEdgesCount <= 2) continue;

                        // Double checks if the cell intersects the trianglex
                        int3 cellCoords = new int3(x, y, z);
                        int cellID = to1D(cellCoords);
                        AddCell(cellID);
                        if (cellColParticlesIdx[cellID] == null)
                        {
                            cellColParticlesIdx[cellID] = new Dictionary<Vec3Key, int>();
                            cellColParticles[cellID] = new List<float3>();
                            cellColTriangles[cellID] = new List<int3>();
                            cellColEdges[cellID] = new HashSet<int2>();
                        }
                        
                        // Add each particle individually
                        for (int i = 0; i < zSliceEdgesCount; i++)
                        {
                            if (cellColParticlesIdx[cellID].TryAdd(new Vec3Key(zSliceEdges[i], 1e-06f), cellColParticles[cellID].Count))
                            {
                                totalColParticles++;
                                cellColParticles[cellID].Add(zSliceEdges[i]);
                            }
                        }

                        // Cache the main vertex
                        int v1 = cellColParticlesIdx[cellID][new Vec3Key(zSliceEdges[0], 1e-06f)];
                        for (int i = 1; i < zSliceEdgesCount - 1; i++)
                        {
                            // Compute the next 2 vertices
                            int v2 = cellColParticlesIdx[cellID][new Vec3Key(zSliceEdges[i], 1e-06f)];
                            int v3 = cellColParticlesIdx[cellID][new Vec3Key(zSliceEdges[i+1], 1e-06f)];
                            
                            // Add the triangle
                            cellColTriangles[cellID].Add(new int3(v1, v2, v3));
                            totalColTriangles++;

                            // Add the edges in ascending index order
                            if (cellColEdges[cellID].Add(v1 < v2 ? new int2(v1, v2) : new int2(v2, v1))) totalColEdges++;
                            if (cellColEdges[cellID].Add(v2 < v3 ? new int2(v2, v3) : new int2(v3, v2))) totalColEdges++;
                            if (cellColEdges[cellID].Add(v3 < v1 ? new int2(v3, v1) : new int2(v1, v3))) totalColEdges++;
                        }
                    }
                }
            }
        #endregion
        }

        // Setup all permanent collision data
        collisionVertices  = new DynamicVertex[totalColParticles];
        collisionTriangles = new int3[totalColTriangles];
        collisionEdges     = new int2[totalColEdges];
        collisionWrapers   = new CollisionWraper[activeCells.Count];

        // For knowing at which element we are
        int currentColVertCount   = 0;
        int currentColTriCount    = 0;
        int currentColEdgeCount   = 0;
        int currentColWraperCount = 0;

        for (int i = 0; i < activeCells.Count; i++)
        {
            // Store the initial list count
            int currentCellID = activeCells[i];
            int initialColVertCount   = currentColVertCount;
            int initialColTriCount    = currentColTriCount;
            int initialColEdgeCount   = currentColEdgeCount;

            // Loop through each collision vertex and store a dynamic vertex
            for (int j = 0; j < cellColParticles[currentCellID].Count; j++)
            {
                collisionVertices[currentColVertCount++] = GetCellDynamicVertex(
                    currentCellID, 
                    cellColParticles[currentCellID][j]
                );
            }

            // Store the triangles
            for (int j = 0; j < cellColTriangles[currentCellID].Count; j++)
            {
                collisionTriangles[currentColTriCount++] = cellColTriangles[currentCellID][j];
            }

            // Store the edges
            foreach (int2 edge in cellColEdges[currentCellID]) collisionEdges[currentColEdgeCount++] = edge;

            // Make a wraper that has this actice cell index
            collisionWrapers[currentColWraperCount++] = new CollisionWraper(
                i,
                new int2(initialColVertCount, currentColVertCount - initialColVertCount),
                new int2(initialColTriCount, currentColTriCount - initialColTriCount),
                new int2(initialColEdgeCount, currentColEdgeCount - initialColEdgeCount)
            );
        }

        // GC helper data
        cellColParticlesIdx = null;
        cellColParticles = null;
        cellColTriangles = null;
        cellColEdges = null;

    #endregion
    #region Visual Mesh Creator
        
    #endregion
    }

#region Common Helper Functions
    public int to1D(int3 pos)
    {
        return (pos.z * latticeResolution.x * latticeResolution.y) + (pos.y * latticeResolution.x) + pos.x;
    }

    public int3 to3D(int idx)
    {
        int3 result = int3.zero;
        result.z = idx / (latticeResolution.x * latticeResolution.y);
        idx -= result.z * latticeResolution.x * latticeResolution.y;
        result.y = idx / latticeResolution.x;
        result.x = idx % latticeResolution.x;

        return result;
    }

    public int to1D_Particle(int3 pos)
    {
        int3 particleResolution = latticeResolution + new int3(1, 1, 1);
        return (pos.z * particleResolution.x * particleResolution.y) + (pos.y * particleResolution.x) + pos.x;
    }

    public int3 to3D_Particle(int idx)
    {
        int3 particleResolution = latticeResolution + new int3(1, 1, 1);
        int3 result = int3.zero;
        result.z = idx / (particleResolution.x * particleResolution.y);
        idx -= result.z * particleResolution.x * particleResolution.y;
        result.y = idx / particleResolution.x;
        result.x = idx % particleResolution.x;

        return result;
    }

    private void AddCell(int cellID)
    {
        if (latticeCells[cellID] != -1) return;
        latticeCells[cellID] = activeCells.Count;
        activeCells.Add(cellID);

        // Add each cell corner particle
        int3 baseCorner = to3D(cellID);

        // 0: (0,0,0)
        int3 currentCorner = baseCorner;
        ActivateParticle(currentCorner);

        // 1: (1,0,0)
        currentCorner = baseCorner + new int3(1, 0, 0);
        ActivateParticle(currentCorner);

        // 2: (0,1,0)
        currentCorner = baseCorner + new int3(0, 1, 0);
        ActivateParticle(currentCorner);

        // 3: (1,1,0)
        currentCorner = baseCorner + new int3(1, 1, 0);
        ActivateParticle(currentCorner);

        // 4: (0,0,1)
        currentCorner = baseCorner + new int3(0, 0, 1);
        ActivateParticle(currentCorner);

        // 5: (1,0,1)
        currentCorner = baseCorner + new int3(1, 0, 1);
        ActivateParticle(currentCorner);

        // 6: (0,1,1)
        currentCorner = baseCorner + new int3(0, 1, 1);
        ActivateParticle(currentCorner);

        // 7: (1,1,1)
        currentCorner = baseCorner + new int3(1, 1, 1);
        ActivateParticle(currentCorner);
    }

    private void ActivateParticle(int3 currentCorner)
    {
        int particleID = to1D_Particle(currentCorner);
        if (particles[particleID] == -1)
        {
            particles[particleID] = activeParticles.Count;
            activeParticles.Add(particleID);
        }
    }

    private DynamicVertex GetCellDynamicVertex(int cellID, float3 vertexPos)
    {
        int3 baseCorner = to3D(cellID);
        float3 cellOrigin = latticeNegativeCorner + (float3)baseCorner * latticeDimensions;
        float3 p = vertexPos - cellOrigin;

        // Precompute corners as locals (no array alloc)
        float3 c0 = new float3(0, 0, 0);
        float3 c1 = new float3(latticeDimensions.x, 0, 0);
        float3 c2 = new float3(0, latticeDimensions.y, 0);
        float3 c3 = new float3(latticeDimensions.x, latticeDimensions.y, 0);
        float3 c4 = new float3(0, 0, latticeDimensions.z);
        float3 c5 = new float3(latticeDimensions.x, 0, latticeDimensions.z);
        float3 c6 = new float3(0, latticeDimensions.y, latticeDimensions.z);
        float3 c7 = new float3(latticeDimensions.x, latticeDimensions.y, latticeDimensions.z);

        const float eps = 1e-6f;

        // Simple function for getting the cached corners
        float3 GetParticle(int idx)
        {
            return idx == 0 ? c0 :
                   idx == 1 ? c1 :
                   idx == 2 ? c2 :
                   idx == 3 ? c3 :
                   idx == 4 ? c4 :
                   idx == 5 ? c5 :
                   idx == 6 ? c6 : c7;
        }

        // Loop though each tetra and try to find a match for the given vertex
        for (byte tetIndex = 0; tetIndex < s_CellTets.Length; tetIndex++)
        {
            int4 t = s_CellTets[tetIndex];

            float3 a = GetParticle(t.x);
            float3 b = GetParticle(t.y);
            float3 c = GetParticle(t.z);
            float3 d = GetParticle(t.w);

            float3 e0 = a - d;
            float3 e1 = b - d;
            float3 e2 = c - d;
            float3 r  = p - d;

            float3x3 M = new float3x3(e0, e1, e2);
            float3 lambda012 = math.mul(math.inverse(M), r);
            float lambda3 = 1f - (lambda012.x + lambda012.y + lambda012.z);
            float4 bary = new float4(lambda012, lambda3);

            if (bary.x >= -eps && bary.y >= -eps && bary.z >= -eps && bary.w >= -eps)
                return new DynamicVertex(bary, tetIndex);
        }

        Debug.LogWarning($"GetCellDynamicVertex: no tetra found for cell {cellID}, pos {vertexPos}");
        return new DynamicVertex(new float4(1, 0, 0, 0), 0);
    }

#endregion
}

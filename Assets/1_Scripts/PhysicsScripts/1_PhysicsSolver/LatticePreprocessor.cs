using UnityEngine;
using System;
using Unity.Mathematics;
using System.Collections.Generic;

[Serializable]
public class LatticePreprocessor
{
    public float3 targetLatticeDimensions = new float3(1,1,1);

    // Persistent data
    private LatticeBody latticeBodyRef;
    // Lattice data
    private float3 latticeNegativeCorner;
    private int3 latticeResolution;
    private float3 latticeDimensions;
    private int[] latticeCells;
    private List<int> activeCells;
    private int[] particlesIdx;
    private List<float3> activeParticles;
    // Collision data

    // Visual data

    public void ProcessLatticeBody(LatticeBody latticeBody)
    {
#region Setup
        if (latticeBody.visualMesh == null) return;
        latticeBodyRef = latticeBody;

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
        latticeCells = new int[latticeResolution.x * latticeResolution.y * latticeResolution.z];
        for (int i = 0; i < latticeCells.Length; i++) latticeCells[i] = -1;
        activeCells = new List<int>();
        // Paritcle setup
        particlesIdx = new int[(latticeResolution.x + 1) * (latticeResolution.y + 1) * (latticeResolution.z + 1)];
        for (int i = 0; i < particlesIdx.Length; i++) particlesIdx[i] = -1;
        activeParticles = new List<float3>();
#endregion
        

#region Collision Mesh Creator
        // Try getting the collision mesh if there is one
        Mesh collisionMesh = latticeBody.collisionMesh == null ? latticeBody.visualMesh : latticeBody.collisionMesh;
        Vector3[] meshCollisionVertices = collisionMesh.vertices;
        int[] meshCollisionTriangles = collisionMesh.triangles;
        // Make sure the collision mesh is as small as possible
        MeshMath.RemoveDuplicateVertices(ref meshCollisionVertices, ref meshCollisionTriangles, 1e-06f);

        // Generate new lists for possible collision gemetry
        List<float3> possibleCollVertices = new List<float3>();
        List<int> possibleColVerCellID = new List<int>();
        List<int4> possibleColTriangles = new List<int4>();

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
                        ActivateCell(cellID);
                        
                        // Create new triangles
                        int initialPossVertCount = possibleCollVertices.Count;
                        possibleCollVertices.Add(zSliceEdges[0]);
                        possibleColVerCellID.Add(cellID);
                        for (int i = 1; i < zSliceEdgesCount - 1; i++)
                        {
                            // Add a new triangle formed by the first vertex,
                            // the current one and the next vertex
                            possibleColTriangles.Add(new int4(
                                initialPossVertCount,           // Pivot vertex
                                possibleCollVertices.Count,     // Current vertex
                                possibleCollVertices.Count + 1, // Next vertex
                                cellID                          // current cellID
                            ));

                            // Add the new point
                            possibleCollVertices.Add(zSliceEdges[i]);
                            possibleColVerCellID.Add(cellID);
                        }
                        // Add the last vertex
                        possibleCollVertices.Add(zSliceEdges[zSliceEdgesCount - 1]);
                        possibleColVerCellID.Add(cellID);
                    }
                }
            }
#endregion
        }
        // Clean up to avoid duplicate collision vertices
        MeshMath.RemoveDuplicateVerticesPerCell(possibleCollVertices, possibleColVerCellID, possibleColTriangles, 1e-06f);

        // Radix sort the arrays by the cellID
        HelperAlgorithms.SortVertsAndRemapTris(possibleColVerCellID, possibleCollVertices, possibleColTriangles);
        
        // Biuld edges in ascending order by the cellID (the .z parameter)
        List<int3> possibleCollisionEdges = MeshMath.BuildCellSortedEdges(possibleColTriangles);


#endregion
#region Visual Mesh Creator
        
#endregion
    }

#region Common Helper Functions
    private int to1D(int3 pos)
    {
        return (pos.z * latticeResolution.x * latticeResolution.y) + (pos.y * latticeResolution.x) + pos.x;
    }

    private int3 to3D(int idx)
    {
        int3 result = int3.zero;
        result.z = idx / (latticeResolution.x * latticeResolution.y);
        idx -= result.z * latticeResolution.x * latticeResolution.y;
        result.y = idx / latticeResolution.x;
        result.x = idx % latticeResolution.x;

        return result;
    }

    private int to1D_Particle(int3 pos)
    {
        int3 particleResolution = latticeResolution + new int3(1, 1, 1);
        return (pos.z * particleResolution.x * particleResolution.y) + (pos.y * particleResolution.x) + pos.x;
    }

    private void ActivateCell(int cellID)
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
        if (particlesIdx[to1D_Particle(currentCorner)] == -1)
        {
            particlesIdx[to1D_Particle(currentCorner)] = activeParticles.Count;
            activeParticles.Add(latticeNegativeCorner + latticeDimensions * currentCorner);
        }
    }
#endregion
}

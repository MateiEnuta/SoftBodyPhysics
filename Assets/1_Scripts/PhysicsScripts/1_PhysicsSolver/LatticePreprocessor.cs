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
        new int4(0, 3, 6, 5),       // new int4(0, 1, 3, 7),
        new int4(0, 5, 6, 4),       // new int4(0, 3, 2, 7),
        new int4(0, 6, 3, 2),       // new int4(0, 2, 6, 7),
        new int4(5, 3, 6, 7),       // new int4(0, 6, 4, 7),
        new int4(0, 3, 5, 1)        // new int4(0, 4, 5, 7),
                                    // new int4(0, 5, 1, 7)
    };

    // --------------Lattice data--------------
    // Helper data
    private int[] latticeCells;
    private List<int> activeCells;
    private int[] particles;
    // Persistent data
    public LatticeBody latticeBody { get; private set; }
    public float3 latticeNegativeCorner { get; private set; }
    public int3 latticeResolution { get; private set; }
    public float3 latticeDimensions { get; private set; }
    public List<int>     activeParticles { get; private set; }
    public LatticeCell[] finalLatticeCells { get; private set; }
    public int[]         latticeCellCoords { get; private set; }
    public int[]         chlCount { get; private set; }
    public int[]         cellToColCell { get; private set; }
    public int[]         cellToVisCell { get; private set; }

    // --------------Collision data--------------
    // Persistent data
    public List<DynamicVertex>   collisionVertices { get; private set; }
    public List<int3>            collisionTriangles { get; private set; }
    public List<int2>            collisionEdges { get; private set; }
    public List<CollisionWraper> collisionWrapers { get; private set; }

    // --------------Visual data--------------
    // Persistent data
    public DynamicVertex[] visualVerticesData { get; private set; }
    public int[]           visualVerticesOffset { get; private set; }
    public int3[]          visualVerticesWrapers { get; private set; }

    public LatticePreprocessor(LatticeBody latticeBody, float3 targetLatticeDimensions)
    {
    #region Setup
        if (latticeBody.visualMesh == null) return;
        this.latticeBody = latticeBody;

        // 1) World-space bounds
        Bounds visualMeshBounds = latticeBody.visualMesh.bounds;
        latticeBody.visualMesh.MarkDynamic();
        latticeBody.visualMesh.RecalculateBounds();
        latticeBody.visualMesh.RecalculateNormals();
        float EPSILON = 1e-4f;

        float3 boundsSize = (float3)visualMeshBounds.size
                            + new float3(2 * EPSILON, 2 * EPSILON, 2 * EPSILON);
        latticeNegativeCorner = (float3)visualMeshBounds.min
                                - new float3(EPSILON, EPSILON, EPSILON);

        float3 localTargetDimensions = targetLatticeDimensions / latticeBody.transform.lossyScale;
        float3 safeTarget = new float3(
            math.max(localTargetDimensions.x, 0.001f),
            math.max(localTargetDimensions.y, 0.001f),
            math.max(localTargetDimensions.z, 0.001f));

        // 2) Compute integer cell counts in world space
        int nx = math.clamp((int)math.round(boundsSize.x / safeTarget.x), 1, 40);
        int ny = math.clamp((int)math.round(boundsSize.y / safeTarget.y), 1, 40);
        int nz = math.clamp((int)math.round(boundsSize.z / safeTarget.z), 1, 40);

        // 3) Final lattice data (world-space cell size)
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
        chlCount = new int[8];
        for (int i = 0; i < 8; i++) chlCount[i] = 0;
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

        // Allocate helper hashes
        Dictionary<int2, int> cellToTri   = new Dictionary<int2, int>();
        int[]                 cellStrides = new int[maxNumberOfCells];

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

            // Narrow-phase tri vs box test, assuming AABBs already overlap.
            bool TriangleIntersectsCell_Narrow(float3 v0, float3 v1, float3 v2,
                                            float3 boxCenter, float3 halfExtents)
            {
                // Move triangle into box-centered space
                v0 -= boxCenter;
                v1 -= boxCenter;
                v2 -= boxCenter;

                float3 e0 = v1 - v0;
                float3 e1 = v2 - v1;
                float3 e2 = v0 - v2;

                // ---- 1) Triangle plane vs box ----
                float3 normal = math.cross(e0, e1);
                if (math.all(math.abs(normal) < 1e-8f))
                    return true; // degenerate triangle, treat as hit

                float p0 = math.dot(v0, normal);
                float p1 = math.dot(v1, normal);
                float p2 = math.dot(v2, normal);
                float minP = math.min(p0, math.min(p1, p2));
                float maxP = math.max(p0, math.max(p1, p2));

                float rPlane =
                    halfExtents.x * math.abs(normal.x) +
                    halfExtents.y * math.abs(normal.y) +
                    halfExtents.z * math.abs(normal.z);

                if (minP > rPlane || maxP < -rPlane)
                    return false;

                // ---- 2) Edge × box-axis tests (no arrays, fully unrolled) ----
                // Helper local function
                bool TestAxis(float3 axis)
                {
                    if (math.all(math.abs(axis) < 1e-8f))
                        return true; // degenerate axis, skip

                    float t0 = math.dot(v0, axis);
                    float t1 = math.dot(v1, axis);
                    float t2 = math.dot(v2, axis);
                    float triMin = math.min(t0, math.min(t1, t2));
                    float triMax = math.max(t0, math.max(t1, t2));

                    float r =
                        halfExtents.x * math.abs(axis.x) +
                        halfExtents.y * math.abs(axis.y) +
                        halfExtents.z * math.abs(axis.z);

                    return !(triMin > r || triMax < -r);
                }

                // Box axes: (1,0,0), (0,1,0), (0,0,1)
                // e0
                if (!TestAxis(math.cross(e0, new float3(1,0,0)))) return false;
                if (!TestAxis(math.cross(e0, new float3(0,1,0)))) return false;
                if (!TestAxis(math.cross(e0, new float3(0,0,1)))) return false;

                // e1
                if (!TestAxis(math.cross(e1, new float3(1,0,0)))) return false;
                if (!TestAxis(math.cross(e1, new float3(0,1,0)))) return false;
                if (!TestAxis(math.cross(e1, new float3(0,0,1)))) return false;

                // e2
                if (!TestAxis(math.cross(e2, new float3(1,0,0)))) return false;
                if (!TestAxis(math.cross(e2, new float3(0,1,0)))) return false;
                if (!TestAxis(math.cross(e2, new float3(0,0,1)))) return false;

                return true;
            }
            for (int x = minNegativeCorner.x; x <= maxNegativeCorner.x; x++)
            {
                for (int y = minNegativeCorner.y; y <= maxNegativeCorner.y; y++)
                {
                    for (int z = minNegativeCorner.z; z <= maxNegativeCorner.z; z++)
                    {
                        int3 cellCoords = new int3(x, y, z);
                        float3 cellNegativeCorner = latticeNegativeCorner + cellCoords * latticeDimensions;
                        float3 cellMiddle = cellNegativeCorner + 0.5f * latticeDimensions;

                        if (!TriangleIntersectsCell_Narrow(t1, t2, t3, cellMiddle, latticeDimensions * 0.5f)) continue;

                        int cellID = to1D(cellCoords);
                        AddCell(cellID);

                        // Add cell reference
                        cellToTri.Add(new int2(cellID, cellStrides[cellID]++), t);
                    }
                }
            }
        }

        // Pre allocate data for the mesh cutter
        float3[] xSliceEdges = new float3[5];
        int xSliceEdgesCount;
        float3[] ySliceEdges = new float3[7];
        int ySliceEdgesCount;
        float3[] zSliceEdges = new float3[9];
        int zSliceEdgesCount;

        collisionVertices = new List<DynamicVertex>();
        collisionTriangles = new List<int3>();
        collisionEdges = new List<int2>();
        collisionWrapers = new List<CollisionWraper>();

        Dictionary<Vec3Key, int> cellVertDeduper = new Dictionary<Vec3Key, int>();
        HashSet<int2>            cellEdgeDeduper = new HashSet<int2>();

        static void TryCutEdge(float3[] slicedEdges, ref int currentEdgeIdx,
            float3 v0, float3 v1,  float planeValueBack,  float planeValueFront, int axis)
        {
            // Tries to first add the current vertex then the cut edge if it exists
            float v0Val = v0[axis];
            bool firstWasAdded = false;
            if (v0Val >= planeValueBack - 1e-06f && v0Val <= planeValueFront + 1e-06f) 
                { slicedEdges[currentEdgeIdx++] = v0; firstWasAdded = true; }
            // Test for plane intersections
            float3 edgeDir = v1 - v0;
            if (math.abs(edgeDir[axis]) > 1e-06f)
            {
                float t1 = (planeValueBack - v0[axis]) / edgeDir[axis];
                float t2 = (planeValueFront - v0[axis]) / edgeDir[axis];

                // Swap to preserve clock wise orientation
                if (t2 < t1) (t2, t1) = (t1, t2);
                if (firstWasAdded && t2 < 1e-04f) return;

                // Add them only if they are exactly between planes
                if (t1 > 1e-06f && t1 < 1f - 1e-06f && !firstWasAdded)
                    slicedEdges[currentEdgeIdx++] = v0 + t1 * edgeDir;
                if (t2 > 1e-06f && t2 < 1f - 1e-06f)
                    slicedEdges[currentEdgeIdx++] = v0 + t2 * edgeDir;
            }
        }
        
        for (int activeCellID = 0; activeCellID < activeCells.Count; activeCellID++)
        {
            int cellID = activeCells[activeCellID];
            int3 cellCoords = to3D(cellID);

            int initialColVertCount = collisionVertices.Count;
            int initialColTriCount = collisionTriangles.Count;
            int initialColEdgeCount = collisionEdges.Count;

            for (int stride = 0; stride < cellStrides[cellID]; stride++)
            {
                int triangleIdx = cellToTri[new int2(cellID, stride)];
                float3 t1 = meshCollisionVertices[meshCollisionTriangles[triangleIdx]];
                float3 t2 = meshCollisionVertices[meshCollisionTriangles[triangleIdx+1]];
                float3 t3 = meshCollisionVertices[meshCollisionTriangles[triangleIdx+2]];

                // X plane cutting
                float xPlaneValueBack = latticeNegativeCorner.x + latticeDimensions.x * cellCoords.x;
                float xPlaneValueFront = latticeNegativeCorner.x + latticeDimensions.x * (cellCoords.x + 1);
                xSliceEdgesCount = 0;

                // Test each triangle edge and
                // biuld the remaining tri array
                // v0-v1 edge
                TryCutEdge(xSliceEdges, ref xSliceEdgesCount, t1, t2, xPlaneValueBack, xPlaneValueFront, 0);
                // v1-v2 edge
                TryCutEdge(xSliceEdges, ref xSliceEdgesCount, t2, t3, xPlaneValueBack, xPlaneValueFront, 0);
                // v2-v0 edge
                TryCutEdge(xSliceEdges, ref xSliceEdgesCount, t3, t1, xPlaneValueBack, xPlaneValueFront, 0);


                // Y plane cutting
                float yPlaneValueBack = latticeNegativeCorner.y + latticeDimensions.y * cellCoords.y;
                float yPlaneValueFront = latticeNegativeCorner.y + latticeDimensions.y * (cellCoords.y + 1);
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


                // Z plane cutting
                float zPlaneValueBack = latticeNegativeCorner.z + latticeDimensions.z * cellCoords.z;
                float zPlaneValueFront = latticeNegativeCorner.z + latticeDimensions.z * (cellCoords.z + 1);
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


                // Add the vertices
                for (int i = 0; i < zSliceEdgesCount; i++)
                    if (cellVertDeduper.TryAdd(new Vec3Key(zSliceEdges[i], 1e-05f), collisionVertices.Count))
                        collisionVertices.Add(GetCellDynamicVertex(cellID, zSliceEdges[i]));

                // Add the triangles and edges
                int c0 = cellVertDeduper[new Vec3Key(zSliceEdges[0], 1e-05f)];
                for (int i = 1; i < zSliceEdgesCount - 1; i++)
                {
                    int c1 = cellVertDeduper[new Vec3Key(zSliceEdges[i], 1e-05f)];
                    int c2 = cellVertDeduper[new Vec3Key(zSliceEdges[i+1], 1e-05f)];

                    collisionTriangles.Add(new int3(c0, c1, c2));

                    if (cellEdgeDeduper.Add(c0 < c1 ? new int2(c0, c1) : new int2(c1, c0))) collisionEdges.Add(new int2(c0, c1));
                    if (cellEdgeDeduper.Add(c1 < c2 ? new int2(c1, c2) : new int2(c2, c1))) collisionEdges.Add(new int2(c1, c2));
                    if (cellEdgeDeduper.Add(c2 < c0 ? new int2(c2, c0) : new int2(c0, c2))) collisionEdges.Add(new int2(c2, c0));
                }
            }

            // Empty the stride array to use later for vertices
            cellStrides[cellID] = 0;

            // Clear dedupers for the next cell
            cellVertDeduper.Clear();
            cellEdgeDeduper.Clear();

            // Safety exit if there is no data added
            if (initialColVertCount == collisionVertices.Count) continue;

            // Create a new wraper
            collisionWrapers.Add(new CollisionWraper(
                cellID,  // Array index that would later be changed to final list
                new int2(initialColVertCount, collisionVertices.Count),
                new int2(initialColTriCount, collisionTriangles.Count),
                new int2(initialColEdgeCount, collisionEdges.Count)
            ));
        }

    #endregion
    #region Visual Mesh Creator
        
        // Cache and assign new final arrays
        Mesh visualMesh = latticeBody.visualMesh;
        Vector3[] meshVisualVertices = visualMesh.vertices;
        int visualWrapersCount = 0;
        // Translator for each cell
        Dictionary<int2, int> cellToVertex = new Dictionary<int2, int>(meshVisualVertices.Length);
        visualVerticesOffset = new int[meshVisualVertices.Length];
        visualVerticesData = new DynamicVertex[meshVisualVertices.Length];

        for (int i = 0; i < meshVisualVertices.Length; i++)
        {
            // Calculate the position in lattice
            float3 pointInLattice = (float3)meshVisualVertices[i] - latticeNegativeCorner;
            int3 cellCoords = (int3)(pointInLattice / latticeDimensions);
            // Find an add new cell
            int cellID = to1D(cellCoords);
            AddCell(cellID);

            // Append new element using the stride (0...n)
            if (cellStrides[cellID] == 0) visualWrapersCount++;
            cellToVertex.Add(new int2(cellID, cellStrides[cellID]++), i);
        }

        int visualVerticesCount = 0;
        visualVerticesWrapers = new int3[visualWrapersCount];
        visualWrapersCount = 0;

        for (int activeCellID = 0; activeCellID < activeCells.Count; activeCellID++)
        {
            int cellID = activeCells[activeCellID];
            if (cellStrides[cellID] == 0) continue;

            int initialVisualVertCount = visualVerticesCount;

            // Loop though the stride (number of vertices for this cell)
            for (int stride = 0; stride < cellStrides[cellID]; stride++)
            {
                int vertexIdx = cellToVertex[new int2(cellID, stride)];
                visualVerticesOffset[visualVerticesCount] = vertexIdx;

                visualVerticesData[visualVerticesCount] = GetCellDynamicVertex(
                    cellID, meshVisualVertices[vertexIdx]
                );

                visualVerticesCount++;
            }

            // Add a new visual cell wraper
            visualVerticesWrapers[visualWrapersCount++] = new int3(
                cellID,  // Array index that would later be changed to final list
                initialVisualVertCount, visualVerticesCount
            );
        }

    #endregion
    #region Inside Cell Fill

        // Helper data types
        HashSet<int> edgeHits = new HashSet<int>();
        Queue<int> cellQueue = new Queue<int>(100);

        // Marks every lattice cell as either air or filled
        void FloodFill(bool isOutside)
        {
            while (cellQueue.Count != 0)
            {
                // Get current cell and mark it as active or air
                int cellID = cellQueue.Dequeue();
                int3 cellCoords = to3D(cellID);
                if (isOutside) latticeCells[cellID] = -2;
                else AddCell(cellID);

                // Manually add each adjacent cell and verify if it's inside the array
                // +X
                int adjCellID = to1D(cellCoords + new int3(1, 0, 0));
                if (adjCellID >= 0 && adjCellID < maxNumberOfCells && latticeCells[adjCellID] == -1 && cellQueue.Count < 100) cellQueue.Enqueue(adjCellID);
                // -X
                adjCellID = to1D(cellCoords + new int3(-1, 0, 0));
                if (adjCellID >= 0 && adjCellID < maxNumberOfCells && latticeCells[adjCellID] == -1 && cellQueue.Count < 100) cellQueue.Enqueue(adjCellID);
                // +Z
                adjCellID = to1D(cellCoords + new int3(0, 0, 1));
                if (adjCellID >= 0 && adjCellID < maxNumberOfCells && latticeCells[adjCellID] == -1 && cellQueue.Count < 100) cellQueue.Enqueue(adjCellID);
                // -Z
                adjCellID = to1D(cellCoords + new int3(0, 0, -1));
                if (adjCellID >= 0 && adjCellID < maxNumberOfCells && latticeCells[adjCellID] == -1 && cellQueue.Count < 100) cellQueue.Enqueue(adjCellID);
                // +Y
                adjCellID = to1D(cellCoords + new int3(0, 1, 0));
                if (adjCellID >= 0 && adjCellID < maxNumberOfCells && latticeCells[adjCellID] == -1 && cellQueue.Count < 100) cellQueue.Enqueue(adjCellID);
                // -Y
                adjCellID = to1D(cellCoords + new int3(0, -1, 0));
                if (adjCellID >= 0 && adjCellID < maxNumberOfCells && latticeCells[adjCellID] == -1 && cellQueue.Count < 100) cellQueue.Enqueue(adjCellID);
            }
        }

        // Loops though each cell and if it's still initialised (-1)
        // Add it to the queue and check if it's inside or outside the collision mesh
        for (int i = 0; i < maxNumberOfCells; i++)
        {
            if (latticeCells[i] != -1) continue;

            // Calculate cell middle and add it to the queue
            int3 cellCoords = to3D(i);
            float3 cellNegativeCorner = latticeNegativeCorner + cellCoords * latticeDimensions;
            float3 cellMiddle = cellNegativeCorner + 0.5f * latticeDimensions;
            cellQueue.Enqueue(i);

            // Check if it's outside or inside the collision mesh
            if (MeshMath.IsPointInsideMesh(cellMiddle, edgeHits, meshCollisionVertices, meshCollisionTriangles))
                FloodFill(false);
            else FloodFill(true);
        }

    #endregion
    #region Finalisation

    // Create final cells
    finalLatticeCells = new LatticeCell[activeCells.Count];
    latticeCellCoords = new int[activeCells.Count];
    for (int i = 0; i < activeCells.Count; i++)
    {
        LatticeCell newCell = new LatticeCell();

        int3 baseCorner = to3D(activeCells[i]);
        if (latticeCells[activeCells[i]] != i) Debug.Log("wrong");

        // Add each cell corner particle
        // 0: (0,0,0)
        int3 currentCorner = baseCorner;
        newCell.v0 = particles[to1D_Particle(currentCorner)];
        // 1: (1,0,0)
        currentCorner = baseCorner + new int3(1, 0, 0);
        newCell.v1 = particles[to1D_Particle(currentCorner)];
        // 2: (0,1,0)
        currentCorner = baseCorner + new int3(0, 1, 0);
        newCell.v2 = particles[to1D_Particle(currentCorner)];
        // 3: (1,1,0)
        currentCorner = baseCorner + new int3(1, 1, 0);
        newCell.v3 = particles[to1D_Particle(currentCorner)];
        // 4: (0,0,1)
        currentCorner = baseCorner + new int3(0, 0, 1);
        newCell.v4 = particles[to1D_Particle(currentCorner)];
        // 5: (1,0,1)
        currentCorner = baseCorner + new int3(1, 0, 1);
        newCell.v5 = particles[to1D_Particle(currentCorner)];
        // 6: (0,1,1)
        currentCorner = baseCorner + new int3(0, 1, 1);
        newCell.v6 = particles[to1D_Particle(currentCorner)];
        // 7: (1,1,1)
        currentCorner = baseCorner + new int3(1, 1, 1);
        newCell.v7 = particles[to1D_Particle(currentCorner)];

        // Assign the new cell
        finalLatticeCells[i] = newCell;
        latticeCellCoords[i] = activeCells[i];
    }

    // Initialise cell refrences
    cellToColCell = new int[activeCells.Count];
    cellToVisCell = new int[activeCells.Count];
    for (int i = 0; i < activeCells.Count; i++) { cellToColCell[i] = -1; cellToVisCell[i] = -1; }

    // Convert collision and visual cell refrences to point to the new list insted of the whole lattice
    for (int i = 0; i < collisionWrapers.Count; i++)
    {
        CollisionWraper collisionWraper = collisionWrapers[i];
        collisionWraper.cellIdx = latticeCells[collisionWrapers[i].cellIdx];
        collisionWrapers[i] = collisionWraper;

        cellToColCell[collisionWrapers[i].cellIdx] = i;
    }
    for (int i = 0; i < visualVerticesWrapers.Length; i++)
    {
        visualVerticesWrapers[i].x = latticeCells[visualVerticesWrapers[i].x];
        cellToVisCell[visualVerticesWrapers[i].x] = i;
    }

    // GC final helper data
    latticeCells = null;
    activeCells = null;
    particles = null;

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
        int3 baseCorner = to3D(cellID);

        // Determine cell channel
        int chl = baseCorner.x%2 + baseCorner.y%2 * 2 + baseCorner.z%2 * 4;
        chlCount[chl]++;
        // Calculate the position using the previous channel counts
        int newCellPos = 0;
        for (int i = 0; i < chl; i++) newCellPos += chlCount[i];
        // Go through each lattice cell after the newly inseted cell and increase the reference
        for (int i = newCellPos; i < activeCells.Count; i++) latticeCells[activeCells[i]]++;
        activeCells.Insert(newCellPos, cellID);
        latticeCells[cellID] = newCellPos;

        // Add each cell corner particle
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
        // Normalize so the point is between 0 and 1 in all directions
        float3 p = (vertexPos - cellOrigin) / latticeDimensions;

        // Precompute unit cube corners (0/1)
        float3 c0 = new float3(0, 0, 0);
        float3 c1 = new float3(1, 0, 0);
        float3 c2 = new float3(0, 1, 0);
        float3 c3 = new float3(1, 1, 0);
        float3 c4 = new float3(0, 0, 1);
        float3 c5 = new float3(1, 0, 1);
        float3 c6 = new float3(0, 1, 1);
        float3 c7 = new float3(1, 1, 1);

        float3 GetCorner(int idx) =>
            idx == 0 ? c0 :
            idx == 1 ? c1 :
            idx == 2 ? c2 :
            idx == 3 ? c3 :
            idx == 4 ? c4 :
            idx == 5 ? c5 :
            idx == 6 ? c6 : c7;

        const float eps = 1e-5f;

        for (byte tetIndex = 0; tetIndex < s_CellTets.Length; tetIndex++)
        {
            int4 t = s_CellTets[tetIndex];

            float3 a = GetCorner(t.x);
            float3 b = GetCorner(t.y);
            float3 c = GetCorner(t.z);
            float3 d = GetCorner(t.w);

            // p = a*λ0 + b*λ1 + c*λ2 + d*λ3, λ0+λ1+λ2+λ3=1
            // => p - d = (a-d)*λ0 + (b-d)*λ1 + (c-d)*λ2
            float3 e0 = a - d;
            float3 e1 = b - d;
            float3 e2 = c - d;
            float3 r  = p - d;

            float3x3 M = new float3x3(e0, e1, e2);
            float det = math.determinant(M);
            if (math.abs(det) < 1e-8f)
                continue; // degenerate, skip

            float3 lambda012 = math.mul(math.inverse(M), r); // fine for 5 tets per cell
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
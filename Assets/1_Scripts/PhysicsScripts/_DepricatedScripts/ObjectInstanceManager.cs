using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

#region Structs
public struct DistanceConstraintRef
{
    public int2 pidx;
    public float restLenght;

    public DistanceConstraintRef(int2 pidx, float restLenght)
    {
        this.pidx = pidx;
        this.restLenght = restLenght;
    }
}
public struct VolumeConstraintRef
{
    public int4 pidx;
    public float restVolume;

    public VolumeConstraintRef(int4 pidx, float restVolume)
    {
        this.pidx = pidx;
        this.restVolume = restVolume;
    }
}

public struct old_VisualVertex
{
    public int4 tetraIdx;
    public float4 baryCoords;

    public old_VisualVertex(int4 tetraIdx, float4 baryCoords)
    {
        this.tetraIdx = tetraIdx;
        this.baryCoords = baryCoords;
    }
}

public struct old_CollisionVertex
{
    public int4 tetraIdx;
    public float4 baryCoords;

    public old_CollisionVertex(int4 tetraIdx, float4 baryCoords)
    {
        this.tetraIdx = tetraIdx;
        this.baryCoords = baryCoords;
    }
}
#endregion

#region Variables
public class ObjectInstanceManager : MonoBehaviour
{
    // Public object data
    [SerializeField] private Mesh collisionMesh;
    [SerializeField] private int3 latticeDimensions = new int3(2, 2, 2);
    [SerializeField] private int subLatticeMultiplier = 2;
    [SerializeField] private float mass = 1;
    [SerializeField] private float distanceCompliance = 0;
    [SerializeField] private float volumeCompliance = 0;
    // Debug
    [SerializeField] private bool debugCollisionMesh;
    [SerializeField] private bool debugLatticeCells;

    // Mesh data
    private Mesh mesh;
    private int numberOfVertices;
    private Vector3[] meshVertices;

    // Lattice data
    // Main lattice
    private int[] latticeCells;
    private int[] latticeTetraStart;
    private float3 latticeNegativeCorner;
    private float3 latticeLengths;
    private int latticeMaxCount, latticeCount;
    // Sub lattice
    private int[] subLatticeCells;
    private int3 subLatticeDimensions;
    private float3 subLatticeNegativeCorner;
    private float3 subLatticeLenghts;
    private int subLatticeMaxCount, subLatticeCount;

    // Particle data
    private int3 particlesDimensions;
    private List<float3> particles;
    private List<float> particleMasses;
    private int[] particlesDictionary;
    private old_VisualVertex[] visualVerticesBary;

    // Constraint data
    private HashSet<int2> addedEdges; // Redundancy list
    private List<int2>[] distanceConGroups;
    private List<DistanceConstraintRef>[] distanceConstraints;
    private List<VolumeConstraintRef>[] volumeConstraints;

    // Collision mesh data
    private old_CollisionVertex[] collisionVertices;
    private int3[] collisionTriangles;
    private int2[] collisionEdges;
    // Lattice mesh data
    private List<int3> latticeCollisionTriangles;
    private List<int2> latticeCollisionEdges;
#endregion

#region Initialisation
    private void Awake()
    {
        mesh = GetComponent<MeshFilter>().mesh;
        if (mesh == null) return;
        numberOfVertices = mesh.vertexCount;
        mesh.MarkDynamic();
        // mesh.SetVertexBufferParams(numberOfVertices, 
        //     new VertexAttributeDescriptor(VertexAttribute.Position, VertexAttributeFormat.Float32, 3, stream: 0),
        //     new VertexAttributeDescriptor(VertexAttribute.Normal, VertexAttributeFormat.Float32, 3, stream: 1),
        //     new VertexAttributeDescriptor(VertexAttribute.TexCoord0, VertexAttributeFormat.Float32, 2, stream: 2)
        // );

        // Main initialising function
        InitialiseAllVariables();
        // Main functions
        LoopVertices();
        LoopInsideCells();
        // Transforms particle positions from local mesh space to world space
        for (int i = 0; i < particles.Count; i++) particles[i] = transform.TransformPoint(particles[i]);
        // Finalization functions
        CreateCellMesh();
        CleanUp();
        UploadDataToSolver();

        transform.position = Vector3.zero;
        transform.rotation = Quaternion.identity;
        transform.localScale = Vector3.one;
    }

    float EPSILON = 1e-4f;
    private void InitialiseAllVariables()
    {
        latticeDimensions = math.max(latticeDimensions, new int3(1, 1, 1));
        latticeNegativeCorner = (float3)mesh.bounds.min - new float3(EPSILON, EPSILON, EPSILON);
        latticeLengths = (float3)mesh.bounds.size + new float3(2 * EPSILON, 2 * EPSILON, 2 * EPSILON);
        latticeLengths /= (float3)latticeDimensions;

        particlesDimensions = latticeDimensions + new int3(1, 1, 1);

        subLatticeMultiplier = math.max(subLatticeMultiplier, 4);
        subLatticeDimensions = latticeDimensions * subLatticeMultiplier;
        subLatticeNegativeCorner = (float3)mesh.bounds.min - new float3(EPSILON, EPSILON, EPSILON);
        subLatticeLenghts = (float3)mesh.bounds.size + new float3(2 * EPSILON, 2 * EPSILON, 2 * EPSILON);
        subLatticeLenghts /= (float3)subLatticeDimensions;

        latticeMaxCount = (int)(latticeDimensions.x * latticeDimensions.y * latticeDimensions.z);
        subLatticeMaxCount = (int)(subLatticeDimensions.x * subLatticeDimensions.y * subLatticeDimensions.z);
        int numberOfParticles = (int)(particlesDimensions.x * particlesDimensions.y * particlesDimensions.z);

        latticeCount = 0;
        subLatticeCount = 0;

        // -1 for unassigned, -2 for outside and 0 for inside
        latticeCells = new int[latticeMaxCount];
        latticeTetraStart = new int[latticeMaxCount];
        for (int i = 0; i < latticeMaxCount; i++) latticeTetraStart[i] = latticeCells[i] = -1;
        subLatticeCells = new int[subLatticeMaxCount];
        for (int i = 0; i < subLatticeMaxCount; i++) subLatticeCells[i] = -1;

        // -1 for unassigned and >= 0 for assigned with the particle index
        particles = new List<float3>(numberOfParticles);
        particleMasses = new List<float>(numberOfParticles);
        particlesDictionary = new int[numberOfParticles];
        for (int i = 0; i < numberOfParticles; i++) 
        {
            particlesDictionary[i] = -1;
        }

        InitConstraints();
    }
    private void InitConstraints()
    {
        // Distance constraints data
        addedEdges = new HashSet<int2>(); // for duplicate check
        distanceConGroups = new List<int2>[8];
        distanceConstraints = new List<DistanceConstraintRef>[8];

        // Volume constraints data
        volumeConstraints = new List<VolumeConstraintRef>[8];

        for (int i = 0; i < 8; i++)
        {
            // Distance constraints data
            distanceConGroups[i] = new List<int2>();
            distanceConstraints[i] = new List<DistanceConstraintRef>();

            // Volume constraints data
            volumeConstraints[i] = new List<VolumeConstraintRef>();
        }
    }
#endregion

#region Optional cell mesh
    private void CreateCellMesh()
    {
        latticeCollisionTriangles = new List<int3>();
        latticeCollisionEdges = new List<int2>();
        HashSet<int2> addedEdges = new HashSet<int2>();
        for (int cellID = 0; cellID < latticeCells.Length; cellID++)
        {
            if (latticeCells[cellID] != 0) continue;
            int3 negativeCorner = to3D(cellID, latticeDimensions);
            int[] pidx =
            {
                TryAddParticle(negativeCorner + new int3(0, 0, 0)), // 0
                TryAddParticle(negativeCorner + new int3(1, 0, 0)), // 1
                TryAddParticle(negativeCorner + new int3(1, 1, 0)), // 2
                TryAddParticle(negativeCorner + new int3(0, 1, 0)), // 3
                TryAddParticle(negativeCorner + new int3(0, 0, 1)), // 4
                TryAddParticle(negativeCorner + new int3(1, 0, 1)), // 5
                TryAddParticle(negativeCorner + new int3(1, 1, 1)), // 6
                TryAddParticle(negativeCorner + new int3(0, 1, 1)), // 7
            };

            int3 c = negativeCorner;

            // +X face
            if (c.x + 1 < latticeDimensions.x)
            {
                int nID = to1D(c + new int3(1, 0, 0), latticeDimensions);
                if (latticeCells[nID] != 0)
                    AddCellSide(new int4(pidx[2], pidx[6], pidx[5], pidx[1]), addedEdges);
            }
            else AddCellSide(new int4(pidx[2], pidx[6], pidx[5], pidx[1]), addedEdges);

            // -X face
            if (c.x - 1 >= 0)
            {
                int nID = to1D(c + new int3(-1, 0, 0), latticeDimensions);
                if (latticeCells[nID] != 0)
                    AddCellSide(new int4(pidx[4], pidx[7], pidx[3], pidx[0]), addedEdges);
            }
            else AddCellSide(new int4(pidx[4], pidx[7], pidx[3], pidx[0]), addedEdges);

            // +Y face
            if (c.y + 1 < latticeDimensions.y)
            {
                int nID = to1D(c + new int3(0, 1, 0), latticeDimensions);
                if (latticeCells[nID] != 0)
                    AddCellSide(new int4(pidx[7], pidx[6], pidx[2], pidx[3]), addedEdges);
            }
            else AddCellSide(new int4(pidx[7], pidx[6], pidx[2], pidx[3]), addedEdges);

            // -Y face
            if (c.y - 1 >= 0)
            {
                int nID = to1D(c + new int3(0, -1, 0), latticeDimensions);
                if (latticeCells[nID] != 0)
                    AddCellSide(new int4(pidx[1], pidx[5], pidx[4], pidx[0]), addedEdges);
            }
            else AddCellSide(new int4(pidx[1], pidx[5], pidx[4], pidx[0]), addedEdges);

            // +Z face
            if (c.z + 1 < latticeDimensions.z)
            {
                int nID = to1D(c + new int3(0, 0, 1), latticeDimensions);
                if (latticeCells[nID] != 0)
                    AddCellSide(new int4(pidx[5], pidx[6], pidx[7], pidx[4]), addedEdges);
            }
            else AddCellSide(new int4(pidx[5], pidx[6], pidx[7], pidx[4]), addedEdges);

            // -Z face
            if (c.z - 1 >= 0)
            {
                int nID = to1D(c + new int3(0, 0, -1), latticeDimensions);
                if (latticeCells[nID] != 0)
                    AddCellSide(new int4(pidx[3], pidx[2], pidx[1], pidx[0]), addedEdges);
            }
            else AddCellSide(new int4(pidx[3], pidx[2], pidx[1], pidx[0]), addedEdges);
        }
    }

    private void AddCellSide(int4 sideIdx, HashSet<int2> addedEdges)
    {
        latticeCollisionTriangles.Add(sideIdx.xyz);
        latticeCollisionTriangles.Add(sideIdx.zwx);

        latticeCollisionEdges.Add(sideIdx.xz);

        if (!addedEdges.Contains(sideIdx.yx))
        {
            addedEdges.Add(sideIdx.xy);
            latticeCollisionEdges.Add(sideIdx.xy);
        }

        if (!addedEdges.Contains(sideIdx.zy))
        {
            addedEdges.Add(sideIdx.yz);
            latticeCollisionEdges.Add(sideIdx.yz);
        }

        if (!addedEdges.Contains(sideIdx.wz))
        {
            addedEdges.Add(sideIdx.zw);
            latticeCollisionEdges.Add(sideIdx.zw);
        }

        if (!addedEdges.Contains(sideIdx.xw))
        {
            addedEdges.Add(sideIdx.wx);
            latticeCollisionEdges.Add(sideIdx.wx);
        }
    }
#endregion

#region Solver integration

    private void UploadDataToSolver()
    {
        JobPhysicsSolver solver = FindFirstObjectByType(typeof(JobPhysicsSolver)) as JobPhysicsSolver;

        solver.AddSoftBody(
            this, mass, particleMasses, 
            distanceCompliance, volumeCompliance,
            particles, distanceConGroups, distanceConstraints,
            volumeConstraints, visualVerticesBary,
            collisionVertices,
            collisionTriangles,
            collisionEdges,
            latticeCollisionTriangles,
            latticeCollisionEdges
        );
    }

    public void UpdateMesh(NativeList<float3> vertices, int verticesStart, 
        NativeList<old_Particle> particles, int particleStart, int particleCount)
    {
        Vector3 centerPos = Vector3.zero;
        for (int i = particleStart; i < particleStart + particleCount; i++)
        {
            centerPos += (Vector3)particles[i].position;
            this.particles[i - particleStart] = particles[i].position;
        }
        centerPos /= particleCount;
        transform.position = centerPos;

        for (int i = verticesStart; i < verticesStart + numberOfVertices; i++)
            meshVertices[i - verticesStart] = (Vector3)vertices[i] - centerPos;
        mesh.SetVertices(meshVertices);
        mesh.RecalculateBounds();
        mesh.RecalculateNormals();
    }

#endregion

#region Shape addition
    /// <summary>
    /// Loops through all vertices and adds them to the coresponding lattice cell.
    /// Also handles the trilinear interpolation for each particle.
    /// </summary>
    private void LoopVertices()
    {
        meshVertices = mesh.vertices;
        visualVerticesBary = new old_VisualVertex[numberOfVertices];

        for (int i = 0; i < numberOfVertices; i++)
        {
            // Calculates the lattice cell of the current mesh vertex
            float3 positionInLattice = (float3)meshVertices[i] - latticeNegativeCorner;
            int3 negativeCorner = (int3)(positionInLattice / latticeLengths);
            int cellID = to1D(negativeCorner, latticeDimensions);

            // Try adding the containing cell
            AddLatticeCell(negativeCorner, cellID);

            // Stores data for barycentric interpolation
            int cellParity = negativeCorner.x%2 + negativeCorner.y%2 * 2 + negativeCorner.z%2 * 4;
            int tetraStart = latticeTetraStart[cellID];
            for (int tet = tetraStart; tet < tetraStart + 5; tet++)
            {
                int4 currentTet = volumeConstraints[cellParity][tet].pidx;
                if (TryGetBarycentricInclusive((float3)meshVertices[i],
                    particles[currentTet.x], particles[currentTet.y],
                    particles[currentTet.z], particles[currentTet.w],
                    out float4 bary))
                {
                    visualVerticesBary[i] = new old_VisualVertex(
                        currentTet,
                        bary
                    );
                }
            }
        }
    }
#endregion

#region Cell addition
    /// <summary>
    /// Adds new cells, each newly added cell also has all of it's
    /// constraints added and indexed.
    /// </summary>
    private void AddLatticeCell(int3 negativeCorner, int cellID)
    {
        if (latticeCells[cellID] == 0) return;
        
        // Adds particles in a specific order
        int[] pidx =
        {
            TryAddParticle(negativeCorner + new int3(0, 0, 0)), // 0
            TryAddParticle(negativeCorner + new int3(1, 0, 0)), // 1
            TryAddParticle(negativeCorner + new int3(1, 1, 0)), // 2
            TryAddParticle(negativeCorner + new int3(0, 1, 0)), // 3
            TryAddParticle(negativeCorner + new int3(0, 0, 1)), // 4
            TryAddParticle(negativeCorner + new int3(1, 0, 1)), // 5
            TryAddParticle(negativeCorner + new int3(1, 1, 1)), // 6
            TryAddParticle(negativeCorner + new int3(0, 1, 1)), // 7
        };

        int cellParity = negativeCorner.x%2 + negativeCorner.y%2 * 2 + negativeCorner.z%2 * 4;
        
        AddDistanceConstraints(pidx, cellParity);
        // Current cell respective tetra start index
        latticeTetraStart[cellID] = volumeConstraints[cellParity].Count;
        AddVolumeConstraints(pidx, cellParity);

        latticeCells[cellID] = 0;
        latticeCount++;
    }

    /// <summary>
    /// Returns the index of the present or newly added vertex.
    /// </summary>
    private int TryAddParticle(int3 position)
    {
        int particleIndex = to1D(position, particlesDimensions);
        
        if (particlesDictionary[particleIndex] >= 0) return particlesDictionary[particleIndex];

        particlesDictionary[particleIndex] = particles.Count;
        particles.Add(latticeNegativeCorner + position * latticeLengths);
        particleMasses.Add(0);
        return particles.Count - 1;
    }

    private void AddDistanceConstraints(int[] pidx, int cellParity)
    {
        int initDisCount, disCount;
        initDisCount = disCount = distanceConstraints[cellParity].Count;
        for (int x = 0; x < 8; x++)
        {
            for (int y = x+1; y < 8; y++)
            {
                int2 edge = new int2(pidx[x], pidx[y]);

                if (addedEdges.Contains(edge) || addedEdges.Contains(edge.yx)) continue;

                addedEdges.Add(edge);
                disCount++;
                distanceConstraints[cellParity].Add(new DistanceConstraintRef(edge, 
                    math.distance(transform.TransformPoint(particles[edge.x]), 
                                transform.TransformPoint(particles[edge.y]))));
            }
        }
        distanceConGroups[cellParity].Add(new int2(initDisCount, disCount));
    }

    private void AddVolumeConstraints(int[] pidx, int cellParity)
    {
        int[,] tetrahedra = new int[,]
        {
            {0,2,7,5}, {0,5,1,2}, {0,5,7,4}, {0,2,3,7}, {6,5,7,2}
        };

        for (int i = 0; i < 5; i++)
        {
            int p0 = pidx[tetrahedra[i, 0]];
            int p1 = pidx[tetrahedra[i, 1]];
            int p2 = pidx[tetrahedra[i, 2]];
            int p3 = pidx[tetrahedra[i, 3]];

            float volume = math.dot(
                transform.TransformPoint(particles[p1]) - transform.TransformPoint(particles[p0]), 
                math.cross(transform.TransformPoint(particles[p2]) - transform.TransformPoint(particles[p0]), 
                    transform.TransformPoint(particles[p3]) - transform.TransformPoint(particles[p0]))
            ) / 6f;
            // Debug.Log($"Tetra {i} volume = {volume}");

            volumeConstraints[cellParity].Add(new VolumeConstraintRef(new int4(p0, p1, p2, p3), volume));
        }
    }
#endregion

#region Collision shape
    /// <summary>
    /// Handles the collision mesh and
    /// the cells that can be on or inside the mesh but not
    /// added before.
    /// </summary>
    private void LoopInsideCells()
    {
        if (collisionMesh == null) return;
        Vector3[] colMeshVerts = collisionMesh.vertices;
        int[] colMeshTris = collisionMesh.triangles;
        RemoveDuplicateVertices(ref colMeshVerts, ref colMeshTris);

        // Create temporary lists for collision data
        List<old_CollisionVertex> tempCollisionVertices = new List<old_CollisionVertex>();
        List<int3> tempCollisionTriangles = new List<int3>();
        List<int2> tempCollisionEdges = new List<int2>();

        // Loop though each triangle and add all the cells it intersects
        for (int i = 0; i < colMeshTris.Length; i+=3)
        {
            float3 v0 = colMeshVerts[colMeshTris[i]];
            float3 v1 = colMeshVerts[colMeshTris[i+1]];
            float3 v2 = colMeshVerts[colMeshTris[i+2]];

            // Calculate the min and max corners of the bounding box
            float3 minBB = new float3(
                v0.x < (v1.x < v2.x ? v1.x : v2.x) ? v0.x : (v1.x < v2.x ? v1.x : v2.x),
                v0.y < (v1.y < v2.y ? v1.y : v2.y) ? v0.y : (v1.y < v2.y ? v1.y : v2.y),
                v0.z < (v1.z < v2.z ? v1.z : v2.z) ? v0.z : (v1.z < v2.z ? v1.z : v2.z)
            );
            float3 maxBB = new float3(
                v0.x > (v1.x > v2.x ? v1.x : v2.x) ? v0.x : (v1.x > v2.x ? v1.x : v2.x),
                v0.y > (v1.y > v2.y ? v1.y : v2.y) ? v0.y : (v1.y > v2.y ? v1.y : v2.y),
                v0.z > (v1.z > v2.z ? v1.z : v2.z) ? v0.z : (v1.z > v2.z ? v1.z : v2.z)
            );

            // Main lattice tests
            TestMainLatticeCells(minBB, maxBB, v0, v1, v2);
            // Sub lattice tests
            TestSubLatticeCells(minBB, maxBB, v0, v1, v2);


            // Calculates the min negative corner idxs
            float3 minPosInLattice = minBB - latticeNegativeCorner;
            int3 minNegativeCorner = (int3)(minPosInLattice / latticeLengths);

            // Calculates the max negative corner idxs
            float3 maxPosInLattice = maxBB - latticeNegativeCorner;
            int3 maxNegativeCorner = (int3)(maxPosInLattice / latticeLengths);

            // Loop though each cell touched by the triangle bounding box
            // And cut the triangle for each on of them
            float3[] xSliceEdges = new float3[5];
            int xSliceEdgesCount;
            float3[] ySliceEdges = new float3[7];
            int ySliceEdgesCount;
            float3[] zSliceEdges = new float3[9];
            int zSliceEdgesCount;
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
                float xPlaneValueBack = latticeNegativeCorner.x + latticeLengths.x * x;
                float xPlaneValueFront = latticeNegativeCorner.x + latticeLengths.x * (x + 1);
                xSliceEdgesCount = 0;

                // Test each triangle edge and
                // biuld the remaining tri array
                // v0-v1 edge
                TryCutEdge(xSliceEdges, ref xSliceEdgesCount, v0, v1, xPlaneValueBack, xPlaneValueFront, 0);
                // v1-v2 edge
                TryCutEdge(xSliceEdges, ref xSliceEdgesCount, v1, v2, xPlaneValueBack, xPlaneValueFront, 0);
                // v2-v0 edge
                TryCutEdge(xSliceEdges, ref xSliceEdgesCount, v2, v0, xPlaneValueBack, xPlaneValueFront, 0);

                for (int y = minNegativeCorner.y; y <= maxNegativeCorner.y; y++)
                {
                    float yPlaneValueBack = latticeNegativeCorner.y + latticeLengths.y * y;
                    float yPlaneValueFront = latticeNegativeCorner.y + latticeLengths.y * (y + 1);
                    ySliceEdgesCount = 0;

                    // Loop though each edge of the x slice
                    // cutting and adding it to the y slice list
                    for (int j = 0; j < xSliceEdgesCount; j++)
                    {
                        float3 e0 = xSliceEdges[j];
                        float3 e1 = j+1 == xSliceEdgesCount ? xSliceEdges[0] : xSliceEdges[j+1];
                        
                        // Cut if possible
                        TryCutEdge(ySliceEdges, ref ySliceEdgesCount, e0, e1, yPlaneValueBack, yPlaneValueFront, 1);
                    }

                    for (int z = minNegativeCorner.z; z <= maxNegativeCorner.z; z++)
                    {
                        float zPlaneValueBack = latticeNegativeCorner.z + latticeLengths.z * z;
                        float zPlaneValueFront = latticeNegativeCorner.z + latticeLengths.z * (z + 1);
                        zSliceEdgesCount = 0;

                        // Loop though each edge of the y slice
                        // cutting and adding it to the z slice list
                        for (int j = 0; j < ySliceEdgesCount; j++)
                        {
                            float3 e0 = ySliceEdges[j];
                            float3 e1 = j+1 == ySliceEdgesCount ? ySliceEdges[0] : ySliceEdges[j+1];

                            // Cut if possible
                            TryCutEdge(zSliceEdges, ref zSliceEdgesCount, e0, e1, zPlaneValueBack, zPlaneValueFront, 2);
                        }
                        if (zSliceEdgesCount <= 2) continue;

                        // Double checks if the cell intersects the trianglex
                        int3 cellCoords = new int3(x, y, z);
                        int cellID = to1D(cellCoords, latticeDimensions);
                        AddLatticeCell(cellCoords, cellID);
                        
                        // Create new triangles
                        int startingTempColCount = tempCollisionVertices.Count;
                        for (int j = 0; j < zSliceEdgesCount; j++)
                        {
                            // Create new collision vertex with the barycentric interpolation based on main mesh
                            int cellParity = cellCoords.x%2 + cellCoords.y%2 * 2 + cellCoords.z%2 * 4;
                            int tetraStart = latticeTetraStart[cellID];
                            for (int tet = tetraStart; tet < tetraStart + 5; tet++)
                            {
                                // Test in what tetra does the new collision vertex is contained
                                int4 currentTet = volumeConstraints[cellParity][tet].pidx;
                                if (TryGetBarycentricInclusive(zSliceEdges[j],
                                    particles[currentTet.x], particles[currentTet.y],
                                    particles[currentTet.z], particles[currentTet.w],
                                    out float4 bary))
                                {
                                    tempCollisionVertices.Add(new old_CollisionVertex(
                                        currentTet, bary
                                    ));
                                }
                            }

                            // Test if a triangle could be created
                            if (j == 0 || j+1 >= zSliceEdgesCount) continue;

                            // Add a new triangle formed by the first vertex
                            // the current one and the next vertex
                            tempCollisionTriangles.Add(new int3(
                                startingTempColCount, 
                                tempCollisionVertices.Count - 1, 
                                tempCollisionVertices.Count
                            ));

                            // Add the outside and inside edge (common between 2 tries if they exist)
                            tempCollisionEdges.Add(new int2(startingTempColCount, tempCollisionVertices.Count - 1));
                            tempCollisionEdges.Add(new int2(tempCollisionVertices.Count - 1, tempCollisionVertices.Count));
                        }
                        // Then add the last outisde edge
                        tempCollisionEdges.Add(new int2(tempCollisionVertices.Count - 1, startingTempColCount));
                    }
                }
            }
        }

        // Assign newly created lists
        collisionVertices = tempCollisionVertices.ToArray();
        collisionTriangles = tempCollisionTriangles.ToArray();
        collisionEdges = tempCollisionEdges.ToArray();


        // Add cells that are inside using flood fill for speed
        // Hash set for casses where it hits an edge of the triangle
        HashSet<int> edgeHits = new HashSet<int>();

        // Queue for both the main and sub lattice flood fills
        Queue<int> queue = new Queue<int>(subLatticeMaxCount * (1/8));

        // Main lattice flood fill
        FillMainLattice(colMeshVerts, colMeshTris, edgeHits, queue);
        // Sub lattice flood fill
        FillSubLattice(colMeshVerts, colMeshTris, edgeHits, queue);

        // Add weight to particles using the sub lattice
        float subCellMass = mass / subLatticeCount;
        for (int i = 0; i < subLatticeMaxCount; i++)
        {
            if (subLatticeCells[i] != 0) continue;

            int3 mainLatticeCorner = to3D(i, subLatticeDimensions) / subLatticeMultiplier;

            // Add mass to each xpbd particle
            particleMasses[TryAddParticle(mainLatticeCorner + new int3(0, 0, 0))] += subCellMass / 8f;
            particleMasses[TryAddParticle(mainLatticeCorner + new int3(1, 0, 0))] += subCellMass / 8f;
            particleMasses[TryAddParticle(mainLatticeCorner + new int3(1, 1, 0))] += subCellMass / 8f;
            particleMasses[TryAddParticle(mainLatticeCorner + new int3(0, 1, 0))] += subCellMass / 8f;
            particleMasses[TryAddParticle(mainLatticeCorner + new int3(0, 0, 1))] += subCellMass / 8f;
            particleMasses[TryAddParticle(mainLatticeCorner + new int3(1, 0, 1))] += subCellMass / 8f;
            particleMasses[TryAddParticle(mainLatticeCorner + new int3(1, 1, 1))] += subCellMass / 8f;
            particleMasses[TryAddParticle(mainLatticeCorner + new int3(0, 1, 1))] += subCellMass / 8f;
        }
    }

#region Triangle Fill
    /// <summary>
    /// Tests which cells the triangle actually intesects
    /// with the main lattice and adds them if they do intersect.
    /// </summary>
    private void TestMainLatticeCells(float3 minBB, float3 maxBB, float3 a, float3 b, float3 c)
    {
        // Calculates the min negative corner
        float3 minPosInLattice = minBB - latticeNegativeCorner;
        int3 minNegativeCorner = (int3)(minPosInLattice / latticeLengths);

        // Calculates the max negative corner
        float3 maxPosInLattice = maxBB - latticeNegativeCorner;
        int3 maxNegativeCorner = (int3)(maxPosInLattice / latticeLengths);

        // Loop though each cell touched by the triangle bounding box
        for (int x = minNegativeCorner.x; x <= maxNegativeCorner.x; x++)
        {
            for (int y = minNegativeCorner.y; y <= maxNegativeCorner.y; y++)
            {
                for (int z = minNegativeCorner.z; z <= maxNegativeCorner.z; z++)
                {
                    // Double checks if the cell intersects the triangle
                    int3 cellCoords = new int3(x, y, z);
                    int cellID = to1D(cellCoords, latticeDimensions);
                    if (latticeCells[cellID] == 0) continue;
                    if (!TriangleIntersectsCell(latticeNegativeCorner + cellCoords * latticeLengths, latticeLengths, a, b, c)) continue;

                    AddLatticeCell(cellCoords, cellID);
                }
            }
        }
    }

    /// <summary>
    /// Tests which cells the triangle actually intesects
    /// with the sub lattice and adds them if they do intersect.
    /// </summary>
    private void TestSubLatticeCells(float3 minBB, float3 maxBB, float3 a, float3 b, float3 c)
    {
        // Calculates the min negative corner
        float3 minPosInLattice = minBB - subLatticeNegativeCorner;
        int3 minNegativeCorner = (int3)(minPosInLattice / subLatticeLenghts);

        // Calculates the max negative corner
        float3 maxPosInLattice = maxBB - subLatticeNegativeCorner;
        int3 maxNegativeCorner = (int3)(maxPosInLattice / subLatticeLenghts);

        // Loop though each cell touched by the triangle bounding box
        for (int x = minNegativeCorner.x; x <= maxNegativeCorner.x; x++)
        {
            for (int y = minNegativeCorner.y; y <= maxNegativeCorner.y; y++)
            {
                for (int z = minNegativeCorner.z; z <= maxNegativeCorner.z; z++)
                {
                    // Double checks if the cell intersects the triangle
                    int3 cellCoords = new int3(x, y, z);
                    int cellID = to1D(cellCoords, subLatticeDimensions);
                    if (subLatticeCells[cellID] == 0) continue;
                    if (!TriangleIntersectsCell(subLatticeNegativeCorner + subLatticeLenghts * cellCoords, subLatticeLenghts, a, b, c)) continue;

                    subLatticeCells[cellID] = 0;
                    subLatticeCount++;
                }
            }
        }
    }

    /// <summary>
    /// Gets the closest point inside the triangle from the middle of the cell
    /// if the point is inside the cell return ture.
    /// </summary>
    private bool TriangleIntersectsCell(float3 cellNegativeCorner, float3 cellLenghts, 
        float3 a, float3 b, float3 c)
    {
        float3 cellMiddle = cellNegativeCorner + cellLenghts * 0.5f;

        float3 closestPoint = ClosestPointInTriangle(cellMiddle, a, b, c);

        // Check if the closest point is inside the lattice
        if (closestPoint.x > cellNegativeCorner.x + cellLenghts.x + 1e-06f || closestPoint.x < cellNegativeCorner.x - 1e-06f)
            return false;
        if (closestPoint.y > cellNegativeCorner.y + cellLenghts.y + 1e-06f || closestPoint.y < cellNegativeCorner.y - 1e-06f)
            return false;
        if (closestPoint.z > cellNegativeCorner.z + cellLenghts.z + 1e-06f || closestPoint.z < cellNegativeCorner.z - 1e-06f)
            return false;
        return true;
    }
#endregion

#region Inside Fill
    /// <summary>
    /// Fills the main lattice.
    /// </summary>
    private void FillMainLattice(Vector3[] colMeshVerts, int[] colMeshTris,
            HashSet<int> edgeHits, Queue<int> queue)
    {
        for (int i = 0; i < latticeMaxCount; i++)
        {
            if (latticeCells[i] != -1) continue;

            int3 cellCoords = to3D(i, latticeDimensions);
            float3 cellNegativeCorner = latticeNegativeCorner + cellCoords * latticeLengths;
            float3 cellMiddle = cellNegativeCorner + 0.5f * latticeLengths;

            // Check if the current cell is inside the collision mesh
            if (IsPointInsideMesh(cellMiddle, edgeHits, colMeshVerts, colMeshTris))
            {
                // If it's inside add it to the queue
                AddLatticeCell(cellCoords, i);
                queue.Enqueue(i);

                FloodFillCells(queue, true, true);
            }
            else
            {
                // If it's not then set it to outside and add to the queue
                latticeCells[i] = -2;
                queue.Enqueue(i);

                FloodFillCells(queue, false, true);
            }
        }
    }

    /// <summary>
    /// Fills the sub lattice.
    /// </summary>
    private void FillSubLattice(Vector3[] colMeshVerts, int[] colMeshTris,
            HashSet<int> edgeHits, Queue<int> queue)
    {
        for (int i = 0; i < subLatticeMaxCount; i++)
        {
            if (subLatticeCells[i] != -1) continue;

            int3 cellCoords = to3D(i, subLatticeDimensions);
            float3 cellNegativeCorner = subLatticeNegativeCorner + cellCoords * subLatticeLenghts;
            float3 cellMiddle = cellNegativeCorner + 0.5f * subLatticeLenghts;

            // Check if the current cell is inside the collision mesh
            if (IsPointInsideMesh(cellMiddle, edgeHits, colMeshVerts, colMeshTris))
            {
                // If it's inside add it to the queue
                subLatticeCells[i] = 0;
                subLatticeCount++;
                queue.Enqueue(i);

                FloodFillCells(queue, true, false);
            }
            else
            {
                // If it's not then set it to outside and add to the queue
                subLatticeCells[i] = -2;
                queue.Enqueue(i);

                FloodFillCells(queue, false, false);
            }
        }
    }

    /// <summary>
    /// Flood fills the either the main or the sub lattice.
    /// </summary>
    private void FloodFillCells(Queue<int> queue, bool forInside, bool mainLattice)
    {
        while (queue.Count > 0)
        {
            int cellID = queue.Dequeue();
            int3 cellCoords = to3D(cellID, mainLattice ? latticeDimensions : subLatticeDimensions);

            TryFloodCell(cellCoords, 0, true, queue, forInside, mainLattice);
            TryFloodCell(cellCoords, 0, false, queue, forInside, mainLattice);
            TryFloodCell(cellCoords, 1, true, queue, forInside, mainLattice);
            TryFloodCell(cellCoords, 1, false, queue, forInside, mainLattice);
            TryFloodCell(cellCoords, 2, true, queue, forInside, mainLattice);
            TryFloodCell(cellCoords, 2, false, queue, forInside, mainLattice);
        }
    }

    /// <summary>
    /// Checks if if any adjacent cell is valid or not on the given axis and direction.
    /// </summary>
    private void TryFloodCell(int3 cellCoords, int axis, bool sign, Queue<int> queue, 
        bool forInside, bool mainLattice)
    {
        if (cellCoords[axis] + (sign ? 1 : -1) < (mainLattice ? latticeDimensions[axis] : subLatticeDimensions[axis])
            && cellCoords[axis] + (sign ? 1 : -1) >= 0)
        {
            int3 addition = int3.zero;
            addition[axis] = sign ? 1 : -1;
            int3 nextCellCords = cellCoords + addition;
            int nextCellID = to1D(nextCellCords, mainLattice ? latticeDimensions : subLatticeDimensions);
            if (mainLattice ? latticeCells[nextCellID] == -1 : subLatticeCells[nextCellID] == -1)
            {
                if (forInside)
                {
                    if (mainLattice) AddLatticeCell(nextCellCords, nextCellID);
                    else { subLatticeCells[nextCellID] = 0; subLatticeCount++; }
                    queue.Enqueue(nextCellID);
                }
                else
                {
                    if (mainLattice) latticeCells[nextCellID] = -2;
                    else subLatticeCells[nextCellID] = -2;
                    queue.Enqueue(nextCellID);
                }
            }
        }
    }
#endregion
#endregion

#region Clean up
    /// <summary>
    /// Frees up memory.
    /// </summary>
    private void CleanUp()
    {
        addedEdges = null;
        particlesDictionary = null;
        latticeCells = null;
        latticeTetraStart = null;
        subLatticeCells = null;
    }
#endregion

#region Helper functions
    public int to1D(int3 pos, int3 dimensions)
    {
        return (pos.z * dimensions.x * dimensions.y) + (pos.y * dimensions.x) + pos.x;
    }

    public int3 to3D(int idx, int3 dimensions)
    {
        int3 result = int3.zero;
        result.z = idx / (dimensions.x * dimensions.y);
        idx -= result.z * dimensions.x * dimensions.y;
        result.y = idx / dimensions.x;
        result.x = idx % dimensions.x;

        return result;
    }

    /// <summary>
    /// Finds the closest point on or inside a triangle to a given point in 3D space.
    /// </summary>
    /// <param name="point">The point to find the closest location for</param>
    /// <param name="a">First vertex of the triangle</param>
    /// <param name="b">Second vertex of the triangle</param>
    /// <param name="c">Third vertex of the triangle</param>
    /// <returns>The closest point inside or on the triangle</returns>
    public static float3 ClosestPointInTriangle(float3 point, float3 a, float3 b, float3 c)
    {
        // Project point onto the triangle's plane
        float3 ab = b - a;
        float3 ac = c - a;
        float3 ap = point - a;

        float ab2   = math.dot(ab, ab);
        float ac2   = math.dot(ac, ac);
        float ab_ac = math.dot(ab, ac);
        float ap_ab = math.dot(ap, ab);
        float ap_ac = math.dot(ap, ac);

        float denom = ab2 * ac2 - ab_ac * ab_ac;
        float u = ac2 * ap_ab - ab_ac * ap_ac;
        float v = ab2 * ap_ac - ab_ac * ap_ab;

        // Check if point is inside the triangle
        if (u >= 0 && v >= 0 && u + v <= denom)
        {
            // Point projects inside the triangle
            u /= denom;
            v /= denom;
            return a + u * ab + v * ac;
        }

        // Point is outside, find closest point on the edges
        float minDistSq = float.MaxValue;
        float3 closestPoint = a;

        // Check edge AB
        float t = math.dot(ap, ab) / ab2;
        t = Mathf.Clamp01(t);
        float3 edgePoint = a + t * ab;
        float distSq = Vector3.SqrMagnitude(point - edgePoint);
        if (distSq < minDistSq)
        {
            minDistSq = distSq;
            closestPoint = edgePoint;
        }

        // Check edge BC
        float3 bc = c - b;
        float3 bp = point - b;
        t = math.dot(bp, bc) / math.dot(bc, bc);
        t = Mathf.Clamp01(t);
        edgePoint = b + t * bc;
        distSq = Vector3.SqrMagnitude(point - edgePoint);
        if (distSq < minDistSq)
        {
            minDistSq = distSq;
            closestPoint = edgePoint;
        }

        // Check edge CA
        float3 ca = a - c;
        float3 cp = point - c;
        t = math.dot(cp, ca) / math.dot(ca, ca);
        t = Mathf.Clamp01(t);
        edgePoint = c + t * ca;
        distSq = Vector3.SqrMagnitude(point - edgePoint);
        if (distSq < minDistSq)
        {
            minDistSq = distSq;
            closestPoint = edgePoint;
        }

        // Check vertices
        if (Vector3.SqrMagnitude(point - a) < minDistSq)
            closestPoint = a;
        if (Vector3.SqrMagnitude(point - b) < minDistSq)
            closestPoint = b;
        if (Vector3.SqrMagnitude(point - c) < minDistSq)
            closestPoint = c;

        return closestPoint;
    }

    /// <summary>
    /// Checks if a given point is inside the given mesh.
    /// Uses ray intersection test with a way to detect edge hits.
    /// </summary>
    private bool IsPointInsideMesh(float3 point, HashSet<int> edgeHits, Vector3[] colMeshVerts, int[] colMeshTris)
    {
        edgeHits.Clear();
        int intersections = 0;
        float3 randDir = math.normalize(UnityEngine.Random.insideUnitSphere);

        // Loops through the triangles of the mesh collider and checks for intersections
        for (int i = 0; i < colMeshTris.Length; i+=3)
        {
            float3 v0 = colMeshVerts[colMeshTris[i]];
            float3 v1 = colMeshVerts[colMeshTris[i+1]];
            float3 v2 = colMeshVerts[colMeshTris[i+2]];

            // Main check for casses where the cell is completly inside the collider mesh
            float result = RayIntersectsTriangleDetailed(point, randDir, v0, v1, v2);
            if (result == -2f) intersections++;
            else if (result > 0)
            {
                // In case of an edge hit, indexes it with some kind of accuracy
                // and checks if it was already added and if so counts that as an intersection
                int key = Mathf.RoundToInt(result * 100000f);
                if (!edgeHits.Add(key))
                {
                    edgeHits.Remove(key);
                    intersections++;
                }
            }
        }

        // Test fail condition, either outside or interected with the mesh
        if (intersections % 2 == 0) return false;
        return true;
    }

    /// <summary>
    /// Performs ray-triangle intersection using the Möller–Trumbore algorithm.
    /// Returns:
    /// - -1f if there is no intersection,
    /// - -2f if there is a valid intersection (not on an edge),
    /// - positive t value if the ray intersects the triangle near an edge (used for edge de-duplication).
    /// </summary>
    private float RayIntersectsTriangleDetailed(float3 rayOrigin, float3 rayDir, float3 v0, float3 v1, float3 v2)
    {
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        float3 h = math.cross(rayDir, edge2);
        float a = math.dot(edge1, h);
        if (math.abs(a) < 1e-5f) return -1f;

        float f = 1f / a;
        float3 s = rayOrigin - v0;
        float u = f * math.dot(s, h);
        if (u < 0f || u > 1f) return -1f;

        float3 q = math.cross(s, edge1);
        float v = f * math.dot(rayDir, q);
        if (v < 0f || u + v > 1f) return -1f;

        float t = f * math.dot(edge2, q);
        if (t < 1e-5f) return -1f;

        if (u < 1e-5f || v < 1e-5f || math.abs(u + v - 1f) < 1e-5f)
            return t;

        return -2f;
    }

    /// <summary>
    /// Tests if any edge of the cell intersects the given triangle.
    /// </summary>
    private bool CellEdgesIntersectsTri(float3 negativeCornerPos, float3 v0, float3 v1, float3 v2)
    {
        float3 t0 = negativeCornerPos;
        float3 t1 = negativeCornerPos + latticeLengths * new float3(1, 0, 0);
        float3 t2 = negativeCornerPos + latticeLengths * new float3(1, 1, 0);
        float3 t3 = negativeCornerPos + latticeLengths * new float3(0, 1, 0);
        float3 t4 = negativeCornerPos + latticeLengths * new float3(0, 0, 1);
        float3 t5 = negativeCornerPos + latticeLengths * new float3(1, 0, 1);
        float3 t6 = negativeCornerPos + latticeLengths * new float3(1, 1, 1);
        float3 t7 = negativeCornerPos + latticeLengths * new float3(0, 1, 1);

        if (LineSegmentIntersectsTriangle(t0, t1, v0, v1, v2)) return true;
        if (LineSegmentIntersectsTriangle(t0, t3, v0, v1, v2)) return true;
        if (LineSegmentIntersectsTriangle(t0, t4, v0, v1, v2)) return true;
        if (LineSegmentIntersectsTriangle(t2, t1, v0, v1, v2)) return true;
        if (LineSegmentIntersectsTriangle(t2, t3, v0, v1, v2)) return true;
        if (LineSegmentIntersectsTriangle(t2, t6, v0, v1, v2)) return true;
        if (LineSegmentIntersectsTriangle(t5, t1, v0, v1, v2)) return true;
        if (LineSegmentIntersectsTriangle(t5, t4, v0, v1, v2)) return true;
        if (LineSegmentIntersectsTriangle(t5, t6, v0, v1, v2)) return true;
        if (LineSegmentIntersectsTriangle(t7, t6, v0, v1, v2)) return true;
        if (LineSegmentIntersectsTriangle(t7, t4, v0, v1, v2)) return true;
        if (LineSegmentIntersectsTriangle(t7, t3, v0, v1, v2)) return true;
        return false;
    }

    /// <summary>
    /// Similar to RayIntersectsTriangleDetailed but only returns a bool if the 
    /// given lattice cell intersects the given triangle
    /// </summary>
    private bool LineSegmentIntersectsTriangle(float3 p0, float3 p1, float3 v0, float3 v1, float3 v2)
    {
        float3 dir = p1 - p0;
        float segLenSq = math.lengthsq(dir);
        if (segLenSq < 1e-5f * 1e-5f) return false; // degenerate (very short) segment

        // Möller–Trumbore with dir = p1 - p0
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        float3 h = math.cross(dir, edge2);
        float a = math.dot(edge1, h);
        if (math.abs(a) < 1e-5f) return false; // parallel or degenerate

        float f = 1f / a;
        float3 s = p0 - v0;
        float u = f * math.dot(s, h);
        if (u < -1e-5f || u > 1f + 1e-5f) return false;

        float3 q = math.cross(s, edge1);
        float v = f * math.dot(dir, q);
        if (v < -1e-5f || u + v > 1f + 1e-5f) return false;

        float t = f * math.dot(edge2, q);
        // Here t is in parametric units of dir: intersection point = p0 + t * dir.
        // For the segment we need 0 <= t <= 1 (with epsilon tolerance)
        if (t < -1e-5f || t > 1f + 1e-5f) return false;

        return true;
    }

    public static void RemoveDuplicateVertices(
        ref Vector3[] vertices,
        ref int[] triangles,
        float tolerance = 0f
    )
    {
        if (vertices == null) throw new ArgumentNullException(nameof(vertices));
        if (triangles == null) throw new ArgumentNullException(nameof(triangles));
        if (triangles.Length % 3 != 0) throw new ArgumentException("Triangles length must be a multiple of 3.", nameof(triangles));

        var newVerts = new List<Vector3>(vertices.Length);
        var newTris = new int[triangles.Length];

        int[] oldToNew = new int[vertices.Length];

        if (tolerance <= 0f)
        {
            // Exact match
            var map = new Dictionary<Vector3, int>(vertices.Length);

            for (int i = 0; i < vertices.Length; i++)
            {
                var v = vertices[i];
                if (!map.TryGetValue(v, out int newIndex))
                {
                    newIndex = newVerts.Count;
                    newVerts.Add(v);
                    map.Add(v, newIndex);
                }
                oldToNew[i] = newIndex;
            }
        }
        else
        {
            // Tolerance match (quantize to a 3D grid)
            var map = new Dictionary<Vec3Key, int>(vertices.Length);

            for (int i = 0; i < vertices.Length; i++)
            {
                var v = vertices[i];
                var key = new Vec3Key(v, tolerance);

                if (!map.TryGetValue(key, out int newIndex))
                {
                    newIndex = newVerts.Count;
                    newVerts.Add(v);
                    map.Add(key, newIndex);
                }
                oldToNew[i] = newIndex;
            }
        }

        for (int t = 0; t < triangles.Length; t++)
        {
            int oldIndex = triangles[t];
            newTris[t] = oldToNew[oldIndex];
        }

        vertices = newVerts.ToArray();
        triangles = newTris;
    }
    private readonly struct Vec3Key : IEquatable<Vec3Key>
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

    // bary = (lambdaA, lambdaB, lambdaC, lambdaD) for (a,b,c,d)
    // Returns true for inside OR on boundary (faces/edges/vertices), within epsilon.
    public static bool TryGetBarycentricInclusive(
        float3 p,
        float3 a, float3 b, float3 c, float3 d,
        out float4 bary,
        float epsilon = 1e-6f,
        bool renormalize = true)
    {
        float3 ab = b - a;
        float3 ac = c - a;
        float3 ad = d - a;

        float3x3 M = new float3x3(ab, ac, ad); // columns
        float det = math.determinant(M);
        if (math.abs(det) <= epsilon)
        {
            bary = default;
            return false; // degenerate / near-degenerate tet
        }

        float3 uvw = math.mul(math.inverse(M), p - a);
        float u = uvw.x;
        float v = uvw.y;
        float w = uvw.z;

        bary = new float4(1f - u - v - w, u, v, w);

        if (renormalize)
        {
            float s = bary.x + bary.y + bary.z + bary.w;
            if (math.abs(s) > epsilon) bary /= s;
        }

        // Inclusive test (counts faces/edges/vertices as inside).
        // Upper bound isn't strictly necessary if sum==1, but helps with numerical drift.
        return math.all(bary >= -epsilon) && math.all(bary <= 1f + epsilon);
    }

#region Debug
    private void OnDrawGizmosSelected()
    {
        if (mesh == null) return;

        if (debugLatticeCells)
        {
            for (int f = 0; f < 8; f++)
            {
                if (f == 0) Gizmos.color = Color.red;
                if (f == 1) Gizmos.color = Color.orange;
                if (f == 2) Gizmos.color = Color.yellow;
                if (f == 3) Gizmos.color = Color.green;
                if (f == 4) Gizmos.color = Color.cyan;
                if (f == 5) Gizmos.color = Color.blue;
                if (f == 6) Gizmos.color = Color.magenta;
                if (f == 7) Gizmos.color = Color.white;
                for (int g = 0; g < distanceConGroups[f].Count; g++)
                {
                    for (int i = distanceConGroups[f][g].x; i < distanceConGroups[f][g].y; i++)
                    {
                        Vector3 lineStart = particles[distanceConstraints[f][i].pidx.x];
                        Vector3 lineEnd = particles[distanceConstraints[f][i].pidx.y];

                        Gizmos.DrawLine(lineStart, lineEnd);
                    }
                }
            }
        }

        if (debugCollisionMesh)
        {
            Gizmos.color = Color.red;
            for (int i = 0; i < collisionVertices.Length; i++)
            {
                float4 b = collisionVertices[i].baryCoords;
                int4 idx = collisionVertices[i].tetraIdx;
                Vector3 pos = b.x * particles[idx.x] +
                            b.y * particles[idx.y] +
                            b.z * particles[idx.z] +
                            b.w * particles[idx.w];

                Gizmos.DrawSphere(pos, 0.05f);
            }

            for (int i = 0; i < collisionEdges.Length; i++)
            {
                float4 b1 = collisionVertices[collisionEdges[i].x].baryCoords;
                int4 idx1 = collisionVertices[collisionEdges[i].x].tetraIdx;
                Vector3 pos1 = b1.x * particles[idx1.x] +
                            b1.y * particles[idx1.y] +
                            b1.z * particles[idx1.z] +
                            b1.w * particles[idx1.w];

                float4 b2 = collisionVertices[collisionEdges[i].y].baryCoords;
                int4 idx2 = collisionVertices[collisionEdges[i].y].tetraIdx;
                Vector3 pos2 = b2.x * particles[idx2.x] +
                            b2.y * particles[idx2.y] +
                            b2.z * particles[idx2.z] +
                            b2.w * particles[idx2.w];

                Gizmos.DrawLine(pos1, pos2);
            }
        }
    }
#endregion
#endregion
}

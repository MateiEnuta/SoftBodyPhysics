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

public struct VertexTriLinear
{
    public int4 idx03, idx47;
    public float3 triLinearValues;

    public VertexTriLinear(int4 idx03, int4 idx47, float3 triLinearValues)
    {
        this.idx03 = idx03;
        this.idx47 = idx47;
        this.triLinearValues = triLinearValues;
    }
}

public struct CollisionVertex
{
    public int4 idx03, idx47;
    public float3 triLinearValues;
    public float invMass;

    public CollisionVertex(int4 idx03, int4 idx47, float3 triLinearValues, float invMass)
    {
        this.idx03 = idx03;
        this.idx47 = idx47;
        this.triLinearValues = triLinearValues;
        this.invMass = invMass;
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

    // Mesh data
    private Mesh mesh;
    private int numberOfVertices;
    private Vector3[] meshVertices;

    // Lattice data
    // Main lattice
    private int[] latticeCells;
    private float3 latticeNegativeCorner;
    private float3 latticeLengths;
    private int latticeCount;
    // Sub lattice
    private int[] subLatticeCells;
    private int3 subLatticeDimensions;
    private float3 subLatticeNegativeCorner;
    private float3 subLatticeLenghts;
    private int subLatticeCount;

    // Particle data
    private int3 particlesDimensions;
    private List<float3> particles;
    private int[] particlesDictionary;
    private VertexTriLinear[] verticesTriLinear;

    // Constraint data
    private HashSet<int2> addedEdges; // Redundancy list
    private List<int2>[] distanceConGroups;
    private List<DistanceConstraintRef>[] distanceConstraints;
    private List<VolumeConstraintRef>[] volumeConstraints;

    // Collision mesh data
    private CollisionVertex[] collisionVertices;
    private int3[] collisionTriangles;
    private int2[] collisionEdges;
#endregion

#region Initialisation
    private void Awake()
    {
        mesh = GetComponent<MeshFilter>().mesh;
        if (mesh == null) return;
        numberOfVertices = mesh.vertexCount;
        mesh.MarkDynamic();
        mesh.SetVertexBufferParams(numberOfVertices, 
        new VertexAttributeDescriptor(VertexAttribute.Position, VertexAttributeFormat.Float32, 3, stream: 0),
        new VertexAttributeDescriptor(VertexAttribute.Normal, VertexAttributeFormat.Float32, 3, stream: 1),
        new VertexAttributeDescriptor(VertexAttribute.TexCoord0, VertexAttributeFormat.Float32, 2, stream: 2)
        );

        // Initialising functions
        InitLatticeAndParticles();
        InitConstraints();
        // Main functions
        LoopVertices();
        LoopInsideCells();
        // Transforms particle positions from local mesh space to world space
        for (int i = 0; i < particles.Count; i++) particles[i] = transform.TransformPoint(particles[i]);
        // Finalization functions
        CleanUp();
        UploadDataToSolver();

        transform.position = Vector3.zero;
        transform.rotation = Quaternion.identity;
        transform.localScale = Vector3.one;
    }

    float EPSILON = 1e-4f;
    private void InitLatticeAndParticles()
    {
        latticeDimensions = math.max(latticeDimensions, new int3(1, 1, 1));
        latticeNegativeCorner = (float3)mesh.bounds.min - new float3(EPSILON, EPSILON, EPSILON);
        latticeLengths = (float3)mesh.bounds.size + new float3(2 * EPSILON, 2 * EPSILON, 2 * EPSILON);
        latticeLengths /= (float3)latticeDimensions;

        particlesDimensions = latticeDimensions + new int3(1, 1, 1);

        subLatticeMultiplier = math.max(subLatticeMultiplier, 2);
        subLatticeDimensions = latticeDimensions * subLatticeMultiplier;
        subLatticeNegativeCorner = (float3)mesh.bounds.min - new float3(EPSILON, EPSILON, EPSILON);
        subLatticeLenghts = (float3)mesh.bounds.size + new float3(2 * EPSILON, 2 * EPSILON, 2 * EPSILON);
        subLatticeLenghts /= (float3)subLatticeDimensions;

        latticeCount = (int)(latticeDimensions.x * latticeDimensions.y * latticeDimensions.z);
        subLatticeCount = (int)(subLatticeDimensions.x * subLatticeDimensions.y * subLatticeDimensions.z);
        int numberOfParticles = (int)(particlesDimensions.x * particlesDimensions.y * particlesDimensions.z);

        // -1 for unassigned, -2 for outside and 0 for inside
        latticeCells = new int[latticeCount];
        for (int i = 0; i < latticeCount; i++) latticeCells[i] = -1;
        subLatticeCells = new int[subLatticeCount];
        for (int i = 0; i < subLatticeCount; i++) subLatticeCells[i] = -1;

        // -1 for unassigned and >= 0 for assigned with the particle index
        particles = new List<float3>(numberOfParticles);
        particlesDictionary = new int[numberOfParticles];
        for (int i = 0; i < numberOfParticles; i++) particlesDictionary[i] = -1;
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

#region Solver addition

    public void UploadDataToSolver()
    {
        JobPhysicsSolver solver = FindFirstObjectByType(typeof(JobPhysicsSolver)) as JobPhysicsSolver;

        solver.AddSoftBody(
            this, mass, distanceCompliance, volumeCompliance,
            particles, distanceConGroups, distanceConstraints,
            volumeConstraints, verticesTriLinear
        );
    }

    public void UpdateMesh(NativeArray<float3> vertices, int dataStart)
    {
        mesh.SetVertexBufferData(vertices, dataStart, 0, numberOfVertices);
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
        verticesTriLinear = new VertexTriLinear[numberOfVertices];

        for (int i = 0; i < numberOfVertices; i++)
        {
            // Calculates the lattice cell of the current mesh vertex
            float3 positionInLattice = (float3)meshVertices[i] - latticeNegativeCorner;
            int3 negativeCorner = (int3)(positionInLattice / latticeLengths);
            int cellID = to1D(negativeCorner, latticeDimensions);

            // Try adding the containing cell
            AddLatticeCell(negativeCorner, cellID);

            // Stores data for trilinear interpolation
            verticesTriLinear[i] = new VertexTriLinear(
                new int4(TryAddParticle(negativeCorner + new int3(0, 0, 0)),
                          TryAddParticle(negativeCorner + new int3(1, 0, 0)),
                          TryAddParticle(negativeCorner + new int3(1, 1, 0)),
                          TryAddParticle(negativeCorner + new int3(0, 1, 0))),
                new int4(TryAddParticle(negativeCorner + new int3(0, 0, 1)),
                          TryAddParticle(negativeCorner + new int3(1, 0, 1)),
                          TryAddParticle(negativeCorner + new int3(1, 1, 1)),
                          TryAddParticle(negativeCorner + new int3(0, 1, 1))),
                math.frac(positionInLattice / latticeLengths)
            );
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
        AddVolumeConstraints(pidx, cellParity);

        latticeCells[cellID] = 0;
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
        }


        // Add cells that are inside using flood fill for speed
        // Hash set for casses where it hits an edge of the triangle
        HashSet<int> edgeHits = new HashSet<int>();

        // Queue for both the main and sub lattice flood fills
        Queue<int> queue = new Queue<int>(subLatticeCount * (1/8));

        // Main lattice flood fill
        FillMainLattice(colMeshVerts, colMeshTris, edgeHits, queue);
        // Sub lattice flood fill
        FillSubLattice(colMeshVerts, colMeshTris, edgeHits, queue);

        // Create collision mesh data
        // Disperses mass and assigns the collision vertices
        DisperseSubLatticeMass(colMeshVerts);

        // Assign triangles and edges
        collisionTriangles = new int3[colMeshTris.Length / 3];
        HashSet<int2> edgeHash = new HashSet<int2>();
        for (int i = 0; i < colMeshTris.Length / 3; i++)
        {
            // Get the coresponding triangle vertex indexes
            int i0 = colMeshTris[i*3];
            int i1 = colMeshTris[i*3+1];
            int i2 = colMeshTris[i*3+2];

            // Assign triangles
            collisionTriangles[i] = new int3(i0, i1, i2);

            // Adds the edges with ascending order in the hash
            edgeHash.Add(i0 < i1 ? new int2(i0, i1) : new int2(i1, i0));
            edgeHash.Add(i1 < i2 ? new int2(i1, i2) : new int2(i2, i1));
            edgeHash.Add(i2 < i0 ? new int2(i2, i0) : new int2(i0, i2));
        }

        // Assigns edges
        collisionEdges = edgeHash.ToArray();
    }

#region Triangle Case
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
        if (closestPoint.x > cellNegativeCorner.x + cellLenghts.x || closestPoint.x < cellNegativeCorner.x)
            return false;
        if (closestPoint.y > cellNegativeCorner.y + cellLenghts.y || closestPoint.y < cellNegativeCorner.y)
            return false;
        if (closestPoint.z > cellNegativeCorner.z + cellLenghts.z || closestPoint.z < cellNegativeCorner.z)
            return false;
        return true;
    }
#endregion

#region Inside Case
    /// <summary>
    /// Fills the main lattice.
    /// </summary>
    private void FillMainLattice(Vector3[] colMeshVerts, int[] colMeshTris,
            HashSet<int> edgeHits, Queue<int> queue)
    {
        for (int i = 0; i < latticeCount; i++)
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
        for (int i = 0; i < subLatticeCount; i++)
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
                    else subLatticeCells[nextCellID] = 0;
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

#region Mass dispersion
    /// <summary>
    /// Disperses the mass on the populated sub lattice cells.
    /// A populated cell is one in which at least one collision vertex is present.
    /// Runs flood fill to disperse the whole lattice.
    /// And creates the collision vertices list.
    /// </summary>
    private void DisperseSubLatticeMass(Vector3[] colMeshVerts)
    {
        // Queue for setting sub lattice cell influence
        Queue<int2> massQueue = new Queue<int2>(subLatticeCount * (1/8));

        // For cases where 2 collision vertices are in the same sub cell
        int[] massDivider = new int[colMeshVerts.Length];

        // Loop though each particle and enqueue their values
        for (int i = 0; i < colMeshVerts.Length; i++)
        {
            float3 posInSubLattice = (float3)colMeshVerts[i] - subLatticeNegativeCorner;
            int3 negativeCorner = (int3)(posInSubLattice / subLatticeLenghts);
            int cellID = to1D(negativeCorner, subLatticeDimensions);

            // Check if the cellID is empty
            if (subLatticeCells[cellID] == 0)
            {
                massDivider[i] = 1;
                subLatticeCells[cellID] = 1;
                massQueue.Enqueue(new int2(cellID, cellID));
            }
            else
            {
                // If not then loop though the particles until now and update them if they share the same cell
                for (int j = 0; j < i; j++)
                {
                    float3 secPosInSubLattice = (float3)colMeshVerts[j] - subLatticeNegativeCorner;
                    int3 secNegativeCorner = (int3)(secPosInSubLattice / subLatticeLenghts);
                    int secCellID = to1D(secNegativeCorner, subLatticeDimensions);

                    // Increment if the secondary ID is the same
                    if (cellID == secCellID) {
                        massDivider[j]++;
                        massDivider[i] = massDivider[j];
                    }
                }
            }
        }

        // Number of active sub cells, used for even distribution
        int activeSubCellCount = 0;

        // Flood fill with incrementing of the starting cell
        while (massQueue.Count > 0)
        {
            activeSubCellCount++;
            int2 currentCell = massQueue.Dequeue();
            int3 cellCoords = to3D(currentCell.x, subLatticeDimensions);

            TryFloodAndIncrementCell(currentCell, cellCoords, 0, false, massQueue);
            TryFloodAndIncrementCell(currentCell, cellCoords, 0, true, massQueue);
            TryFloodAndIncrementCell(currentCell, cellCoords, 1, false, massQueue);
            TryFloodAndIncrementCell(currentCell, cellCoords, 1, true, massQueue);
            TryFloodAndIncrementCell(currentCell, cellCoords, 2, false, massQueue);
            TryFloodAndIncrementCell(currentCell, cellCoords, 2, true, massQueue);
        }

        // Mass of each sub cell
        float subCellMass = mass / activeSubCellCount;
        
        collisionVertices = new CollisionVertex[colMeshVerts.Length];

        // Loop though each collision vertex add trilinear values and calculate mass
        for (int i = 0; i < colMeshVerts.Length; i++)
        {
            // position in main lattice
            float3 posInMainLattice = (float3)colMeshVerts[i] - latticeNegativeCorner;
            int3 mainNegativeCorner = (int3)(posInMainLattice / latticeLengths);

            // position and cellID for the sub lattice
            float3 posInSubLattice = (float3)colMeshVerts[i] - subLatticeNegativeCorner;
            int3 negativeCorner = (int3)(posInSubLattice / subLatticeLenghts);
            int subCellID = to1D(negativeCorner, subLatticeDimensions);

            // Create new collision vertex with the trilinear based on main mesh
            // and invMass based on sub mesh cell influence and how many other particles
            // are in the same cell (massDivider)
            collisionVertices[i] = new CollisionVertex(
                new int4(TryAddParticle(mainNegativeCorner + new int3(0, 0, 0)),
                          TryAddParticle(mainNegativeCorner + new int3(1, 0, 0)),
                          TryAddParticle(mainNegativeCorner + new int3(1, 1, 0)),
                          TryAddParticle(mainNegativeCorner + new int3(0, 1, 0))),
                new int4(TryAddParticle(mainNegativeCorner + new int3(0, 0, 1)),
                          TryAddParticle(mainNegativeCorner + new int3(1, 0, 1)),
                          TryAddParticle(mainNegativeCorner + new int3(1, 1, 1)),
                          TryAddParticle(mainNegativeCorner + new int3(0, 1, 1))),
                math.frac(posInMainLattice / latticeLengths),
                1f / (subLatticeCells[subCellID] * subCellMass / massDivider[i])
            );
        }
    }

    /// <summary>
    /// Tries flood filing the given adjacent cell, if it succed it would also increment
    /// the starting cell influence.
    /// </summary>
    private void TryFloodAndIncrementCell(int2 currentCell, int3 cellCoords, int axis, bool sign, Queue<int2> massQueue)
    {
        if (cellCoords[axis] + (sign ? 1 : -1) < subLatticeDimensions[axis]
            && cellCoords[axis] + (sign ? 1 : -1) >= 0)
        {
            int3 addition = int3.zero;
            addition[axis] = sign ? 1 : -1;
            int3 nextCellCords = cellCoords + addition;
            int nextCellID = to1D(nextCellCords, subLatticeDimensions);

            if (subLatticeCells[nextCellID] == 0)
            {
                subLatticeCells[nextCellID] = -1;
                massQueue.Enqueue(new int2(nextCellID, currentCell.y));
                subLatticeCells[currentCell.y]++;
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
        meshVertices = null;
        latticeCells = null;
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

#region Debug
    private void OnDrawGizmosSelected()
    {
        if (mesh == null) return;

        // Gizmos.color = Color.green;
        // for (int x = 0; x < subLatticeDimensions.x; x++)
        // {
        //     for (int y = 0; y < subLatticeDimensions.y; y++)
        //     {
        //         for (int z = 0; z < subLatticeDimensions.z; z++)
        //         {
        //             if (subLatticeCells[to1D(new int3(x, y, z), subLatticeDimensions)] != 1) continue;
        //             float3 cellCenter = subLatticeNegativeCorner + new float3(x + 0.5f, y + 0.5f, z + 0.5f) * subLatticeLenghts;
        //             Gizmos.DrawWireCube(cellCenter, subLatticeLenghts);
        //         }
        //     }
        // }

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
#endregion
#endregion
}

using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

#region Variables
public class DistanceConstraintRef
{
    public uint2 pidx;
    public float restLenght;

    public DistanceConstraintRef(uint2 pidx, float restLenght)
    {
        this.pidx = pidx;
        this.restLenght = restLenght;
    }
}
public class VolumeConstraintRef
{
    public uint4 pidx;
    public float restVolume;

    public VolumeConstraintRef(uint4 pidx, float restVolume)
    {
        this.pidx = pidx;
        this.restVolume = restVolume;
    }
}
public class LatticeCell
{
    public uint cellID;
    public uint[] latticeParticles;
    public List<uint> meshVertices;

    public LatticeCell(uint cellID, uint[] latticeParticles)
    {
        this.cellID = cellID;
        this.latticeParticles = latticeParticles;
        meshVertices = new List<uint>();
    }
}

public class VertexTriLinear
{
    public uint4 idx03, idx47;
    public float3 triLinearValues;

    public VertexTriLinear(uint4 idx03, uint4 idx47, float3 triLinearValues)
    {
        this.idx03 = idx03;
        this.idx47 = idx47;
        this.triLinearValues = triLinearValues;
    }
}

public class ObjectInstanceManager : MonoBehaviour
{
    // Public object data
    [SerializeField] private Mesh collisionMesh;
    [SerializeField] private uint3 latticeDimensions;
    [SerializeField] private float mass = 1;
    [SerializeField] private float distanceCompliance = 0;
    [SerializeField] private float volumeCompliance = 0;


    // Mesh data
    private Mesh mesh;
    private Vector3[] meshVertices;
    private VertexTriLinear[] verticesInter;

    // Lattice data
    private float3 latticeNegativeCorner;
    private float3 latticeLengths;
    private Dictionary<uint, LatticeCell> latticeCells;

    // Particle data
    private uint3 particlesDimensions;
    private List<float3> particles;
    private Dictionary<uint, uint> particlesDictionary;

    // Constraint data
    private HashSet<uint2> addedEdges; // Redundancy list
    private List<uint2>[] distanceConGroups;
    private List<DistanceConstraintRef>[] distanceConstraints;
    private List<VolumeConstraintRef>[] volumeConstraints;
#endregion

#region Initialisation
    private void Awake()
    {
        mesh = GetComponent<MeshFilter>().mesh;
        if (mesh == null) return;

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
    }

    private void InitConstraints()
    {
        // Distance constraints data
        addedEdges = new HashSet<uint2>(); // for duplicate check
        distanceConGroups = new List<uint2>[8];
        distanceConstraints = new List<DistanceConstraintRef>[8];

        // Volume constraints data
        volumeConstraints = new List<VolumeConstraintRef>[8];

        for (int i = 0; i < 8; i++)
        {
            // Distance constraints data
            distanceConGroups[i] = new List<uint2>();
            distanceConstraints[i] = new List<DistanceConstraintRef>();

            // Volume constraints data
            volumeConstraints[i] = new List<VolumeConstraintRef>();
        }
    }

    float EPSILON = 1e-4f;
    private void InitLatticeAndParticles()
    {
        latticeDimensions = math.max(latticeDimensions, new uint3(1, 1, 1));
        particlesDimensions = latticeDimensions + new uint3(1, 1, 1);
        latticeNegativeCorner = (float3)mesh.bounds.min - new float3(EPSILON, EPSILON, EPSILON);
        latticeLengths = (float3)mesh.bounds.size + new float3(2 * EPSILON, 2 * EPSILON, 2 * EPSILON);
        latticeLengths /= (float3)latticeDimensions;

        int numberOfCells = (int)(latticeDimensions.x * latticeDimensions.y * latticeDimensions.z);
        int numberOfParticles = (int)(particlesDimensions.x * particlesDimensions.y * particlesDimensions.z);

        latticeCells = new Dictionary<uint, LatticeCell>(numberOfCells);
        particles = new List<float3>(numberOfParticles);
        particlesDictionary = new Dictionary<uint, uint>(numberOfParticles);
    }
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
        verticesInter = new VertexTriLinear[mesh.vertexCount];

        for (int i = 0; i < meshVertices.Length; i++)
        {
            // Calculates the lattice cell of the current mesh vertex
            float3 positionInLattice = (float3)meshVertices[i] - latticeNegativeCorner;
            uint3 negativeCorner = (uint3)(positionInLattice / latticeLengths);
            uint cellID = to1D(negativeCorner, latticeDimensions);

            // Try adding the containing cell
            AddLatticeCell(negativeCorner, cellID);
            // Update the containing cell
            latticeCells[cellID].meshVertices.Add((uint)i);

            // Stores data for trilinear interpolation
            verticesInter[i] = new VertexTriLinear(
                new uint4(TryAddParticle(negativeCorner + new uint3(0, 0, 0)),
                          TryAddParticle(negativeCorner + new uint3(1, 0, 0)),
                          TryAddParticle(negativeCorner + new uint3(1, 1, 0)),
                          TryAddParticle(negativeCorner + new uint3(0, 1, 0))), 
                new uint4(TryAddParticle(negativeCorner + new uint3(0, 0, 1)),
                          TryAddParticle(negativeCorner + new uint3(1, 0, 1)),
                          TryAddParticle(negativeCorner + new uint3(1, 1, 1)),
                          TryAddParticle(negativeCorner + new uint3(0, 1, 1))), 
                positionInLattice / latticeLengths
            );
        }
    }
#endregion

#region Cell addition
    /// <summary>
    /// Adds new cells, each newly added cell also has all of it's
    /// constraints added and indexed.
    /// </summary>
    private void AddLatticeCell(uint3 negativeCorner, uint cellID)
    {
        if (!latticeCells.ContainsKey(cellID))
        {
            // Adds particles in a specific order
            uint[] pidx =
            {
                TryAddParticle(negativeCorner + new uint3(0, 0, 0)), // 0
                TryAddParticle(negativeCorner + new uint3(1, 0, 0)), // 1
                TryAddParticle(negativeCorner + new uint3(1, 1, 0)), // 2
                TryAddParticle(negativeCorner + new uint3(0, 1, 0)), // 3
                TryAddParticle(negativeCorner + new uint3(0, 0, 1)), // 4
                TryAddParticle(negativeCorner + new uint3(1, 0, 1)), // 5
                TryAddParticle(negativeCorner + new uint3(1, 1, 1)), // 6
                TryAddParticle(negativeCorner + new uint3(0, 1, 1)), // 7
            };

            uint cellParity = negativeCorner.x%2 + negativeCorner.y%2 * 2 + negativeCorner.z%2 * 4;
            
            AddDistanceConstraints(pidx, cellParity);
            AddVolumeConstraints(pidx, cellParity);

            LatticeCell newCell = new LatticeCell(cellID, pidx);
            latticeCells.Add(cellID, newCell);
        }
    }

    /// <summary>
    /// Returns the index of the present or newly added vertex.
    /// </summary>
    private uint TryAddParticle(uint3 position)
    {
        uint particleIndex = to1D(position, particlesDimensions);
        
        if (particlesDictionary.TryGetValue(particleIndex, out uint index)) return index;

        particlesDictionary.Add(particleIndex, (uint)particles.Count);
        particles.Add(latticeNegativeCorner + position * latticeLengths);
        return (uint)particles.Count - 1;
    }

    private void AddDistanceConstraints(uint[] pidx, uint cellParity)
    {
        uint initDisCount, disCount;
        initDisCount = disCount = (uint)distanceConstraints[cellParity].Count;
        for (int x = 0; x < 8; x++)
        {
            for (int y = x+1; y < 8; y++)
            {
                uint2 edge = new uint2(pidx[x], pidx[y]);

                if (addedEdges.Contains(edge) || addedEdges.Contains(edge.yx)) continue;

                addedEdges.Add(edge);
                disCount++;
                distanceConstraints[cellParity].Add(new DistanceConstraintRef(edge, 
                    math.distance(transform.TransformPoint(particles[(int)edge.x]), 
                                transform.TransformPoint(particles[(int)edge.y]))));
            }
        }
        distanceConGroups[cellParity].Add(new uint2(initDisCount, disCount));
    }

    private void AddVolumeConstraints(uint[] pidx, uint cellParity)
    {
        int[,] tetrahedra = new int[,]
        {
            {0,2,7,5}, {0,5,1,2}, {0,5,7,4}, {0,2,3,7}, {6,5,7,2}
        };

        for (int i = 0; i < 5; i++)
        {
            int p0 = (int)pidx[tetrahedra[i, 0]];
            int p1 = (int)pidx[tetrahedra[i, 1]];
            int p2 = (int)pidx[tetrahedra[i, 2]];
            int p3 = (int)pidx[tetrahedra[i, 3]];

            float volume = math.dot(
                transform.TransformPoint(particles[p1]) - transform.TransformPoint(particles[p0]), 
                math.cross(transform.TransformPoint(particles[p2]) - transform.TransformPoint(particles[p0]), 
                    transform.TransformPoint(particles[p3]) - transform.TransformPoint(particles[p0]))
            ) / 6f;
            // Debug.Log($"Tetra {i} volume = {volume}");

            volumeConstraints[cellParity].Add(new VolumeConstraintRef(new uint4((uint)p0, (uint)p1, (uint)p2, (uint)p3), volume));
        }
    }
#endregion

#region Collision shape addition
    /// <summary>
    /// Loops though cells that were not added yet and checks
    /// if they either intersect the collision mesh or fully
    /// inside of it. 
    /// Also handles the creation of the collision mesh and 
    /// it's constriants.
    /// </summary>
    private void LoopInsideCells()
    {
        if (collisionMesh == null) return;
        Vector3[] colMeshVerts = collisionMesh.vertices;
        int[] colMeshTris = collisionMesh.triangles;
        // Hash set for casses where it hits an edge of the triangle
        HashSet<int> edgeHits = new HashSet<int>();

        int numberOfCells = (int)(latticeDimensions.x * latticeDimensions.y * latticeDimensions.z);
        // A random direction for avoiding hitting a triangle on the edges
        float3 randDir = math.normalize(UnityEngine.Random.insideUnitSphere);

        // Loops through each cell and only checks those that were not added already
        for (uint cellID = 0; cellID < numberOfCells; cellID++)
        {
            if (latticeCells.ContainsKey(cellID)) continue;
            uint3 negativeCorner = to3D(cellID);
            float3 negativeCornerPos = latticeNegativeCorner + negativeCorner * latticeLengths;

            edgeHits.Clear();
            int intersections = 0;

            // Loops through the triangles of the mesh collider and checks for intersections
            for (int i = 0; i < colMeshTris.Length; i+=3)
            {
                float3 v0 = colMeshVerts[colMeshTris[i]];
                float3 v1 = colMeshVerts[colMeshTris[i+1]];
                float3 v2 = colMeshVerts[colMeshTris[i+2]];

                // Main check for casses where the cell is completly inside the collider mesh
                float result = RayIntersectsTriangleDetailed(negativeCornerPos, randDir, v0, v1, v2);
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

                // Secundary check where the cell collides with the mesh but not inside of it
                if (CellIntersectsTriangle(negativeCornerPos, v0, v1, v2))
                {
                    intersections = 1;
                    break;
                }
            }

            // Test fail condition, either outside or interected with the mesh
            if (intersections % 2 == 0) continue;

            AddLatticeCell(negativeCorner, cellID);
        }
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

    private bool CellIntersectsTriangle(float3 negativeCornerPos, float3 v0, float3 v1, float3 v2)
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
#endregion

#region Helper functions
    public uint to1D(uint3 pos, uint3 dimensions)
    {
        return (pos.z * dimensions.x * dimensions.y) + (pos.y * dimensions.x) + pos.x;
    }

    public uint3 to3D(uint idx)
    {
        uint3 result = uint3.zero;
        result.z = idx / (latticeDimensions.x * latticeDimensions.y);
        idx -= result.z * latticeDimensions.x * latticeDimensions.y;
        result.y = idx / latticeDimensions.x;
        result.x = idx % latticeDimensions.x;

        return result;
    }

    private void OnDrawGizmosSelected()
    {
        if (mesh == null) return;

        // Gizmos.color = Color.green;
        // for (int x = 0; x < latticeDimensions.x; x++)
        // {
        //     for (int y = 0; y < latticeDimensions.y; y++)
        //     {
        //         for (int z = 0; z < latticeDimensions.z; z++)
        //         {
        //             float3 cellCenter = latticeNegativeCorner + new float3(x + 0.5f, y + 0.5f, z + 0.5f) * latticeLengths;
        //             Gizmos.DrawWireCube(cellCenter, latticeLengths);
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
            for (uint g = 0; g < distanceConGroups[f].Count; g++)
            {
                for (uint i = distanceConGroups[f][(int)g].x; i < distanceConGroups[f][(int)g].y; i++)
                {
                    Vector3 lineStart = particles[(int)distanceConstraints[f][(int)i].pidx.x];
                    Vector3 lineEnd = particles[(int)distanceConstraints[f][(int)i].pidx.y];

                    Gizmos.DrawLine(lineStart, lineEnd);
                }
            }
        }
    }
#endregion
}

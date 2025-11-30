using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public class DistanceConstraint
{
    public uint2 pidx;
    public float restLenght;

    public DistanceConstraint(uint2 pidx, float restLenght)
    {
        this.pidx = pidx;
        this.restLenght = restLenght;
    }
}
public class VolumeConstraint
{
    public uint4 pidx;
    public float restVolume;

    public VolumeConstraint(uint4 pidx, float restVolume)
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

public class ObjectInstanceManager : MonoBehaviour
{
    // Mesh data
    [SerializeField] private Mesh collisionMesh;
    private Mesh mesh;
    private float3[] meshVertices;

    // Lattice data
    [SerializeField] private uint3 latticeDimensions;
    private float3 latticeNegativeCorner;
    private float3 latticeLengths;
    private Dictionary<uint, LatticeCell> latticeCells;

    // Particle data
    private uint3 particlesDimensions;
    private List<float3> particles;
    private Dictionary<uint, uint> particlesDictionary;

    // Constraint data
    private HashSet<uint2> addedEdges;
    private List<uint2>[] disLatticeGroups;
    private List<DistanceConstraint>[] distanceConstraints;
    private List<VolumeConstraint>[] volumeConstraints;

    private void Awake()
    {
        mesh = GetComponent<MeshFilter>().mesh;
        if (mesh == null) return;
        meshVertices = new float3[mesh.vertexCount];
        for (int i = 0; i < mesh.vertexCount; i++)
        {
            meshVertices[i] = mesh.vertices[i];
        }

        // Setup
        latticeDimensions = math.max(latticeDimensions, new uint3(1, 1, 1));
        particlesDimensions = latticeDimensions + new uint3(1, 1, 1);

        addedEdges = new HashSet<uint2>();
        disLatticeGroups = new List<uint2>[8];
        distanceConstraints = new List<DistanceConstraint>[8];
        volumeConstraints = new List<VolumeConstraint>[8];
        for (int i = 0; i < 8; i++)
        {
            disLatticeGroups[i] = new List<uint2>();
            distanceConstraints[i] = new List<DistanceConstraint>();
            volumeConstraints[i] = new List<VolumeConstraint>();
        }


        InitLattice();
        LoopVertices();
        LoopInsideCells();

        for (int i = 0; i < particles.Count; i++) particles[i] = transform.TransformPoint(particles[i]);
    }

    float EPSILON = 1e-4f;
    private void InitLattice()
    {
        latticeNegativeCorner = (float3)mesh.bounds.min - new float3(EPSILON, EPSILON, EPSILON);
        latticeLengths = (float3)mesh.bounds.size + new float3(2 * EPSILON, 2 * EPSILON, 2 * EPSILON);
        latticeLengths /= (float3)latticeDimensions;

        int numberOfCells = (int)(latticeDimensions.x * latticeDimensions.y * latticeDimensions.z);
        int numberOfParticles = (int)(particlesDimensions.x * particlesDimensions.y * particlesDimensions.z);

        latticeCells = new Dictionary<uint, LatticeCell>(numberOfCells);
        particles = new List<float3>(numberOfParticles);
        particlesDictionary = new Dictionary<uint, uint>(numberOfParticles);
    }

    private void LoopVertices()
    {
        for (int i = 0; i < meshVertices.Length; i++)
        {
            float3 positionInLattice = meshVertices[i] - latticeNegativeCorner;
            uint3 negativeCorner = (uint3)(positionInLattice / latticeLengths);
            uint cellID = to1D(negativeCorner, latticeDimensions);

            AddLatticeCell(negativeCorner, cellID);
            latticeCells[cellID].meshVertices.Add((uint)i);
        }
    }

    private void AddLatticeCell(uint3 negativeCorner, uint cellID)
    {
        if (!latticeCells.ContainsKey(cellID))
        {
            uint[] pidx =
            {
                TryAddParticle(negativeCorner + new uint3(0, 0, 0)),
                TryAddParticle(negativeCorner + new uint3(1, 0, 0)),
                TryAddParticle(negativeCorner + new uint3(1, 1, 0)),
                TryAddParticle(negativeCorner + new uint3(0, 1, 0)),
                TryAddParticle(negativeCorner + new uint3(0, 0, 1)),
                TryAddParticle(negativeCorner + new uint3(1, 0, 1)),
                TryAddParticle(negativeCorner + new uint3(1, 1, 1)),
                TryAddParticle(negativeCorner + new uint3(0, 1, 1)),
            };

            uint cellParity = negativeCorner.x%2 + negativeCorner.y%2 * 2 + negativeCorner.z%2 * 4;
            
            AddDistanceConstraints(pidx, cellParity);
            AddVolumeConstraints(pidx, cellParity);

            LatticeCell newCell = new LatticeCell(cellID, pidx);
            latticeCells.Add(cellID, newCell);
        }
    }

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
                distanceConstraints[cellParity].Add(new DistanceConstraint(edge, 
                    math.distance(particles[(int)edge.x], particles[(int)edge.y])));
            }
        }
        disLatticeGroups[cellParity].Add(new uint2(initDisCount, disCount));
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
                particles[p1] - particles[p0], math.cross(particles[p2] - particles[p0], particles[p3] - particles[p0])
            ) / 6f;
            // Debug.Log($"Tetra {i} volume = {volume}");

            volumeConstraints[cellParity].Add(new VolumeConstraint(new uint4((uint)p0, (uint)p1, (uint)p2, (uint)p3), volume));
        }
    }

    private void LoopInsideCells()
    {
        if (collisionMesh == null) return;
        Vector3[] colMeshVerts = collisionMesh.vertices;
        int[] colMeshTris = collisionMesh.triangles;
        HashSet<int> edgeHits = new HashSet<int>();

        int numberOfCells = (int)(latticeDimensions.x * latticeDimensions.y * latticeDimensions.z);
        float3 randDir = math.normalize(UnityEngine.Random.insideUnitSphere);

        for (uint cellID = 0; cellID < numberOfCells; cellID++)
        {
            if (latticeCells.ContainsKey(cellID)) continue;
            uint3 negativeCorner = to3D(cellID);
            float3 negativeCornerPos = latticeNegativeCorner + negativeCorner * latticeLengths;

            edgeHits.Clear();
            int intersections = 0;

            for (int i = 0; i < colMeshTris.Length; i+=3)
            {
                float3 v0 = colMeshVerts[colMeshTris[i]];
                float3 v1 = colMeshVerts[colMeshTris[i+1]];
                float3 v2 = colMeshVerts[colMeshTris[i+2]];

                float result = RayIntersectsTriangleDetailed(negativeCornerPos, randDir, v0, v1, v2);

                if (result == -2f) intersections++;
                else if (result > 0)
                {
                    int key = Mathf.RoundToInt(result * 100000f);
                    if (!edgeHits.Add(key))
                    {
                        edgeHits.Remove(key);
                        intersections++;
                    }
                }

                if (CellIntersectsTriangle(negativeCornerPos, v0, v1, v2))
                {
                    intersections = 1;
                    break;
                }
            }

            // Test failed condition
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
            for (uint g = 0; g < disLatticeGroups[f].Count; g++)
            {
                for (uint i = disLatticeGroups[f][(int)g].x; i < disLatticeGroups[f][(int)g].y; i++)
                {
                    Vector3 lineStart = particles[(int)distanceConstraints[f][(int)i].pidx.x];
                    Vector3 lineEnd = particles[(int)distanceConstraints[f][(int)i].pidx.y];

                    Gizmos.DrawLine(lineStart, lineEnd);
                }
            }
        }
    }
}

using System.Collections.Generic;
using JetBrains.Annotations;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

public class DistanceConstraint
{
    
}
public class VolumeConstraints
{
    
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

public class vertex
{
    uint c0, c1, c2, c3;
    uint c4, c5, c6, c7;
    float3 weights;
}

public class ObjectInstanceManager : MonoBehaviour
{
    // Mesh data
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


        InitLattice();
        LoopVertices();
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
        }
    }

    private void AddLatticeCell(uint3 negativeCorner, uint cellID)
    {
        if (!latticeCells.ContainsKey(cellID))
        {
            uint[] particles =
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
            LatticeCell newCell = new LatticeCell(cellID, particles);


            latticeCells.Add(cellID, newCell);
        }
    }

    private uint TryAddParticle(uint3 position)
    {
        uint particleIndex = to1D(position, particlesDimensions);
        
        if (particlesDictionary.TryGetValue(particleIndex, out uint index)) return index;

        particlesDictionary.Add(particleIndex, (uint)particles.Count);
        particles.Add(position * latticeLengths);
        return (uint)particles.Count - 1;
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

        Gizmos.color = Color.green;
        for (int x = 0; x < latticeDimensions.x; x++)
        {
            for (int y = 0; y < latticeDimensions.y; y++)
            {
                for (int z = 0; z < latticeDimensions.z; z++)
                {
                    float3 cellCenter = latticeNegativeCorner + new float3(x + 0.5f, y + 0.5f, z + 0.5f) * latticeLengths;
                    Gizmos.DrawWireCube(cellCenter, latticeLengths);
                }
            }
        }
    }
}

using System.Collections.Generic;
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
    public DistanceConstraint[] distanceConstraints;
    public VolumeConstraints[] volumeConstraints;
}

public class ObjectInstanceManager : MonoBehaviour
{
    [SerializeField] private int3 latticeDimensions;

    private Mesh mesh;

    private float3[] meshVertices;

    private float3 latticeNegativeCorner;
    private float3 latticeLengths;
    private Dictionary<int3, LatticeCell>[] latticeCells;

    private void Awake()
    {
        mesh = GetComponent<MeshFilter>().mesh;
        if (mesh == null) return;
        meshVertices = new float3[mesh.vertexCount];
        for (int i = 0; i < mesh.vertexCount; i++)
        {
            meshVertices[i] = mesh.vertices[i];
        }

        // Safety checks
        latticeDimensions = math.max(latticeDimensions, new int3(1, 1, 1));


        InitLattice();
        LoopVertices();
    }

    float EPSILON = 1e-4f;
    private void InitLattice()
    {
        latticeNegativeCorner = (float3)mesh.bounds.min - new float3(EPSILON, EPSILON, EPSILON);
        latticeLengths = (float3)mesh.bounds.size + new float3(2 * EPSILON, 2 * EPSILON, 2 * EPSILON);
        latticeLengths /= (float3)latticeDimensions;

        int numberOfCells = latticeDimensions.x * latticeDimensions.y * latticeDimensions.z;

        latticeCells = new Dictionary<int3, LatticeCell>[8];
        for (int i = 0; i < 8; i++)
        {
            latticeCells[i] = new Dictionary<int3, LatticeCell>(numberOfCells/8);
        }
    }

    private void LoopVertices()
    {
        for (int i = 0; i < meshVertices.Length; i++)
        {
            float3 positionInLattice = meshVertices[i] - latticeNegativeCorner;
            int3 negativeCorner = (int3)(positionInLattice / latticeLengths);

            AddLatticeCell(negativeCorner);
        }
    }

    private void AddLatticeCell(int3 negativeCorner)
    {
        
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

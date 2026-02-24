using UnityEngine;
using System;
using Unity.Mathematics;

[Serializable]
public class LatticePreprocessor
{
    public float3 targetLatticeDimensions = new float3(1,1,1);


    private float3 latticeNegativeCorner;
    private int3 latticeResolution;
    private float3 latticeDimensions;

    public void ProcessLatticeBody(LatticeBody latticeBody)
    {
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

        
    }
}

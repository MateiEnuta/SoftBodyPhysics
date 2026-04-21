using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using SolverStructs;

public class CollisionAlgoTest : MonoBehaviour
{
    public List<LatticeBody> collisionBodies;
    public float3 targetLatticeDimensions = new float3(0.5f);
    public float hashCellSize = 1f;

    private List<LatticePreprocessor> preprocessors;
    private List<float3> currentParticles;
    private List<LatticeCell> currentCells;

    // Collision test data
    private List<int2> touchedCells = new List<int2>();

    private void Awake()
    {
        if (collisionBodies == null) return;
        targetLatticeDimensions = math.max(targetLatticeDimensions, new float3(0.2f));
        preprocessors = new List<LatticePreprocessor>(collisionBodies.Count);

        foreach (var body in collisionBodies)
        {
            body.Initialize();
            preprocessors.Add(new LatticePreprocessor(body, targetLatticeDimensions));
        }
    }


    private void OnDrawGizmos()
    {
        if (collisionBodies == null || preprocessors == null) return;
        if (currentParticles == null) currentParticles = new List<float3>();
        if (currentCells == null) currentCells = new List<LatticeCell>();
        currentParticles.Clear();
        currentCells.Clear();

        // Populate data lists
        foreach (var body in preprocessors)
        {
            int initialParticleCount = currentParticles.Count;
            
            for (int i = 0; i < body.activeParticles.Count; i++)
            {
                // Calculate the particle position in the lattice
                int3 particleCoords = body.to3D_Particle(body.activeParticles[i]);
                float3 posInLattice = body.latticeNegativeCorner + particleCoords * body.latticeDimensions;
                
                // Transform the position to world space
                float3 posInWorld = body.latticeBody.transform.TransformPoint(posInLattice);
                currentParticles.Add(posInWorld);
            }

            for (int i = 0; i < body.finalLatticeCells.Length; i++)
            {
                LatticeCell currentCell = body.finalLatticeCells[i];
                currentCell.v0 += initialParticleCount;
                currentCell.v1 += initialParticleCount;
                currentCell.v2 += initialParticleCount;
                currentCell.v3 += initialParticleCount;
                currentCell.v4 += initialParticleCount;
                currentCell.v5 += initialParticleCount;
                currentCell.v6 += initialParticleCount;
                currentCell.v7 += initialParticleCount;
                currentCells.Add(currentCell);
            }
        }


        // Run the collision testing algorithm
        touchedCells.Clear();
        for (int i = 0; i < currentCells.Count; i++)
        {
            float3 p0 = currentParticles[currentCells[i].v0];
            float3 p1 = currentParticles[currentCells[i].v1];
            float3 p2 = currentParticles[currentCells[i].v2];
            float3 p3 = currentParticles[currentCells[i].v3];
            float3 p4 = currentParticles[currentCells[i].v4];
            float3 p5 = currentParticles[currentCells[i].v5];
            float3 p6 = currentParticles[currentCells[i].v6];
            float3 p7 = currentParticles[currentCells[i].v7];

            float3 minPos = math.min(math.min(math.min(p0, p1), math.min(p2, p3)), math.min(math.min(p4, p5), math.min(p6, p7)));
            float3 maxPos = math.max(math.max(math.max(p0, p1), math.max(p2, p3)), math.max(math.max(p4, p5), math.max(p6, p7)));

            float invCellSize = 1.0f / hashCellSize;

            int3 minCell = (int3)math.floor(minPos * invCellSize);
            int3 maxCell = (int3)math.floor(maxPos * invCellSize);

            int HashCell(int3 c)
            {
                unchecked
                {
                    return c.x * 73856093 ^ c.y * 19349663 ^ c.z * 83492791;
                }
            }

            for (int z = minCell.z; z <= maxCell.z; z++)
            for (int y = minCell.y; y <= maxCell.y; y++)
            for (int x = minCell.x; x <= maxCell.x; x++)
            {
                int3 cellCoords = new int3(x, y, z);
                int cellID = HashCell(cellCoords);

                touchedCells.Add(new int2(cellID, i));
            }
        }

        // Sort based on the cellID's
        touchedCells.Sort((a, b) => a.x.CompareTo(b.x));

        // Create unuquie collision pairs
        for (int i = 0; i < touchedCells.Count - 1; i++)
        {
            for (int j = i + 1; j < touchedCells.Count; j++)
            {
                if (touchedCells[i].x != touchedCells[j].x) break;

                
            }
        }


        // ------particle drawing------
        Gizmos.color = Color.coral;
        foreach (var particle in currentParticles) Gizmos.DrawSphere(particle, 0.05f);

        // ------Cell drawing------
        Gizmos.color = Color.green;
        foreach (var cell in currentCells)
        {
            Gizmos.DrawLine(currentParticles[cell.v0], currentParticles[cell.v1]);
            Gizmos.DrawLine(currentParticles[cell.v2], currentParticles[cell.v3]);
            Gizmos.DrawLine(currentParticles[cell.v4], currentParticles[cell.v5]);
            Gizmos.DrawLine(currentParticles[cell.v6], currentParticles[cell.v7]);

            Gizmos.DrawLine(currentParticles[cell.v0], currentParticles[cell.v2]);
            Gizmos.DrawLine(currentParticles[cell.v1], currentParticles[cell.v3]);
            Gizmos.DrawLine(currentParticles[cell.v4], currentParticles[cell.v6]);
            Gizmos.DrawLine(currentParticles[cell.v5], currentParticles[cell.v7]);

            Gizmos.DrawLine(currentParticles[cell.v0], currentParticles[cell.v4]);
            Gizmos.DrawLine(currentParticles[cell.v1], currentParticles[cell.v5]);
            Gizmos.DrawLine(currentParticles[cell.v2], currentParticles[cell.v6]);
            Gizmos.DrawLine(currentParticles[cell.v3], currentParticles[cell.v7]);
        }
    }
}

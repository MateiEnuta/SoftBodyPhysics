using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using SolverStructs;
using Unity.Collections;
using static Unity.Mathematics.math;

public class CollisionAlgoTest : MonoBehaviour
{
    public List<LatticeBody> collisionBodies;
    public float3 targetLatticeDimensions = new float3(0.5f);
    public float hashCellSize = 1f;
    public bool debugLatticeCells = false;

    private List<LatticePreprocessor> preprocessors;
    private List<float3> currentParticles;
    private List<LatticeCell> currentCells;

    // Collision test data
    private List<int2> touchedCells = new List<int2>();
    private HashSet<int2> collisionPairs = new HashSet<int2>();

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

        // Populate data lists
        currentParticles = new List<float3>();
        currentCells = new List<LatticeCell>();
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
    }

    private void Update()
    {
        if (collisionBodies == null || preprocessors == null ||
            currentParticles == null || currentCells == null) return;

        // Update lists
        int pIdx = 0;
        foreach (var body in preprocessors)
        for (int i = 0; i < body.activeParticles.Count; i++)
        {
            // Calculate the particle position in the lattice
            int3 particleCoords = body.to3D_Particle(body.activeParticles[i]);
            float3 posInLattice = body.latticeNegativeCorner + particleCoords * body.latticeDimensions;
            
            // Transform the position to world space
            float3 posInWorld = body.latticeBody.transform.TransformPoint(posInLattice);
            currentParticles[pIdx++] = posInWorld;
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

            float3 minPos = min(min(min(p0, p1), min(p2, p3)), min(min(p4, p5), min(p6, p7)));
            float3 maxPos = max(max(max(p0, p1), max(p2, p3)), max(max(p4, p5), max(p6, p7)));

            float invCellSize = 1.0f / hashCellSize;

            int3 minCell = (int3)floor(minPos * invCellSize);
            int3 maxCell = (int3)floor(maxPos * invCellSize);

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
        touchedCells.Sort((int2 a, int2 b) => a.x.CompareTo(b.x));

        // Create unique collision pairs
        collisionPairs.Clear();
        for (int i = 0; i < touchedCells.Count - 1; i++)
        for (int j = i + 1; j < touchedCells.Count; j++)
        {
            // Go to the next i cell if they are not in the same cell
            if (touchedCells[i].x != touchedCells[j].x) break;

            // Conditions for creating a pair
            if (touchedCells[i].y == touchedCells[j].y) continue;
            LatticeCell cellA = currentCells[touchedCells[i].y];
            LatticeCell cellB = currentCells[touchedCells[j].y];

            bool valid = true;
            for (int avi = 0; avi < 8; avi++)
            for (int bvi = 0; bvi < 8; bvi++)
            {
                if (GetCellVertex(cellA, avi) == GetCellVertex(cellB, bvi)) valid = false;
            }
            if (!valid) continue;

            // Create only if they are not the same cells or touching in ascending index order
            if (touchedCells[i].y < touchedCells[j].y)
                collisionPairs.Add(new int2(touchedCells[i].y, touchedCells[j].y));
            else
                collisionPairs.Add(new int2(touchedCells[j].y, touchedCells[i].y));
        }

        // Check for collision
        foreach (var colPair in collisionPairs)
        {
            LatticeCell cellA = currentCells[colPair.x];
            LatticeCell cellB = currentCells[colPair.y];

            float3 posA = (currentParticles[cellA.v0] + currentParticles[cellA.v7]) * 0.5f;
            float3 posB = (currentParticles[cellB.v0] + currentParticles[cellB.v7]) * 0.5f;

            if (!CellsIntersectSAT(cellA, cellB, currentParticles)) continue;

            Debug.DrawLine(posA, posB, Color.pink, Time.deltaTime);
        }
    }


    private void OnDrawGizmos()
    {
        if (collisionBodies == null || preprocessors == null ||
            currentParticles == null || currentCells == null) return;
        if (!debugLatticeCells) return;
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


    static int GetCellVertex(LatticeCell c, int i)
    {
        switch (i)
        {
            case 0: return c.v0;
            case 1: return c.v1;
            case 2: return c.v2;
            case 3: return c.v3;
            case 4: return c.v4;
            case 5: return c.v5;
            case 6: return c.v6;
            default:return c.v7;
        }
    }
    // Assumes your cell indexing is box-like and that from vertex 0,
    // vertices 1, 3, and 4 are the 3 adjacent corners.
    // If your lattice indexing differs, just change those 3 picks.
    static void GetCellVertices(LatticeCell c, List<float3> particles, float3[] v)
    {
        for (int i = 0; i < 8; i++)
            v[i] = particles[GetCellVertex(c, i)];
    }

    static bool OverlapOnAxis(float3[] a, float3[] b, float3 axis)
    {
        float lenSq = lengthsq(axis);
        if (lenSq < 1e-8f)
            return true; // degenerate cross-product axis, ignore

        axis *= rsqrt(lenSq); // normalize

        float minA = dot(a[0], axis);
        float maxA = minA;

        for (int i = 1; i < 8; i++)
        {
            float p = dot(a[i], axis);
            minA = min(minA, p);
            maxA = max(maxA, p);
        }

        float minB = dot(b[0], axis);
        float maxB = minB;

        for (int i = 1; i < 8; i++)
        {
            float p = dot(b[i], axis);
            minB = min(minB, p);
            maxB = max(maxB, p);
        }

        return !(maxA < minB || maxB < minA);
    }

    static bool CellsIntersectSAT(
        LatticeCell aCell,
        LatticeCell bCell,
        List<float3> particles)
    {
        float3[] a = new float3[8];
        float3[] b = new float3[8];

        GetCellVertices(aCell, particles, a);
        GetCellVertices(bCell, particles, b);

        float3 ax = normalize(a[1] - a[0]);
        float3 ay = normalize(a[3] - a[0]);
        float3 az = normalize(a[4] - a[0]);

        float3 bx = normalize(b[1] - b[0]);
        float3 by = normalize(b[3] - b[0]);
        float3 bz = normalize(b[4] - b[0]);

        // 3 face-normal axes from A
        if (!OverlapOnAxis(a, b, ax)) return false;
        if (!OverlapOnAxis(a, b, ay)) return false;
        if (!OverlapOnAxis(a, b, az)) return false;

        // 3 face-normal axes from B
        if (!OverlapOnAxis(a, b, bx)) return false;
        if (!OverlapOnAxis(a, b, by)) return false;
        if (!OverlapOnAxis(a, b, bz)) return false;

        // 9 cross-product axes
        if (!OverlapOnAxis(a, b, cross(ax, bx))) return false;
        if (!OverlapOnAxis(a, b, cross(ax, by))) return false;
        if (!OverlapOnAxis(a, b, cross(ax, bz))) return false;

        if (!OverlapOnAxis(a, b, cross(ay, bx))) return false;
        if (!OverlapOnAxis(a, b, cross(ay, by))) return false;
        if (!OverlapOnAxis(a, b, cross(ay, bz))) return false;

        if (!OverlapOnAxis(a, b, cross(az, bx))) return false;
        if (!OverlapOnAxis(a, b, cross(az, by))) return false;
        if (!OverlapOnAxis(a, b, cross(az, bz))) return false;

        return true;
    }
}

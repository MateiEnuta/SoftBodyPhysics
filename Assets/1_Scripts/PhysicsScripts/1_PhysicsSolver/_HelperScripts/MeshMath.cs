using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public static class MeshMath
{
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

    // vertices[i] has cellIds[i], triangles list uses indices into vertices.
    public static void RemoveDuplicateVerticesPerCell(
        List<float3> vertices,
        List<int> cellIds,
        List<int4> triangles,
        float tolerance = 0f)
    {
        if (vertices == null) throw new ArgumentNullException(nameof(vertices));
        if (cellIds == null) throw new ArgumentNullException(nameof(cellIds));
        if (triangles == null) throw new ArgumentNullException(nameof(triangles));
        if (vertices.Count != cellIds.Count)
            throw new ArgumentException("vertices and cellIds must have same length.");

        int vertCount = vertices.Count;

        var newVerts   = new List<float3>(vertCount);
        var newCellIds = new List<int>(vertCount);
        var newTris    = new List<int4>(triangles.Count);

        int[] oldToNew = new int[vertCount];

        if (tolerance <= 0f)
        {
            var map = new Dictionary<CellVec3KeyExact, int>(vertCount);

            for (int i = 0; i < vertCount; i++)
            {
                var key = new CellVec3KeyExact(vertices[i], cellIds[i]);
                if (!map.TryGetValue(key, out int newIndex))
                {
                    newIndex = newVerts.Count;
                    newVerts.Add(vertices[i]);
                    newCellIds.Add(cellIds[i]);
                    map.Add(key, newIndex);
                }
                oldToNew[i] = newIndex;
            }
        }
        else
        {
            var map = new Dictionary<CellVec3KeyQuant, int>(vertCount);

            for (int i = 0; i < vertCount; i++)
            {
                var key = new CellVec3KeyQuant(vertices[i], cellIds[i], tolerance);
                if (!map.TryGetValue(key, out int newIndex))
                {
                    newIndex = newVerts.Count;
                    newVerts.Add(vertices[i]);
                    newCellIds.Add(cellIds[i]);
                    map.Add(key, newIndex);
                }
                oldToNew[i] = newIndex;
            }
        }

        // Remap int4 triangles: x,y,z are indices, w is kept (e.g. cell or anything else)
        for (int i = 0; i < triangles.Count; i++)
        {
            int4 tri = triangles[i];
            tri.x = oldToNew[tri.x];
            tri.y = oldToNew[tri.y];
            tri.z = oldToNew[tri.z];
            newTris.Add(tri);
        }

        // Overwrite input lists
        vertices.Clear();
        vertices.AddRange(newVerts);

        cellIds.Clear();
        cellIds.AddRange(newCellIds);

        triangles.Clear();
        triangles.AddRange(newTris);
    }

    private readonly struct CellVec3KeyExact : IEquatable<CellVec3KeyExact>
    {
        private readonly int cellId;
        private readonly float x, y, z;

        public CellVec3KeyExact(float3 v, int cellId)
        {
            this.cellId = cellId;
            x = v.x; y = v.y; z = v.z;
        }

        public bool Equals(CellVec3KeyExact other)
            => cellId == other.cellId && x == other.x && y == other.y && z == other.z;

        public override bool Equals(object obj)
            => obj is CellVec3KeyExact other && Equals(other);

        public override int GetHashCode()
        {
            unchecked
            {
                int h = 17;
                h = h * 31 + cellId;
                h = h * 31 + x.GetHashCode();
                h = h * 31 + y.GetHashCode();
                h = h * 31 + z.GetHashCode();
                return h;
            }
        }
    }

    private readonly struct CellVec3KeyQuant : IEquatable<CellVec3KeyQuant>
    {
        private readonly int cellId;
        private readonly int qx, qy, qz;

        public CellVec3KeyQuant(float3 v, int cellId, float tol)
        {
            this.cellId = cellId;
            float inv = 1f / tol;
            qx = (int)math.round(v.x * inv);
            qy = (int)math.round(v.y * inv);
            qz = (int)math.round(v.z * inv);
        }

        public bool Equals(CellVec3KeyQuant other)
            => cellId == other.cellId && qx == other.qx && qy == other.qy && qz == other.qz;

        public override bool Equals(object obj)
            => obj is CellVec3KeyQuant other && Equals(other);

        public override int GetHashCode()
        {
            unchecked
            {
                int h = 17;
                h = h * 31 + cellId;
                h = h * 31 + qx;
                h = h * 31 + qy;
                h = h * 31 + qz;
                return h;
            }
        }
    }

    public static List<int3> BuildCellSortedEdges(List<int4> triangles)
    {
        if (triangles == null) throw new ArgumentNullException(nameof(triangles));

        var edges = new List<int3>(triangles.Count * 2);

        int i = 0;
        int triCount = triangles.Count;

        // hash used only per cell
        var set = new HashSet<int2>();

        while (i < triCount)
        {
            int cellId = triangles[i].w;

            while (i < triCount && triangles[i].w == cellId)
            {
                int4 t = triangles[i];

                AddEdge(set, t.x, t.y);
                AddEdge(set, t.y, t.z);
                AddEdge(set, t.z, t.x);

                i++;
            }

            // Emit unique edges for this cell
            foreach (var e in set)
            {
                edges.Add(new int3(e.x, e.y, cellId));
            }
            set.Clear();
        }

        return edges;
    }

    static void AddEdge(HashSet<int2> set, int v0, int v1)
    {
        if (v0 == v1) return;
        if (v1 < v0) (v0, v1) = (v1, v0); // Swap to be in ascending order
        set.Add(new int2(v0, v1));
    }
}

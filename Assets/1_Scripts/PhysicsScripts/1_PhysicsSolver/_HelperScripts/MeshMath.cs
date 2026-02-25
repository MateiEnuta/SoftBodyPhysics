using System;
using System.Collections.Generic;
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
}

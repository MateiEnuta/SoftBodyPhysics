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

    /// <summary>
    /// Checks if a given point is inside the given mesh.
    /// Uses ray intersection test with a way to detect edge hits.
    /// </summary>
    public static bool IsPointInsideMesh(float3 point, HashSet<int> edgeHits, Vector3[] colMeshVerts, int[] colMeshTris)
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
    public static float RayIntersectsTriangleDetailed(float3 rayOrigin, float3 rayDir, float3 v0, float3 v1, float3 v2)
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
}

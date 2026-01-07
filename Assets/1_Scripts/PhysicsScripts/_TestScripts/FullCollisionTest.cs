using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class FullCollisionTest : MonoBehaviour
{
    public struct TriangleOrder
    {
        public int a, b, c;
        public float distance;

        public TriangleOrder(int a, int b, int c, float distance)
        {
            this.a = a;
            this.b = b;
            this.c = c;
            this.distance = distance;
        }
    }

    public Transform previousPointPos, currentPointPos;
    private Mesh colMesh;
    private Vector3[] vertices;
    private int[] triangles;
    
    void OnDrawGizmos()
    {
        DisplayMesh();
        
        if (previousPointPos == null || currentPointPos == null) return;

        Gizmos.color = Color.green;
        Gizmos.DrawSphere(previousPointPos.position, 0.05f);
        Gizmos.DrawSphere(currentPointPos.position, 0.05f);
        Gizmos.DrawLine(previousPointPos.position, currentPointPos.position);

        DeflectPoint();
    }

    private void DisplayMesh()
    {
        colMesh = GetComponent<MeshFilter>().sharedMesh;
        if (colMesh == null) return;
        //Gizmos.DrawWireMesh(colMesh, transform.position, transform.rotation);
        
        vertices = colMesh.vertices;
        triangles = colMesh.triangles;
    }

    private void DeflectPoint()
    {
        List<TriangleOrder> trianglesOrdered = new List<TriangleOrder>();

        for (int i = 0; i < triangles.Length; i+=3)
        {
            Vector3 p1 = vertices[triangles[i]];
            Vector3 p2 = vertices[triangles[i+1]];
            Vector3 p3 = vertices[triangles[i+2]];

            Vector3 velocityDir = currentPointPos.position - previousPointPos.position;
            velocityDir.Normalize();

            Vector3 intersectionPoint = RayTriangleUtil.RayToTriangleClamped(
                previousPointPos.position, velocityDir, p1, p2, p3,
                out bool usedParallelFallback, out bool rayIntersectsTriangle, 
                out float planeT
            );
            Vector3 closestPoint = RayTriangleUtil.ClosestPointOnTriangle(
                previousPointPos.position, p1, p2, p3
            );

            Vector3 intersectionDirection = intersectionPoint - previousPointPos.position;
            intersectionDirection.Normalize();

            float alligment = Vector3.Dot(intersectionDirection, currentPointPos.position - previousPointPos.position);
            float distanceToIntersection = Vector3.Distance(previousPointPos.position, intersectionPoint);

            // float totalDistance = Vector3.Distance(
            //     previousPointPos.position, 
            //     previousPointPos.position + velocityDir * planeT
            // );
            // if (!rayIntersectsTriangle) totalDistance += Vector3.Distance(
            //     intersectionPoint, 
            //     previousPointPos.position + velocityDir * planeT
            // );

            float relativeDistance = alligment - distanceToIntersection;
            if (relativeDistance >= 0) trianglesOrdered.Add(
                new TriangleOrder(
                    triangles[i], triangles[i+1], triangles[i+2], 
                    relativeDistance
                )
            );
        }

        trianglesOrdered.Sort(
            (x, y) => y.distance.CompareTo(x.distance)
        );

        DebugOrder(trianglesOrdered);

        Gizmos.color = Color.red;
        Vector3 previousPos = previousPointPos.position;
        Vector3 currentPos = currentPointPos.position;
        for (int i = 0; i < trianglesOrdered.Count; i++)
        {
            Vector3 p1 = vertices[trianglesOrdered[i].a];
            Vector3 p2 = vertices[trianglesOrdered[i].b];
            Vector3 p3 = vertices[trianglesOrdered[i].c];

            if (RayTriangleUtil.IntersectSegmentTriangleAndSlide(
                previousPos, currentPos, p1, p2, p3, out Vector3 hitPoint, out Vector3 slideEnd,
                out float tOnSegment
            ))
            {
                Vector3 ab = p2 - p1;
                Vector3 ac = p3 - p1;

                // Triangle normal from winding; direction depends on clockwise/counterclockwise,
                Vector3 n = Vector3.Cross(ab, ac);

                previousPos = hitPoint + (n * 0.001f);
                currentPos = slideEnd + (n * 0.001f);

                Gizmos.DrawLine(previousPos, currentPos);
            }
        }
        Gizmos.DrawSphere(currentPos, 0.05f);
    }

    public int clock = 0;
    private int step = 1;
    private void DebugOrder(List<TriangleOrder> trianglesOrdered)
    {
        Gizmos.color = Color.white;
        if (clock / step > trianglesOrdered.Count) clock = 0;
        for (int i = 0; i < clock / step; i++)
        {
            Vector3 p1 = vertices[trianglesOrdered[i].a];
            Vector3 p2 = vertices[trianglesOrdered[i].b];
            Vector3 p3 = vertices[trianglesOrdered[i].c];

            Gizmos.DrawLine(p1, p2);
            Gizmos.DrawLine(p2, p3);
            Gizmos.DrawLine(p3, p1);
        }
        //clock++;
    }
}


public static class RayTriangleUtil
{
    // Returns a point guaranteed to be on/inside the triangle.
    // If the ray hits the triangle's plane, it clamps that plane-hit to the triangle.
    // If the ray is nearly parallel to the plane, it returns the closest point on the triangle
    // to the ray origin projected onto the plane (stable fallback).
    public static Vector3 RayToTriangleClamped(
        Vector3 rayOrigin,
        Vector3 rayDir,
        Vector3 a,
        Vector3 b,
        Vector3 c,
        out bool usedParallelFallback,
        out bool rayIntersectsTriangle,
        out float planeT,
        float parallelEpsilon = 1e-6f)
    {
        usedParallelFallback = false;
        planeT = 0f;

        // Handle "ray" with no direction as just a point query.
        if (rayDir.sqrMagnitude < 1e-12f)
        {
            usedParallelFallback = true;
            rayIntersectsTriangle = false;
            return ClosestPointOnTriangle(rayOrigin, a, b, c);
        }

        Vector3 ab = b - a;
        Vector3 ac = c - a;

        // Triangle normal from winding; direction depends on clockwise/counterclockwise,
        // but plane intersection/clamping does not require a specific facing.
        Vector3 n = Vector3.Cross(ab, ac);
        float nSqr = n.sqrMagnitude;

        // Degenerate triangle (collinear or near-zero area).
        if (nSqr < 1e-12f)
        {
            usedParallelFallback = true;
            rayIntersectsTriangle = false;
            // Best effort: return closest of vertices to the origin
            Vector3 pa = a, pb = b, pc = c;
            float da = (rayOrigin - pa).sqrMagnitude;
            float db = (rayOrigin - pb).sqrMagnitude;
            float dc = (rayOrigin - pc).sqrMagnitude;
            return (da <= db && da <= dc) ? pa : (db <= dc ? pb : pc);
        }

        float denom = Vector3.Dot(n, rayDir);

        // If denom ~ 0, ray direction is (near) parallel to the plane [web:6].
        if (Mathf.Abs(denom) <= parallelEpsilon)
        {
            usedParallelFallback = true;
            rayIntersectsTriangle = false;

            // Project ray origin onto the plane along the plane normal (stable).
            Vector3 nUnit = n / Mathf.Sqrt(nSqr);
            float signedDist = Vector3.Dot(nUnit, rayOrigin - a);
            Vector3 originOnPlane = rayOrigin - signedDist * nUnit;

            return ClosestPointOnTriangle(originOnPlane, a, b, c);
        }

        // Ray-plane intersection: t = dot(n, (a - O)) / dot(n, D)
        planeT = Vector3.Dot(n, a - rayOrigin) / denom;

        // If plane intersection is behind the ray origin, use a stable fallback.
        if (planeT < 0f)
        {
            usedParallelFallback = true;
            rayIntersectsTriangle = false;
            return ClosestPointOnTriangle(rayOrigin, a, b, c);
        }

        Vector3 planeHit = rayOrigin + rayDir * planeT;
        Vector3 pointOnTriangle = ClosestPointOnTriangle(planeHit, a, b, c);

        if (Equals(planeHit, pointOnTriangle)) rayIntersectsTriangle = true;
        else rayIntersectsTriangle = false;

        // Clamp to triangle so result is always inside/on edges/vertices [page:1].
        return pointOnTriangle;
    }

    // Closest point on triangle ABC to point P.
    // This is the standard region-test method popularized by Real-Time Collision Detection [page:1].
    public static Vector3 ClosestPointOnTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 ab = b - a;
        Vector3 ac = c - a;
        Vector3 ap = p - a;

        float d1 = Vector3.Dot(ab, ap);
        float d2 = Vector3.Dot(ac, ap);
        if (d1 <= 0f && d2 <= 0f) return a; // barycentric (1,0,0)

        Vector3 bp = p - b;
        float d3 = Vector3.Dot(ab, bp);
        float d4 = Vector3.Dot(ac, bp);
        if (d3 >= 0f && d4 <= d3) return b; // barycentric (0,1,0)

        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0f && d1 >= 0f && d3 <= 0f)
        {
            float v = d1 / (d1 - d3);
            return a + v * ab; // edge AB
        }

        Vector3 cp = p - c;
        float d5 = Vector3.Dot(ab, cp);
        float d6 = Vector3.Dot(ac, cp);
        if (d6 >= 0f && d5 <= d6) return c; // barycentric (0,0,1)

        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0f && d2 >= 0f && d6 <= 0f)
        {
            float w = d2 / (d2 - d6);
            return a + w * ac; // edge AC
        }

        float va = d3 * d6 - d5 * d4;
        if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
        {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + w * (c - b); // edge BC
        }

        // Inside face region
        float denom = 1f / (va + vb + vc);
        float v2 = vb * denom;
        float w2 = vc * denom;
        return a + ab * v2 + ac * w2;
    }

    /// <summary>
    /// Segment vs triangle. If hit, returns:
    /// - hitPoint: intersection point on the triangle
    /// - slideEnd: endpoint you get if you "slide" the remaining segment along the triangle plane
    ///
    /// Notes:
    /// - Uses a segment-adapted Möller–Trumbore test (t in [0,1]).
    /// - Does not handle coplanar segment-triangle overlap (treated as no hit).
    /// </summary>
    public static bool IntersectSegmentTriangleAndSlide(
        Vector3 p0, Vector3 p1,
        Vector3 a, Vector3 b, Vector3 c,
        out Vector3 hitPoint,
        out Vector3 slideEnd,
        out float tOnSegment,
        float epsilon = 1e-6f)
    {
        hitPoint = default;
        slideEnd = default;
        tOnSegment = 0f;

        Vector3 dir = p1 - p0;

        Vector3 e1 = b - a;
        Vector3 e2 = c - a;

        Vector3 pvec = Vector3.Cross(dir, e2);
        float det = Vector3.Dot(e1, pvec);

        // Parallel or (nearly) coplanar: treat as "no single intersection".
        if (Mathf.Abs(det) < epsilon)
            return false;

        float invDet = 1.0f / det;

        Vector3 tvec = p0 - a;
        float u = Vector3.Dot(tvec, pvec) * invDet;
        if (u < 0f || u > 1f)
            return false;

        Vector3 qvec = Vector3.Cross(tvec, e1);
        float v = Vector3.Dot(dir, qvec) * invDet;
        if (v < 0f || (u + v) > 1f)
            return false;

        // If dir is (p1 - p0), then t is in [0,1] for points along the segment.
        float t = Vector3.Dot(e2, qvec) * invDet;
        if (t < 0f || t > 1f)
            return false;

        tOnSegment = t;
        hitPoint = p0 + dir * t;

        Vector3 n = Vector3.Cross(e1, e2);
        float nSqr = n.sqrMagnitude;
        if (nSqr < 1e-12f)
        {
            // Degenerate triangle; treat as no hit for sliding purposes.
            return false;
        }

        Vector3 nUnit = n / Mathf.Sqrt(nSqr);

        // Remaining motion after impact, projected onto plane (sliding direction).
        Vector3 remaining = p1 - hitPoint;
        Vector3 remainingProjected = Vector3.ProjectOnPlane(remaining, nUnit);

        slideEnd = hitPoint + remainingProjected;
        return true;
    }
}

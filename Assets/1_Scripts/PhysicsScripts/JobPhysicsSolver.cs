using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.XR;

#region Jobs
// -----------------------------------------------------------------------------
// POSITION UPDATE JOB
// Integrates particle positions using explicit Euler (XPBD-style).
// Stores previousPosition so velocities can be recomputed later.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct PositionUpdate : IJobParallelFor
{
    [ReadOnly] public float3 gravity;
    [ReadOnly] public float dt;
    public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        // Read particle
        Particle currentParticle = particles[index];

        // Semi-implicit Euler: update velocity first
        currentParticle.velocity += gravity * dt;

        // Save previous position for velocity reconstruction
        currentParticle.previousPosition = currentParticle.position;

        // Integrate new position: x += v * dt
        currentParticle.position += currentParticle.velocity * dt;

        // Write back
        particles[index] = currentParticle;
    }
}

// -----------------------------------------------------------------------------
// VELOCITY UPDATE JOB
// Reconstructs velocities from positions after constraints have corrected them.
// XPBD keeps constraints stable by recomputing velocity from the correction.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct VelocityUpdate : IJobParallelFor
{
    [ReadOnly] public float dt;
    public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        Particle currentParticle = particles[index];

        // Velocity = (x_new - x_old) / dt
        currentParticle.velocity = 
            (currentParticle.position - currentParticle.previousPosition) / dt;

        particles[index] = currentParticle;
    }
}

// -----------------------------------------------------------------------------
// VERTEX INTERPOLATION JOB
// Computes mesh vertex positions by trilinearly interpolating 8 lattice nodes.
// Each visual vertex is a weighted blend of 8 particles (cube cell corners).
// -----------------------------------------------------------------------------
[BurstCompile]
public struct VisualVerticesUpdate : IJobParallelFor
{
    [ReadOnly] public NativeArray<Particle> particles;
    [ReadOnly] public NativeArray<VisualVertex> visualVerticesBary;
    public NativeArray<float3> verticesPositions;

    public void Execute(int index)
    {
        float4 bary = visualVerticesBary[index].baryCoords;
        int4 pIdx = visualVerticesBary[index].tetraIdx;
        float3 positon = bary.x * particles[pIdx.x].position +
                         bary.y * particles[pIdx.y].position +
                         bary.z * particles[pIdx.z].position +
                         bary.w * particles[pIdx.w].position;

        verticesPositions[index] = positon;
    }
}

// -----------------------------------------------------------------------------
// DISTANCE CONSTRAINT JOB
// Solves batches of distance constraints for a given channel.
// Each channel stores ranges [start,end) for that body.
// XPBD formulation including optional damping.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct DistanceConstraintsUpdate : IJobParallelFor
{
    [ReadOnly] public float dt;
    [ReadOnly] public float damping;

    // Every element describes a range of constraints to process.
    [ReadOnly] public NativeArray<int2> distanceConGroups;

    // Full list of actual constraints
    [NativeDisableParallelForRestriction] public NativeArray<DistanceConstraint> distanceConstraints;

    // Particle array (writes allowed)
    [NativeDisableParallelForRestriction] public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        // Range of constraints belonging to this job index
        int2 startEnd = distanceConGroups[index];

        // Process each constraint sequentially
        for (int i = startEnd.x; i < startEnd.y; i++)
        {
            DistanceConstraint constraint = distanceConstraints[i];

            // Load particle data
            Particle p1 = particles[constraint.particlesIdx.x];
            Particle p2 = particles[constraint.particlesIdx.y];

            float3 dir = p2.position - p1.position;
            float currentLength = math.length(dir);
            if (currentLength < 1e-10f) continue; // Prevent NaN

            float C = currentLength - constraint.restLength;

            // Gradients of the constraint
            float3 n = dir / currentLength;
            float3 grad1 = -n;
            float3 grad2 = n;

            // XPBD sum of w_i * |grad|^2  (for distance constraints = invMass1 + invMass2)
            float sumWeightedGradients = p1.invMass + p2.invMass;

            // Compliance term: α = compliance / dt²
            float alphaTilde = constraint.compliance / (dt * dt);
            sumWeightedGradients += alphaTilde;

            if (math.abs(sumWeightedGradients) < 1e-10f) continue;

            // -------------------------
            // Constraint damping (optional)
            // -------------------------
            float dampingTerm = 0f;
            if (damping > 0f)
            {
                float betaTilde = damping * dt * dt;
                float gamma = alphaTilde * betaTilde / dt;

                // Constraint-space velocity
                float3 v1 = p1.position - p1.previousPosition;
                float3 v2 = p2.position - p2.previousPosition;
                float relativeVelocity = math.dot(v2 - v1, grad2);

                dampingTerm = gamma * relativeVelocity;

                // Modify denominator for stable XPBD damping
                sumWeightedGradients *= 1f + gamma;
            }

            // XPBD correction
            float deltaLambda = -(C + dampingTerm) / sumWeightedGradients;

            // Apply positional correction
            p1.position += p1.invMass * deltaLambda * grad1;
            p2.position += p2.invMass * deltaLambda * grad2;

            // Write back
            particles[constraint.particlesIdx.x] = p1;
            particles[constraint.particlesIdx.y] = p2;
        }
    }
}

// -----------------------------------------------------------------------------
// VOLUME CONSTRAINT JOB
// Solves tetrahedral volume preservation constraints in batches of 5.
// Standard XPBD volume formulation with damping.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct VolumeConstraintsUpdate : IJobParallelFor
{
    [ReadOnly] public float dt;
    [ReadOnly] public float damping;

    [NativeDisableParallelForRestriction] public NativeArray<VolumeConstraint> volumeConstraints;
    [NativeDisableParallelForRestriction] public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        // Each tetra uses 5 constraints stored contiguously
        for (int i = index * 5; i < index * 5 + 5; i++)
        {
            VolumeConstraint constraint = volumeConstraints[i];

            // Load tetra particles
            Particle p1 = particles[constraint.particlesIdx.x];
            Particle p2 = particles[constraint.particlesIdx.y];
            Particle p3 = particles[constraint.particlesIdx.z];
            Particle p4 = particles[constraint.particlesIdx.w];

            // Compute signed volume
            float3 v1 = p2.position - p1.position;
            float3 v2 = p3.position - p1.position;
            float3 v3 = p4.position - p1.position;
            float3 cross23 = math.cross(v2, v3);
            float currentVolume = math.dot(v1, cross23) / 6f;

            float C = currentVolume - constraint.restVolume;

            // Gradients for tetrahedral volume constraint
            float3 cross31 = math.cross(v3, v1);
            float3 cross12 = math.cross(v1, v2);

            float3 grad1 = -(cross23 + cross31 + cross12) / 6f;
            float3 grad2 = cross23 / 6f;
            float3 grad3 = cross31 / 6f;
            float3 grad4 = cross12 / 6f;

            // Weighted gradient magnitudes
            float sumWeightedGradients = 0f;
            sumWeightedGradients += p1.invMass * math.lengthsq(grad1);
            sumWeightedGradients += p2.invMass * math.lengthsq(grad2);
            sumWeightedGradients += p3.invMass * math.lengthsq(grad3);
            sumWeightedGradients += p4.invMass * math.lengthsq(grad4);

            float alphaTilde = constraint.compliance / (dt * dt);
            sumWeightedGradients += alphaTilde;

            if (math.abs(sumWeightedGradients) < 1e-10f) continue;

            // -------------------------
            // XPBD damping term
            // -------------------------
            float dampingTerm = 0f;
            if (damping > 0f)
            {
                float betaTilde = damping * dt * dt;
                float gamma = alphaTilde * betaTilde / dt;

                float3 vel1 = p1.position - p1.previousPosition;
                float3 vel2 = p2.position - p2.previousPosition;
                float3 vel3 = p3.position - p3.previousPosition;
                float3 vel4 = p4.position - p4.previousPosition;

                float velocitySum =
                    math.dot(grad1, vel1) +
                    math.dot(grad2, vel2) +
                    math.dot(grad3, vel3) +
                    math.dot(grad4, vel4);

                dampingTerm = gamma * velocitySum;
                sumWeightedGradients *= 1f + gamma;
            }

            // Compute constraint correction
            float deltaLambda = -(C + dampingTerm) / sumWeightedGradients;

            // Apply position correction to all tetra points
            p1.position += p1.invMass * deltaLambda * grad1;
            p2.position += p2.invMass * deltaLambda * grad2;
            p3.position += p3.invMass * deltaLambda * grad3;
            p4.position += p4.invMass * deltaLambda * grad4;

            // Write back
            particles[constraint.particlesIdx.x] = p1;
            particles[constraint.particlesIdx.y] = p2;
            particles[constraint.particlesIdx.z] = p3;
            particles[constraint.particlesIdx.w] = p4;
        }
    }
}

// -----------------------------------------------------------------------------
// SIMPLE COLLISION TEST
// Keeps particles above y=0. Very crude but safe for testing.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct CrudeCollision : IJob
{
    // Lattice particles
    public NativeArray<Particle> particles;

    // Collision vertices (trilinearly embedded in the lattice)
    [ReadOnly] public NativeArray<CollisionVertex> collisionVertices;

    public void Execute()
    {
        for (int v = 0; v < collisionVertices.Length; v++)
        {
            CollisionVertex cv = collisionVertices[v];

            // Indices of the tetra particles
            int i0 = cv.tetraIdx.x;
            int i1 = cv.tetraIdx.y;
            int i2 = cv.tetraIdx.z;
            int i3 = cv.tetraIdx.w;

            float4 bary = cv.baryCoords;

            float yPos = bary.x * particles[i0].position.y +
                         bary.y * particles[i1].position.y +
                         bary.z * particles[i2].position.y +
                         bary.w * particles[i3].position.y;

            if (yPos >= 0) continue;

            float correction = -yPos;

            Particle p0 = particles[i0];
            p0.position.y += bary.x * correction;
            particles[i0] = p0;

            Particle p1 = particles[i1];
            p1.position.y += bary.y * correction;
            particles[i1] = p1;

            Particle p2 = particles[i2];
            p2.position.y += bary.z * correction;
            particles[i2] = p2;

            Particle p3 = particles[i3];
            p3.position.y += bary.w * correction;
            particles[i3] = p3;
        }
    }

    private void ApplyCorrection(int idx, float weight, float correction)
    {
        if (weight <= 0f) return;

        Particle p = particles[idx];
        p.position.y += correction * weight;
        particles[idx] = p;
    }
}

// -----------------------------------------------------------------------------
// COMPLEX COLLISION BETWEEN PARTICLES AND TRIANGLES
// Keeps particles above y=0. Very crude but safe for testing.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct PointTriCollision : IJob
{
    // Dynamic bodies
    public NativeArray<Particle> particles;
    public NativeArray<CollisionVertex> collisionVertices;
    public NativeArray<int3> collisionTriangles;

    // Static bodies
    public NativeArray<float3> staticColPoints;
    public NativeArray<int3> staticColTriangles;

    // Collision buffer list
    public NativeList<CollisionBuffer> collisionBuffer;

    public void Execute()
    {
        // for (int p = 0; p < collisionVertices.Length; p++)
        // {
        //     collisionBuffer.Clear();

        //     float3 previousPosition = TrilinearInterpolation(p, true);
        //     float3 currentPosition = TrilinearInterpolation(p, false);
        //     float3 velocityDir = math.normalize(currentPosition - previousPosition);

        //     for (int t = 0; t < collisionTriangles.Length; t++)
        //     {
        //         int3 currentTriangle = collisionTriangles[t];
        //         if (currentTriangle.x == p ||
        //             currentTriangle.y == p ||
        //             currentTriangle.z == p) continue;

        //         float3 pT1 = TrilinearInterpolation(currentTriangle.x, true);
        //         float3 pT2 = TrilinearInterpolation(currentTriangle.y, true);
        //         float3 pT3 = TrilinearInterpolation(currentTriangle.z, true);
        //         float3 cT1 = TrilinearInterpolation(currentTriangle.x, false);
        //         float3 cT2 = TrilinearInterpolation(currentTriangle.y, false);
        //         float3 cT3 = TrilinearInterpolation(currentTriangle.z, false);
                
        //         float3 collisionPoint = RayToTriangleClampedBarycentric(
        //             previousPosition, velocityDir,
        //             pT1, pT2, pT3, out float3 bary
        //         );
        //         collisionPoint += math.cross(pT2 - pT1, pT3 - pT1) * 0.001f;

        //         float3 currentTrianglePos = cT1 * bary.x + cT2 * bary.y + cT3 * bary.z;

        //         float3 collisionPointDir = math.normalize(collisionPoint - previousPosition);
        //         float pointAlligment = math.dot(collisionPointDir, currentPosition - previousPosition);
        //         float triangleAlligment = math.dot(collisionPointDir, collisionPoint - currentTrianglePos);
        //         float distanceToIntersection = math.distance(previousPosition, collisionPoint);

        //         float relativeDistance = pointAlligment - triangleAlligment - distanceToIntersection;
        //         if (relativeDistance >= 0) collisionBuffer.Add(
        //             new CollisionBuffer(t, false, relativeDistance)
        //         );
        //     }

        //     for (int t = 0; t < staticColTriangles.Length; t++)
        //     {
        //         int3 currentTriangle = staticColTriangles[t];
        //         float3 cT1 = staticColPoints[currentTriangle.x];
        //         float3 cT2 = staticColPoints[currentTriangle.y];
        //         float3 cT3 = staticColPoints[currentTriangle.z];

        //         float3 collisionPoint = RayToTriangleClampedBarycentric(
        //             previousPosition, velocityDir,
        //             cT1, cT2, cT3, out float3 bary
        //         );
        //         collisionPoint += math.normalize(math.cross(cT2 - cT1, cT3 - cT1)) * 0.001f;

        //         float3 collisionPointDir = math.normalize(collisionPoint - previousPosition);
        //         float pointAlligment = math.dot(collisionPointDir, currentPosition - previousPosition);
        //         float distanceToIntersection = math.distance(previousPosition, collisionPoint);

        //         float relativeDistance = pointAlligment - distanceToIntersection;
        //         if (relativeDistance >= 0) collisionBuffer.Add(
        //             new CollisionBuffer(t, true, relativeDistance)
        //         );
        //     }

        //     for (int i = 0; i < collisionBuffer.Length - 1; i++)
        //     {
        //         for (int j = i + 1; j < collisionBuffer.Length; j++)
        //         {
        //             if (collisionBuffer[j].distance < collisionBuffer[i].distance)
        //             {
        //                 CollisionBuffer aux = collisionBuffer[i];
        //                 collisionBuffer[i] = collisionBuffer[j];
        //                 collisionBuffer[j] = aux;
        //             }
        //         }
        //     }

        //     for (int c = 0; c < collisionBuffer.Length; c++)
        //     {
        //         CollisionBuffer currentCol = collisionBuffer[c];
        //         if (currentCol.isStatic)
        //         {
        //             SolveStaticCollision(p, ref previousPosition, ref currentPosition, currentCol.otherIdx);
        //         }
        //         else
        //         {
                    
        //         }
        //     }
        // }

        for (int p = 0; p < collisionVertices.Length; p++)
        {
            CollisionVertex cv = collisionVertices[p];
            float4 bary = cv.baryCoords;
            int4 tetIdx = cv.tetraIdx;
            float3 previousPosition = bary.x * particles[tetIdx.x].previousPosition +
                                      bary.y * particles[tetIdx.y].previousPosition +
                                      bary.z * particles[tetIdx.z].previousPosition +
                                      bary.w * particles[tetIdx.w].previousPosition;
            float3 currentPosition = bary.x * particles[tetIdx.x].position +
                                     bary.y * particles[tetIdx.y].position +
                                     bary.z * particles[tetIdx.z].position +
                                     bary.w * particles[tetIdx.w].position;

            // for (int t = 0; t < collisionTriangles.Length; t++)
            // {
            //     int3 triIdx = collisionTriangles[t];
            //     if (triIdx.x == p || triIdx.y == p || triIdx.z == p) continue;
            //     SolveDynamicCollision(p, ref previousPosition, ref currentPosition, t);
            // }

            for (int t = 0; t < staticColTriangles.Length; t++)
            {
                SolveStaticCollision(p, ref previousPosition, ref currentPosition, t);
            }
        }
    }

    private void SolveDynamicCollision(
        int pointIdx,
        ref float3 previousP,
        ref float3 currentP,
        int triIdx)
    {
        // --- triangle collision vertices ---
        int3 tri = collisionTriangles[triIdx];

        CollisionVertex cv0 = collisionVertices[tri.x];
        CollisionVertex cv1 = collisionVertices[tri.y];
        CollisionVertex cv2 = collisionVertices[tri.z];

        float3 A = cv0.baryCoords.x * particles[cv0.tetraIdx.x].position +
                   cv0.baryCoords.y * particles[cv0.tetraIdx.y].position +
                   cv0.baryCoords.z * particles[cv0.tetraIdx.z].position +
                   cv0.baryCoords.w * particles[cv0.tetraIdx.w].position;
        float3 B = cv1.baryCoords.x * particles[cv1.tetraIdx.x].position +
                   cv1.baryCoords.y * particles[cv1.tetraIdx.y].position +
                   cv1.baryCoords.z * particles[cv1.tetraIdx.z].position +
                   cv1.baryCoords.w * particles[cv1.tetraIdx.w].position;
        float3 C = cv2.baryCoords.x * particles[cv2.tetraIdx.x].position +
                   cv2.baryCoords.y * particles[cv2.tetraIdx.y].position +
                   cv2.baryCoords.z * particles[cv2.tetraIdx.z].position +
                   cv2.baryCoords.w * particles[cv2.tetraIdx.w].position;

        // --- CCD + slide (same function you trust) ---
        if (!IntersectSegmentTriangleAndSlide(
                previousP, currentP,
                A, B, C,
                out float3 hit,
                out float3 slideEnd))
            return;

        float3 n = math.normalize(math.cross(B - A, C - A));
        float3 newP = slideEnd + n * 0.001f;

        float3 delta = newP - currentP;

        // --- barycentric on triangle ---
        float3 triBary;
        ClosestPointOnTriangleBarycentric(hit, A, B, C, out triBary);

        // --- inverse masses ---
        float wP = collisionVertices[pointIdx].invMass;

        float wT =
            triBary.x * triBary.x * cv0.invMass +
            triBary.y * triBary.y * cv1.invMass +
            triBary.z * triBary.z * cv2.invMass;

        float wSum = wP + wT;
        if (wSum <= 0f) return;

        float3 dP =  delta * (wP / wSum);
        float3 dT = -delta * (wT / wSum);

        // --- apply to point ---
        ApplyCollisionVertexCorrection(pointIdx, dP);

        // --- apply to triangle vertices ---
        ApplyCollisionVertexCorrection(tri.x, triBary.x * dT);
        ApplyCollisionVertexCorrection(tri.y, triBary.y * dT);
        ApplyCollisionVertexCorrection(tri.z, triBary.z * dT);

        currentP = newP;
        previousP = hit + n * 0.001f;
    }

    private void ApplyCollisionVertexCorrection(int cvIdx, float3 delta)
    {
        CollisionVertex cv = collisionVertices[cvIdx];
        float4 bary = cv.baryCoords;

        // Indices of the tetra particles
        int i0 = cv.tetraIdx.x;
        int i1 = cv.tetraIdx.y;
        int i2 = cv.tetraIdx.z;
        int i3 = cv.tetraIdx.w;

        Particle p0 = particles[i0];
        p0.position += bary.x * delta;
        particles[i0] = p0;

        Particle p1 = particles[i1];
        p1.position += bary.y * delta;
        particles[i1] = p1;

        Particle p2 = particles[i2];
        p2.position += bary.z * delta;
        particles[i2] = p2;

        Particle p3 = particles[i3];
        p3.position += bary.w * delta;
        particles[i3] = p3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SolveStaticCollision(int p, ref float3 previousPosition, ref float3 currentPosition, int triangleIdx)
    {
        float3 cT1 = staticColPoints[staticColTriangles[triangleIdx].x];
        float3 cT2 = staticColPoints[staticColTriangles[triangleIdx].y];
        float3 cT3 = staticColPoints[staticColTriangles[triangleIdx].z];
        if (IntersectSegmentTriangleAndSlide(previousPosition, currentPosition,
            cT1, cT2, cT3, out float3 hitPoint, out float3 slideEnd
        ))
        {
            // Small normal offset
            float3 normalOffset = math.normalize(math.cross(cT2 - cT1, cT3 - cT1)) * 0.001f;

            previousPosition = hitPoint + normalOffset;
            float3 newCPos = slideEnd + normalOffset;

            CollisionVertex cv = collisionVertices[p];
            // Indices of the tetra particles
            int i0 = cv.tetraIdx.x;
            int i1 = cv.tetraIdx.y;
            int i2 = cv.tetraIdx.z;
            int i3 = cv.tetraIdx.w;

            // Compute correction based on the difference between 
            float3 correction = newCPos - currentPosition;

            // Update collision vertex positions for subsequent CCD
            currentPosition = newCPos;

            // Update actual particles positions
            Particle p0 = particles[i0];
            p0.position += correction;
            particles[i0] = p0;

            Particle p1 = particles[i1];
            p1.position += correction;
            particles[i1] = p1;

            Particle p2 = particles[i2];
            p2.position += correction;
            particles[i2] = p2;

            Particle p3 = particles[i3];
            p3.position += correction;
            particles[i3] = p3;
        }
    }

    // Returns closest point on triangle ABC to point P, and barycentric coords (u,v,w) for that point.
    // Region-test method consistent with Real-Time Collision Detection. [web:237]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float3 ClosestPointOnTriangleBarycentric(
        float3 p, float3 a, float3 b, float3 c,
        out float3 bary) // (u,v,w) sums to 1
    {
        float3 ab = b - a;
        float3 ac = c - a;
        float3 ap = p - a;

        float d1 = math.dot(ab, ap);
        float d2 = math.dot(ac, ap);

        // Vertex region A
        if (d1 <= 0f && d2 <= 0f)
        {
            bary = new float3(1f, 0f, 0f);
            return a;
        }

        float3 bp = p - b;
        float d3 = math.dot(ab, bp);
        float d4 = math.dot(ac, bp);

        // Vertex region B
        if (d3 >= 0f && d4 <= d3)
        {
            bary = new float3(0f, 1f, 0f);
            return b;
        }

        // Edge region AB
        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0f && d1 >= 0f && d3 <= 0f)
        {
            float v = d1 / (d1 - d3); // along AB from A->B
            bary = new float3(1f - v, v, 0f);
            return a + v * ab;
        }

        float3 cp = p - c;
        float d5 = math.dot(ab, cp);
        float d6 = math.dot(ac, cp);

        // Vertex region C
        if (d6 >= 0f && d5 <= d6)
        {
            bary = new float3(0f, 0f, 1f);
            return c;
        }

        // Edge region AC
        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0f && d2 >= 0f && d6 <= 0f)
        {
            float w = d2 / (d2 - d6); // along AC from A->C
            bary = new float3(1f - w, 0f, w);
            return a + w * ac;
        }

        // Edge region BC
        float va = d3 * d6 - d5 * d4;
        if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
        {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); // along BC from B->C
            bary = new float3(0f, 1f - w, w);
            return b + w * (c - b);
        }

        // Face region
        float inv = 1f / (va + vb + vc);
        float v2 = vb * inv;
        float w2 = vc * inv;
        float u2 = 1f - v2 - w2;
        bary = new float3(u2, v2, w2);
        return a + ab * v2 + ac * w2;
    }

    // Ray -> triangle plane hit, clamped to triangle; returns clamped point + barycentric (u,v,w).
    // If ray is degenerate/parallel/behind, falls back to closest point to origin (or origin projected on plane, then clamped).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float3 RayToTriangleClampedBarycentric(
        float3 rayOrigin,
        float3 rayDir,
        float3 a,
        float3 b,
        float3 c,
        out float3 bary,
        float parallelEps = 1e-6f,
        float zeroDirEpsSqr = 1e-12f,
        float degeneracyEpsSqr = 1e-12f)
    {
        // "Ray" with no direction -> point query.
        if (math.lengthsq(rayDir) < zeroDirEpsSqr)
            return ClosestPointOnTriangleBarycentric(rayOrigin, a, b, c, out bary);

        float3 ab = b - a;
        float3 ac = c - a;
        float3 n = math.cross(ab, ac);
        float nSqr = math.lengthsq(n);

        // Degenerate triangle -> return nearest vertex (with bary).
        if (nSqr < degeneracyEpsSqr)
        {
            float da = math.lengthsq(rayOrigin - a);
            float db = math.lengthsq(rayOrigin - b);
            float dc = math.lengthsq(rayOrigin - c);

            if (da <= db && da <= dc) { bary = new float3(1f, 0f, 0f); return a; }
            if (db <= dc)            { bary = new float3(0f, 1f, 0f); return b; }
                                     { bary = new float3(0f, 0f, 1f); return c; }
        }

        float denom = math.dot(n, rayDir);

        // Near-parallel -> project origin onto plane along unit normal, then clamp to triangle.
        if (math.abs(denom) <= parallelEps)
        {
            float invLenN = math.rsqrt(nSqr);
            float3 nUnit = n * invLenN;

            float signedDist = math.dot(nUnit, rayOrigin - a);
            float3 originOnPlane = rayOrigin - signedDist * nUnit;

            return ClosestPointOnTriangleBarycentric(originOnPlane, a, b, c, out bary);
        }

        // Ray-plane: t = dot(n, (a - O)) / dot(n, D)
        float t = math.dot(n, a - rayOrigin) / denom;

        // Behind origin -> stable fallback to closest point to origin.
        if (t < 0f)
            return ClosestPointOnTriangleBarycentric(rayOrigin, a, b, c, out bary);

        float3 planeHit = rayOrigin + rayDir * t;
        return ClosestPointOnTriangleBarycentric(planeHit, a, b, c, out bary);
    }

    /// <summary>
    /// Segment vs triangle. If hit, returns:
    /// - hitPoint: intersection point on the triangle
    /// - slideEnd: endpoint after projecting remaining segment onto the triangle plane
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IntersectSegmentTriangleAndSlide(
        float3 p0, float3 p1,
        float3 a, float3 b, float3 c,
        out float3 hitPoint,
        out float3 slideEnd,
        float epsilon = 1e-6f,
        float degeneracyEpsSqr = 1e-12f)
    {
        hitPoint = default;
        slideEnd = default;

        float3 dir = p1 - p0;

        float3 e1 = b - a;
        float3 e2 = c - a;

        float3 pvec = math.cross(dir, e2);
        float det = math.dot(e1, pvec);

        // Parallel or (nearly) coplanar: treat as "no single intersection". [web:263]
        if (math.abs(det) < epsilon)
            return false;

        float invDet = 1.0f / det;

        float3 tvec = p0 - a;
        float u = math.dot(tvec, pvec) * invDet;
        if (u < 0f || u > 1f)
            return false;

        float3 qvec = math.cross(tvec, e1);
        float v = math.dot(dir, qvec) * invDet;
        if (v < 0f || (u + v) > 1f)
            return false;

        // For segment direction (p1 - p0), t in [0,1] lies along the segment. [web:263]
        float t = math.dot(e2, qvec) * invDet;
        if (t < 0f || t > 1f)
            return false;

        hitPoint = p0 + dir * t;

        // Triangle normal
        float3 n = math.cross(e1, e2);
        float nSqr = math.lengthsq(n);
        if (nSqr < degeneracyEpsSqr)
            return false;

        float3 nUnit = n * math.rsqrt(nSqr); // reciprocal sqrt is provided by Unity.Mathematics. [web:255]

        // Project remaining motion onto plane: v_proj = v - dot(v,n)*n (for unit n). [web:269]
        float3 remaining = p1 - hitPoint;
        float3 remainingProjected = remaining - math.dot(remaining, nUnit) * nUnit;

        slideEnd = hitPoint + remainingProjected;
        return true;
    }
}

#endregion

#region Structs
// General data each soft body
public struct SoftBodyData
{
    public ObjectInstanceManager instanceManager;
    public int verticesStartIndex;

    public SoftBodyData(ObjectInstanceManager instanceManager, int verticesStartIndex)
    {
        this.instanceManager = instanceManager;
        this.verticesStartIndex = verticesStartIndex;
    }
}

// Particle data
public struct Particle
{
    public float3 position, previousPosition, velocity;
    public float invMass;

    public Particle(float3 position, float invMass)
    {
        this.position = position;
        this.invMass = invMass;

        previousPosition = velocity = float3.zero;
    }
}

// Data for the distance constraint
public struct DistanceConstraint
{
    public int2 particlesIdx;
    public float restLength, compliance;

    public DistanceConstraint(DistanceConstraintRef distanceConstraintRef, int particlesOffset, float compliance)
    {
        particlesIdx = distanceConstraintRef.pidx + new int2(particlesOffset, particlesOffset);
        restLength = distanceConstraintRef.restLenght;
        this.compliance = compliance;
    }
}

// Data for the volume constraint
public struct VolumeConstraint
{
    public int4 particlesIdx;
    public float restVolume, compliance;

    public VolumeConstraint(VolumeConstraintRef volumeConstraintRef, int particlesOffset, float compliance)
    {
        particlesIdx = volumeConstraintRef.pidx + new int4(particlesOffset, particlesOffset, particlesOffset, particlesOffset);
        restVolume = volumeConstraintRef.restVolume;
        this.compliance = compliance;
    }
}

public struct CollisionBuffer
{
    public int otherIdx;
    public bool isStatic;
    public float distance;

    public CollisionBuffer(int otherIdx, bool isStatic, float distance)
    {
        this.otherIdx = otherIdx;
        this.isStatic = isStatic;
        this.distance = distance;
    }
}
#endregion

public class JobPhysicsSolver : MonoBehaviour
{
#region Variables
    // Solver settings
    [SerializeField] private int substeps = 5;
    [SerializeField] private float damping = 0;
    [SerializeField] private List<GameObject> staticMeshes;

    // Soft bodies data
    private List<SoftBodyData> softBodies;

    // Particles data
    private NativeList<Particle> particles;

    // Mesh vertices data
    private NativeList<float3> verticesPositions;
    private NativeList<VisualVertex> visualVerticesBary;

    // Constraints data
    private NativeList<int2>[] distanceConGroups;
    private NativeList<DistanceConstraint>[] distanceConstraints;
    private NativeList<VolumeConstraint>[] volumeConstraints;

    // Collision data
    private NativeList<CollisionVertex> collisionVertices;
    private NativeList<int3> collisionTriangles;
    private NativeList<int2> collisionEdges;
    private NativeList<float3> staticColPoints;
    private NativeList<int3> staticColTriangles;
    private NativeList<CollisionBuffer> collisionBuffer;

    // Debug
    private bool initialised = false;
#endregion

#region Simulation
    int particleCount = 0;
    int[] distanceLengths;
    int[] volumeLengths;

    private void FixedUpdate()
    {
        if (!initialised) return;

        JobHandle handle = default;

        // Substep delta time (XPBD requires small timesteps)
        float dt = Time.fixedDeltaTime / substeps;

        particleCount = particles.Length;

        // Cache per-channel list sizes (fixes safety issues when using NativeList.Length)
        if (distanceLengths == null) distanceLengths = volumeLengths = new int[8];
        for (int i = 0; i < 8; i++)
        {
            // Number of distance constraint groups in this solver batch
            distanceLengths[i] = distanceConGroups[i].Length;

            // Number of volumetric constraints (5 entries per tetra constraint)
            volumeLengths[i] = volumeConstraints[i].Length / 5;
        }

        // XPBD multi-step solver
        for (int i = 0; i < substeps; i++)
        {
            // Predict integration step
            UpdatePositions(ref handle, dt);

            // Solve all constraints
            UpdateConstraints(ref handle, dt);

            // Reconstruct velocities after corrections
            UpdateVelocities(ref handle, dt);

            JobHandle.ScheduleBatchedJobs();
        }

        // Copy simulated vertices into meshes
        UpdateVisualMeshes(ref handle);
    }

    private void UpdatePositions(ref JobHandle handle, float dt)
    {
        // Integrate particle positions using semi-implicit Euler
        PositionUpdate positionUpdate = new PositionUpdate
        {
            gravity = Physics.gravity,
            dt = dt,
            particles = particles.AsArray()
        };

        // Parallel integration
        handle = positionUpdate.Schedule(particleCount, 64, handle);
    }

    private void UpdateConstraints(ref JobHandle handle, float dt)
    {
        // Solve distance constraints across 8 solver batches
        for (int i = 0; i < 8; i++)
        {
            DistanceConstraintsUpdate distanceConstraintsUpdate = new DistanceConstraintsUpdate
            {
                dt = dt,
                damping = damping,
                distanceConGroups = distanceConGroups[i].AsArray(),
                distanceConstraints = distanceConstraints[i].AsArray(),
                particles = particles.AsArray()
            };

            // Each range corresponds to multiple constraints
            handle = distanceConstraintsUpdate.Schedule(distanceLengths[i], 1, handle);
        }

        // Solve volume constraints across the 8 solver batches
        for (int i = 0; i < 8; i++)
        {
            VolumeConstraintsUpdate volumeConstraintsUpdate = new VolumeConstraintsUpdate
            {
                dt = dt,
                damping = damping,
                volumeConstraints = volumeConstraints[i].AsArray(),
                particles = particles.AsArray()
            };

            // Each job handles one tetra group (5 entries)
            handle = volumeConstraintsUpdate.Schedule(volumeLengths[i], 1, handle);
        }

        // // Very simple plane collision for starters
        // CrudeCollision crudeCollision = new CrudeCollision
        // {
        //     particles = particles.AsArray(),
        //     collisionVertices = collisionVertices.AsArray()
        // };
        // handle = crudeCollision.Schedule(handle);

        // Complex initial point-triangle collision
        PointTriCollision pointTriCollision = new PointTriCollision
        {
            particles = particles.AsArray(),
            collisionVertices = collisionVertices.AsArray(),
            collisionTriangles = collisionTriangles.AsArray(),
            staticColPoints = staticColPoints.AsArray(),
            staticColTriangles = staticColTriangles.AsArray(),
            collisionBuffer = collisionBuffer
        };
        handle = pointTriCollision.Schedule(handle);
    }

    private void UpdateVelocities(ref JobHandle handle, float dt)
    {
        // Recompute velocities from corrected positional updates
        VelocityUpdate velocityUpdate = new VelocityUpdate
        {
            dt = dt,
            particles = particles.AsArray()
        };

        handle = velocityUpdate.Schedule(particleCount, 10, handle);
    }

    private void UpdateVisualMeshes(ref JobHandle handle)
    {
        // Compute mesh vertices from lattice particles via trilinear interpolation
        VisualVerticesUpdate visualVerticesUpdate = new VisualVerticesUpdate
        {
            particles = particles.AsArray(),
            visualVerticesBary = visualVerticesBary.AsArray(),
            verticesPositions = verticesPositions.AsArray()
        };

        handle = visualVerticesUpdate.Schedule(verticesPositions.Length, 64, handle);

        // Must complete before sending results to the render thread
        handle.Complete();

        // Push updated vertex data into each soft body's mesh instance
        for (int i = 0; i < softBodies.Count; i++)
        {
            softBodies[i].instanceManager.UpdateMesh(verticesPositions.AsArray(),
                                                    softBodies[i].verticesStartIndex);
        }
    }
#endregion

#region Object addition
    public void AddSoftBody(
        ObjectInstanceManager objectManager,
        float bodyMass,
        float bodyDisCompliance,
        float bodyVolCompliance,
        List<float3> bodyParticles,
        List<int2>[] bodyDisConGroups,
        List<DistanceConstraintRef>[] bodyDisConstraints,
        List<VolumeConstraintRef>[] bodyVolConstraints,
        VisualVertex[] bodyVisualVerticesBary,
        CollisionVertex[] bodyCollisionVertices,
        int3[] bodyCollisionTriangles,
        int2[] bodyCollisionEdges)
    {
        if (!initialised) InitialiseSolver();

        // Register new soft body instance
        int verticesOffset = verticesPositions.Length;
        softBodies.Add(new SoftBodyData(objectManager, verticesOffset));
        

        // Add all particles belonging to the body
        int particlesOffset = particles.Length;
        float particleInvMass = 1f / (bodyMass / bodyParticles.Count);

        for (int i = 0; i < bodyParticles.Count; i++)
            particles.Add(new Particle(bodyParticles[i], particleInvMass));
        

        // Add empty vertex entries + trilinear mapping
        for (int i = 0; i < bodyVisualVerticesBary.Length; i++)
        {
            verticesPositions.Add(float3.zero);

            // Remap indices by particle offset
            visualVerticesBary.Add(
                new VisualVertex(
                    bodyVisualVerticesBary[i].tetraIdx +
                    new int4(particlesOffset),
                    bodyVisualVerticesBary[i].baryCoords));
        }


        // Append all constraints into the solver's 8-channel structure
        for (int k = 0; k < 8; k++)
        {
            int disConsOffset = distanceConstraints[k].Length;

            // Add constraint ranges
            for (int i = 0; i < bodyDisConGroups[k].Count; i++)
                distanceConGroups[k].Add(bodyDisConGroups[k][i] + new int2(disConsOffset));

            // Add distance constraints
            for (int i = 0; i < bodyDisConstraints[k].Count; i++)
                distanceConstraints[k].Add(new DistanceConstraint(
                    bodyDisConstraints[k][i], particlesOffset, bodyDisCompliance));

            // Add volume constraints
            for (int i = 0; i < bodyVolConstraints[k].Count; i++)
                volumeConstraints[k].Add(new VolumeConstraint(
                    bodyVolConstraints[k][i], particlesOffset, bodyVolCompliance));
        }


        // Add collision data
        int collisionVerticesOffset = collisionVertices.Length;
        for (int i = 0; i < bodyCollisionVertices.Length; i++)
        {
            // Add a new collision vertex with the offset
            collisionVertices.Add(new CollisionVertex(
                bodyCollisionVertices[i].tetraIdx +
                new int4(particlesOffset),
                bodyCollisionVertices[i].baryCoords,
                bodyCollisionVertices[i].invMass
            ));
        }

        for (int i = 0; i < bodyCollisionTriangles.Length; i++)
        {
            // Add a new triangle with the offset
            collisionTriangles.Add(bodyCollisionTriangles[i] + new int3(collisionVerticesOffset));
        }

        for (int i = 0; i < bodyCollisionEdges.Length; i++)
        {
            // Add a new edge with the offset
            collisionEdges.Add(bodyCollisionEdges[i] + new int2(collisionVerticesOffset));
        }
    }
#endregion

#region Data management
    public void InitialiseSolver()
    {
        initialised = true;

        // List of all soft body instances
        softBodies = new List<SoftBodyData>();

        // Main particle storage
        particles = new NativeList<Particle>(Allocator.Persistent);

        // Storage for mesh vertices & their lattice mapping
        verticesPositions = new NativeList<float3>(Allocator.Persistent);
        visualVerticesBary = new NativeList<VisualVertex>(Allocator.Persistent);

        // Initialise per-channel constraint lists
        distanceConGroups = new NativeList<int2>[8];
        distanceConstraints = new NativeList<DistanceConstraint>[8];
        volumeConstraints = new NativeList<VolumeConstraint>[8];

        for (int i = 0; i < 8; i++)
        {
            distanceConGroups[i] = new NativeList<int2>(Allocator.Persistent);
            distanceConstraints[i] = new NativeList<DistanceConstraint>(Allocator.Persistent);
            volumeConstraints[i] = new NativeList<VolumeConstraint>(Allocator.Persistent);
        }

        // Initialise collision data
        collisionVertices = new NativeList<CollisionVertex>(Allocator.Persistent);
        collisionTriangles = new NativeList<int3>(Allocator.Persistent);
        collisionEdges = new NativeList<int2>(Allocator.Persistent);

        // Initialise static collision data
        staticColPoints = new NativeList<float3>(Allocator.Persistent);
        staticColTriangles = new NativeList<int3>(Allocator.Persistent);
        for (int i = 0; i < staticMeshes.Count; i++)
        {
            Mesh currentStaticMesh = staticMeshes[i].GetComponent<MeshFilter>().mesh;

            int vertexOffset = staticColPoints.Length;
            for (int j = 0; j < currentStaticMesh.vertices.Length; j++)
            {
                staticColPoints.Add(staticMeshes[i].transform.TransformPoint(currentStaticMesh.vertices[j]));
            }

            for (int j = 0; j < currentStaticMesh.triangles.Length; j+=3)
            {
                staticColTriangles.Add(
                    new int3(currentStaticMesh.triangles[j] + vertexOffset,
                    currentStaticMesh.triangles[j+1] + vertexOffset,
                    currentStaticMesh.triangles[j+2] + vertexOffset)
                );
            }
        }
        collisionBuffer = new NativeList<CollisionBuffer>(Allocator.Persistent);
    }

    private void OnDestroy()
    {
        if (!initialised) return;

        // Managed lists
        softBodies = null;

        // Dispose particle data
        particles.Dispose();

        // Dispose vertex data
        verticesPositions.Dispose();
        visualVerticesBary.Dispose();

        // Dispose constraint lists
        for (int i = 0; i < 8; i++)
        {
            distanceConGroups[i].Dispose();
            distanceConstraints[i].Dispose();
            volumeConstraints[i].Dispose();
        }
        distanceConGroups = null;
        distanceConstraints = null;
        volumeConstraints = null;

        // Dispose collision data
        collisionVertices.Dispose();
        collisionTriangles.Dispose();
        collisionEdges.Dispose();
        staticColPoints.Dispose();
        staticColTriangles.Dispose();
        collisionBuffer.Dispose();
    }
#endregion
}

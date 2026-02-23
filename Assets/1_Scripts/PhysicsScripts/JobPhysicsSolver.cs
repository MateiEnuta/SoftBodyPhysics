using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

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
        float3 velocity = (currentParticle.position - currentParticle.previousPosition) / dt;
        velocity += gravity * dt;

        // Save previous position for velocity reconstruction
        currentParticle.previousPosition = currentParticle.position;

        // Integrate new position: x += v * dt
        currentParticle.position += velocity * dt;

        // Write back
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
// COMPLEX COLLISION BETWEEN PARTICLES AND TRIANGLES
// Keeps particles above y=0. Very crude but safe for testing.
// -----------------------------------------------------------------------------
[BurstCompile]
public struct PointTriCollision : IJob
{
    // Dynamic bodies
    public NativeArray<Particle> particles;
    [ReadOnly] public NativeArray<CollisionVertex> collisionVertices;
    [ReadOnly] public NativeArray<int3> collisionTriangles;
    [ReadOnly] public NativeArray<int2> collisionEdges;

    // Static bodies
    [ReadOnly] public NativeArray<float3> staticColPoints;
    [ReadOnly] public NativeArray<int3> staticColTriangles;
    [ReadOnly] public NativeArray<int2> staticCollisionEdges;

#region Main execute
    public void Execute()
    {
        for (int p = 0; p < collisionVertices.Length; p++)
        {
            CollisionVertex cv = collisionVertices[p];
            // Cache the positions for faster acces
            float3 previousPosition = GetPositionsThroughBary(cv, true);
            float3 currentPosition = GetPositionsThroughBary(cv, false);

            for (int t = 0; t < collisionTriangles.Length; t++)
            {
                int3 triIdx = collisionTriangles[t];
                if (triIdx.x == p || triIdx.y == p || triIdx.z == p) continue;
                SolveDynamicCollision(p, previousPosition, ref currentPosition, t);
            }
            
            for (int t = 0; t < staticColTriangles.Length; t++)
            {
                SolveStaticCollision(p, previousPosition, ref currentPosition, t);
            }
        }

        for (int e1 = 0; e1 < collisionEdges.Length; e1++)
        {
            int2 edge1Idx = collisionEdges[e1];
            CollisionVertex cv1E1 = collisionVertices[edge1Idx.x];
            CollisionVertex cv2E1 = collisionVertices[edge1Idx.y];

            // Cache the positions for faster acces
            float3 previous1E1 = GetPositionsThroughBary(cv1E1, true);
            float3 current1E1 = GetPositionsThroughBary(cv1E1, false);
            float3 previous2E1 = GetPositionsThroughBary(cv2E1, true);
            float3 current2E1 = GetPositionsThroughBary(cv2E1, false);

            for (int e2 = e1 + 1; e2 < collisionEdges.Length; e2++)
            {
                int2 edge2Idx = collisionEdges[e2];
                // Avoid casses where edges share a vertex
                if (edge1Idx.x == edge2Idx.x || edge1Idx.x == edge2Idx.y ||
                    edge1Idx.y == edge2Idx.x || edge1Idx.y == edge2Idx.y
                ) continue;

                EECollision(
                    e1, e2, 
                    ref previous1E1, ref current1E1,
                    ref previous2E1, ref current2E1
                );
            }
        }
    }
#endregion

#region Edge test
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EECollision(
        int uIdx, int vIdx,
        ref float3 u1_prev, ref float3 u1_curr,
        ref float3 u2_prev, ref float3 u2_curr
    )
    {
        CollisionVertex v1_cv = collisionVertices[collisionEdges[vIdx].x];
        CollisionVertex v2_cv = collisionVertices[collisionEdges[vIdx].y];

        float3 v1_prev = GetPositionsThroughBary(v1_cv, true);
        float3 v1_curr = GetPositionsThroughBary(v1_cv, false);
        float3 v2_prev = GetPositionsThroughBary(v2_cv, true);
        float3 v2_curr = GetPositionsThroughBary(v2_cv, false);

        // Find closest points to each line at the current positoons
        float3 unclSepparation = LineLineSeparation(u1_curr, u2_curr, v1_curr, v2_curr,
            out float t1, out float t2
        );

        // speculative check
        float3 u = u2_curr - u1_curr;
        float3 v = v2_curr - v1_curr;

        float3 pOnU = u1_curr + t1 * u;
        float3 pOnV = v1_curr + t2 * v;

        // Calculate realtive velocity of the closest points
        float3 velOnU = (1f - t1) * (u1_curr - u1_prev) + t1 * (u2_curr - u2_prev);
        float3 velOnV = (1f - t2) * (v1_curr - v1_prev) + t2 * (v2_curr - v2_prev);
        float3 relInvVelToU = velOnV - velOnU;

        // Clamped separation for cehcking distance between features
        float3 separation = pOnV - pOnU;
        float sepLength = math.length(separation);
        if (sepLength < 1e-08f) return;
        float3 sepDir = separation / sepLength;

        float totalDistance = sepLength - math.dot(relInvVelToU, sepDir);
        // Speculative check, if it fails don't waste resources on CCD
        if (totalDistance > 0.01f) return;

        // Use unclamped separation to let edges slide
        float unclSepLength = math.length(unclSepparation);
        if (unclSepLength < 1e-08f) return;
        sepDir = unclSepparation / unclSepLength;

        // CCD check
        if (!CCDMath.EdgeEdgeCCD_Boolean(
            u1_prev, u2_prev, v1_prev, v2_prev,
            u1_curr, u2_curr, v1_curr, v2_curr
        )) return;

        // Collision response
        CollisionVertex u1_cv = collisionVertices[collisionEdges[uIdx].x];
        CollisionVertex u2_cv = collisionVertices[collisionEdges[uIdx].y];
        float C = -unclSepLength - 0.01f;

        float3 u1_grad = sepDir * (1f - t1);
        float3 u2_grad = sepDir * t1;
        float3 v1_grad = -sepDir * (1f - t2);
        float3 v2_grad = -sepDir * t2;

        float u1_invMass = ColParticleInvMass(u1_cv);
        float u2_invMass = ColParticleInvMass(u2_cv);
        float v1_invMass = ColParticleInvMass(v1_cv);
        float v2_invMass = ColParticleInvMass(v2_cv);

        float sumWeightedGradients = 0;
        sumWeightedGradients += u1_invMass * (1f - t1) * (1f - t1);
        sumWeightedGradients += u2_invMass * t1 * t1;
        sumWeightedGradients += v1_invMass * (1f - t2) * (1f - t2);
        sumWeightedGradients += v2_invMass * t2 * t2;
        if (sumWeightedGradients < 1e-08f) return;

        float deltaLambda = -C / sumWeightedGradients;

        // Update particles and cache the current ones
        UpdateParticlePos(u1_cv, u1_invMass * deltaLambda * u1_grad);
        u1_curr += u1_invMass * deltaLambda * u1_grad;
        UpdateParticlePos(u2_cv, u2_invMass * deltaLambda * u2_grad);
        u2_curr += u2_invMass * deltaLambda * u2_grad;

        UpdateParticlePos(v1_cv, v1_invMass * deltaLambda * v1_grad);
        UpdateParticlePos(v2_cv, v2_invMass * deltaLambda * v2_grad);
    }


#endregion

#region Triangle test
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SolveDynamicCollision(
        int pointIdx, float3 previousP, ref float3 currentP, int triIdx)
    {
        int3 tri = collisionTriangles[triIdx];

        CollisionVertex cvP = collisionVertices[pointIdx];
        CollisionVertex cv0 = collisionVertices[tri.x];
        CollisionVertex cv1 = collisionVertices[tri.y];
        CollisionVertex cv2 = collisionVertices[tri.z];

        float3 A = GetPositionsThroughBary(cv0, true);
        float3 B = GetPositionsThroughBary(cv1, true);
        float3 C = GetPositionsThroughBary(cv2, true);

        float3 cA = GetPositionsThroughBary(cv0, false);
        float3 cB = GetPositionsThroughBary(cv1, false);
        float3 cC = GetPositionsThroughBary(cv2, false);

        // Find the closest point on tri to particle vel
        float3 pVelocityOffser = currentP - previousP;
        float3 hitPoint = ClosestPointOnTriangleBarycentric(
            previousP, A, B, C, out float3 triBary
        );

        // Paritcle to hit point data
        float3 hitOffset = hitPoint - previousP;
        float hitDis = math.length(hitOffset);
        float3 hitDir = hitOffset / hitDis;

        // Triangle data
        float3 triBaryVelOffset = (triBary.x * cA + triBary.y * cB + triBary.z * cC) - 
                                  (triBary.x * A + triBary.y * B + triBary.z * C);
        
        float totalDistance = hitDis - math.dot(hitDir, pVelocityOffser) + math.dot(hitDir, triBaryVelOffset);

        // Discard the collision if speculative fails
        if (totalDistance > 0.01f) return;


        // Collsion response
        float3 triNormal = math.normalize(math.cross(cB - cA, cC - cA));
        float Con = math.dot(currentP - cA, triNormal) - 0.001f;
        if (Con >= 0f) return;
        ClosestPointOnTriangleBarycentric(currentP, cA, cB, cC, out float3 cTriBary, true);

        // Calcualte gradients
        float3 pGrad = triNormal;
        float3 AGrad = -cTriBary.x * triNormal;
        float3 BGrad = -cTriBary.y * triNormal;
        float3 CGrad = -cTriBary.z * triNormal;

        // Store inverse mass
        float invMassP = ColParticleInvMass(cvP);
        float invMass0 = ColParticleInvMass(cv0);
        float invMass1 = ColParticleInvMass(cv1);
        float invMass2 = ColParticleInvMass(cv2);
        
        float sumWeightedGradients = 0;
        sumWeightedGradients += invMassP;
        sumWeightedGradients += invMass0 * (cTriBary.x * cTriBary.x);
        sumWeightedGradients += invMass1 * (cTriBary.y * cTriBary.y);
        sumWeightedGradients += invMass2 * (cTriBary.z * cTriBary.z);
        if (sumWeightedGradients < 1e-08f) return;

        // Calcualte dLambda
        float deltaLambda = -Con / sumWeightedGradients;

        // Apply current particle correction
        UpdateParticlePos(cvP, invMassP * deltaLambda * pGrad);
        currentP += invMassP * deltaLambda * pGrad;

        // Apply correciton to triangle vertices
        UpdateParticlePos(cv0, invMass0 * deltaLambda * AGrad);
        UpdateParticlePos(cv1, invMass1 * deltaLambda * BGrad);
        UpdateParticlePos(cv2, invMass2 * deltaLambda * CGrad);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SolveStaticCollision(int p, float3 previousPosition, ref float3 currentPosition, int triangleIdx)
    {
        // Store static triangle vertices postions
        float3 cT1 = staticColPoints[staticColTriangles[triangleIdx].x];
        float3 cT2 = staticColPoints[staticColTriangles[triangleIdx].y];
        float3 cT3 = staticColPoints[staticColTriangles[triangleIdx].z];

        // Check if the collsion even exists
        float3 triNormal = math.normalize(math.cross(cT2 - cT1, cT3 - cT1));
        float penetration = math.dot(currentPosition - cT1, triNormal) - 0.001f;
        if (penetration >= 0f) return;

        // Find the closest point on tri to particle vel
        float3 pVelocityOffser = currentPosition - previousPosition;
        float3 hitPoint = ClosestPointOnTriangleBarycentric(currentPosition, cT1, cT2, cT3, out float3 triBary);

        // Paritcle to hit point data
        float3 hitOffset = hitPoint - currentPosition;
        float hitDis = math.length(hitOffset);
        if (hitDis <= 0.001f)
        {
            UpdateParticlePos(collisionVertices[p], -penetration * triNormal);
            currentPosition += -penetration * triNormal;
            return;
        }
        float3 hitDir = hitOffset / hitDis;

        float totalDistance = hitDis + math.dot(hitDir, pVelocityOffser);
        // Discard the collision if speculative fails
        if (totalDistance > 0.01f) return;

        UpdateParticlePos(collisionVertices[p], -penetration * triNormal);
        currentPosition += -penetration * triNormal;
    }
#endregion

#region Common functions
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float3 GetPositionsThroughBary(CollisionVertex cv, bool isPrevious)
    {
        float4 bary = cv.baryCoords;
        int4 tetIdx = cv.tetraIdx;

        if (isPrevious) return bary.x * particles[tetIdx.x].previousPosition +
                               bary.y * particles[tetIdx.y].previousPosition +
                               bary.z * particles[tetIdx.z].previousPosition +
                               bary.w * particles[tetIdx.w].previousPosition;

        return bary.x * particles[tetIdx.x].position +
               bary.y * particles[tetIdx.y].position +
               bary.z * particles[tetIdx.z].position +
               bary.w * particles[tetIdx.w].position;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void UpdateParticlePos(CollisionVertex cv, float3 correction)
    {
        // Indices of the tetra particles
        int i0 = cv.tetraIdx.x;
        int i1 = cv.tetraIdx.y;
        int i2 = cv.tetraIdx.z;
        int i3 = cv.tetraIdx.w;

        float4 b = cv.baryCoords;

        // For equalizing the correction
        float denom = b.x*b.x + b.y*b.y + b.z*b.z + b.w*b.w;
        if (denom < 1e-08f) return;

        correction /= denom;
        
        // Update actual particles positions
        Particle p0 = particles[i0];
        p0.position += b.x * correction;
        particles[i0] = p0;

        Particle p1 = particles[i1];
        p1.position += b.y * correction;
        particles[i1] = p1;

        Particle p2 = particles[i2];
        p2.position += b.z * correction;
        particles[i2] = p2;

        Particle p3 = particles[i3];
        p3.position += b.w * correction;
        particles[i3] = p3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float ColParticleInvMass(CollisionVertex cv)
    {
        // Indices of the tetra particles
        int i0 = cv.tetraIdx.x;
        int i1 = cv.tetraIdx.y;
        int i2 = cv.tetraIdx.z;
        int i3 = cv.tetraIdx.w;

        float4 b = cv.baryCoords;

        // Calculate mass
        float mass = 0;
        mass += 1f / particles[i0].invMass * b.x;
        mass += 1f / particles[i1].invMass * b.y;
        mass += 1f / particles[i2].invMass * b.z;
        mass += 1f / particles[i3].invMass * b.w;

        // Inverse the total mass
        return 1f / mass;
    }
#endregion

#region Math functions
    // Returns closest point on triangle ABC to point P, and barycentric coords (u,v,w) sums to 1 for that point.
    // Region-test method consistent with Real-Time Collision Detection.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float3 ClosestPointOnTriangleBarycentric(
        float3 p, float3 a, float3 b, float3 c,
        out float3 bary,
        bool clampToTriangle = true)
    {
        float3 ab = b - a;
        float3 ac = c - a;

        // Degenerate triangle guard
        float3 n = math.cross(ab, ac);
        float n2 = math.lengthsq(n);
        if (n2 < 1e-20f)
        {
            // Triangle is degenerate; pick A as a safe fallback.
            bary = new float3(1f, 0f, 0f);
            return a;
        }

        // Unclamped: barycentric on the supporting plane (can be outside [0,1])
        // This is the standard solve in the (ab, ac) basis. [web:43]
        if (!clampToTriangle)
        {
            float3 ap = p - a;

            float d00 = math.dot(ab, ab);
            float d01 = math.dot(ab, ac);
            float d11 = math.dot(ac, ac);
            float d20 = math.dot(ap, ab);
            float d21 = math.dot(ap, ac);

            float denom = d00 * d11 - d01 * d01;
            if (math.abs(denom) < 1e-20f)
            {
                bary = new float3(1f, 0f, 0f);
                return a;
            }

            float invDenom = 1.0f / denom;
            float v = (d11 * d20 - d01 * d21) * invDenom;
            float w = (d00 * d21 - d01 * d20) * invDenom;
            float u = 1.0f - v - w;

            bary = new float3(u, v, w);
            return a + v * ab + w * ac; // point on the triangle plane, not clamped
        }

        // Clamped: closest point ON triangle (your original Ericson-style region tests). [web:57]
        float3 ap2 = p - a;
        float d1 = math.dot(ab, ap2);
        float d2 = math.dot(ac, ap2);

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
            float v = d1 / (d1 - d3);
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
            float w = d2 / (d2 - d6);
            bary = new float3(1f - w, 0f, w);
            return a + w * ac;
        }

        // Edge region BC
        float va = d3 * d6 - d5 * d4;
        if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
        {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            bary = new float3(0f, 1f - w, w);
            return b + w * (c - b);
        }

        // Face region
        float sum = va + vb + vc;
        if (math.abs(sum) < 1e-20f)
        {
            // Should be rare given the earlier degeneracy check, but safe.
            bary = new float3(1f, 0f, 0f);
            return a;
        }

        float inv = 1f / sum;
        float v2 = vb * inv;
        float w2 = vc * inv;
        float u2 = 1f - v2 - w2;

        bary = new float3(u2, v2, w2);
        return a + ab * v2 + ac * w2;
    }

    // p1,p2 define line 1; p3,p4 define line 2.
    // Returns separation vector between the *clamped* closest points.
    // t1Clamp, t2Clamp are interpolation parameters clamped to [0,1].
    public float3 LineLineSeparation(
        float3 p1, float3 p2,
        float3 p3, float3 p4,
        out float t1Clamp, out float t2Clamp)
    {
        float3 u = p2 - p1;
        float3 v = p4 - p3;
        float3 w0 = p1 - p3;

        float a = math.dot(u, u);
        float b = math.dot(u, v);
        float c = math.dot(v, v);
        float d = math.dot(u, w0);
        float e = math.dot(v, w0);

        float denom = a * c - b * b;

        float s, t;

        // Unclamped parameters for infinite lines
        if (math.abs(denom) < 1e-6f)
        {
            // Nearly parallel: pick one parameter, slide along v
            s = 0f;
            t = (c > 0f) ? (e / c) : 0f;
        }
        else
        {
            s = (b * e - c * d) / denom;
            t = (a * e - b * d) / denom;
        }

        // Clamp to [0,1] to treat them as segments/rays
        t1Clamp = math.clamp(s, 0f, 1f);
        t2Clamp = math.clamp(t, 0f, 1f);

        float3 pOnU = p1 + s * u;
        float3 pOnV = p3 + t * v;

        return pOnV - pOnU;
    }
#endregion
}

#endregion

#region Structs
// General data each soft body
public struct SoftBodyData
{
    public ObjectInstanceManager instanceManager;
    public int verticesStartIndex, particlesStartIndex, particleCount;

    public SoftBodyData(ObjectInstanceManager instanceManager, int verticesStartIndex, 
        int particlesStartIndex, int particleCount)
    {
        this.instanceManager = instanceManager;
        this.verticesStartIndex = verticesStartIndex;
        this.particlesStartIndex = particlesStartIndex;
        this.particleCount = particleCount;
    }
}

// Particle data
public struct Particle
{
    public float3 position, previousPosition;
    public float invMass;

    public Particle(float3 position, float invMass)
    {
        this.position = position;
        this.invMass = invMass;

        previousPosition =  position;
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
    private NativeList<float3> viualVerticesPos;
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
    private NativeList<int2> staticCollisionEdges;

    // Depricated extra data
    // private NativeList<int3> latticeCollisionTriangles;
    // private NativeList<int2> latticeCollisionEdges;

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

            JobHandle.ScheduleBatchedJobs();
        }

        // Copy simulated vertices into meshes
        UpdateVisualMeshes(ref handle);
    }

#region Positions update
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
#endregion

#region Constraints update
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

        // Full initial collision handling
        PointTriCollision pointTriCollision = new PointTriCollision
        {
            particles = particles.AsArray(),
            collisionVertices = collisionVertices.AsArray(),
            collisionTriangles = collisionTriangles.AsArray(),
            collisionEdges = collisionEdges.AsArray(),
            staticColPoints = staticColPoints.AsArray(),
            staticColTriangles = staticColTriangles.AsArray(),
            staticCollisionEdges = staticCollisionEdges.AsArray()
        };
        handle = pointTriCollision.Schedule(handle);
    }
#endregion

#region Visual update
    private void UpdateVisualMeshes(ref JobHandle handle)
    {
        // Compute mesh vertices from lattice particles via trilinear interpolation
        VisualVerticesUpdate visualVerticesUpdate = new VisualVerticesUpdate
        {
            particles = particles.AsArray(),
            visualVerticesBary = visualVerticesBary.AsArray(),
            verticesPositions = viualVerticesPos.AsArray()
        };

        handle = visualVerticesUpdate.Schedule(viualVerticesPos.Length, 64, handle);

        // Must complete before sending results to the render thread
        handle.Complete();

        // Push updated vertex data into each soft body's mesh instance
        for (int i = 0; i < softBodies.Count; i++)
        {
            softBodies[i].instanceManager.UpdateMesh(
                viualVerticesPos, softBodies[i].verticesStartIndex,
                particles, softBodies[i].particlesStartIndex,
                softBodies[i].particleCount
            );
        }
    }
#endregion
#endregion

#region Object addition
    public void AddSoftBody(
        ObjectInstanceManager objectManager,
        float bodyMass, List<float> particleMasses,
        float bodyDisCompliance,
        float bodyVolCompliance,
        List<float3> bodyParticles,
        List<int2>[] bodyDisConGroups,
        List<DistanceConstraintRef>[] bodyDisConstraints,
        List<VolumeConstraintRef>[] bodyVolConstraints,
        VisualVertex[] bodyVisualVerticesBary,
        CollisionVertex[] bodyCollisionVertices,
        int3[] bodyCollisionTriangles,
        int2[] bodyCollisionEdges,
        List<int3> bodyLatticeCollisionTriangles,
        List<int2> bodyLatticeCollisionEdges)
    {
        if (!initialised) InitialiseSolver();

        // Add all particles belonging to the body
        int particlesOffset = particles.Length;
        // float particleInvMass = 1f / (bodyMass / bodyParticles.Count);

        // Register new soft body instance
        int verticesOffset = viualVerticesPos.Length;
        softBodies.Add(new SoftBodyData(objectManager, verticesOffset, particlesOffset, bodyParticles.Count));

        for (int i = 0; i < bodyParticles.Count; i++)
            particles.Add(new Particle(bodyParticles[i], 1f / particleMasses[i]));
        

        // Add empty vertex entries + trilinear mapping
        for (int i = 0; i < bodyVisualVerticesBary.Length; i++)
        {
            viualVerticesPos.Add(float3.zero);

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

        // Add dynamic collision mesh data
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

        // Depricated data
        // for (int i = 0; i < bodyLatticeCollisionTriangles.Count; i++)
        // {
        //     // Add a new lattice triangle with the offset
        //     latticeCollisionTriangles.Add(bodyLatticeCollisionTriangles[i] + new int3(particlesOffset));
        // }

        // for (int i = 0; i < bodyLatticeCollisionEdges.Count; i++)
        // {
        //     // Add a new lattice edge with the offset
        //     latticeCollisionEdges.Add(bodyLatticeCollisionEdges[i] + new int2(particlesOffset));
        // }
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
        viualVerticesPos = new NativeList<float3>(Allocator.Persistent);
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
        staticCollisionEdges = new NativeList<int2>(Allocator.Persistent);
        for (int i = 0; i < staticMeshes.Count; i++)
        {
            Mesh currentStaticMesh = staticMeshes[i].GetComponent<MeshFilter>().mesh;

            // Add static vertices/points
            int vertexOffset = staticColPoints.Length;
            for (int j = 0; j < currentStaticMesh.vertices.Length; j++)
            {
                staticColPoints.Add(staticMeshes[i].transform.TransformPoint(currentStaticMesh.vertices[j]));
            }

            // Add static edges and traingles
            int[] colMeshTris = currentStaticMesh.triangles;
            for (int t = 0; t < colMeshTris.Length / 3; t++)
            {
                // Get the coresponding triangle vertex indexes
                int i0 = colMeshTris[t*3] + vertexOffset;
                int i1 = colMeshTris[t*3+1] + vertexOffset;
                int i2 = colMeshTris[t*3+2] + vertexOffset;

                // Assign triangles
                staticColTriangles.Add(new int3(i0, i1, i2));

                // Adds the edges with ascending order in the hash
                if (i0 < i1) staticCollisionEdges.Add(new int2(i0, i1));
                if (i1 < i2) staticCollisionEdges.Add(new int2(i1, i2));
                if (i2 < i0) staticCollisionEdges.Add(new int2(i2, i0));
            }
        }

        // Initialise extra collision data
        // latticeCollisionTriangles = new NativeList<int3>(Allocator.Persistent);
        // latticeCollisionEdges = new NativeList<int2>(Allocator.Persistent);
    }

    private void OnDestroy()
    {
        if (!initialised) return;

        // Managed lists
        softBodies = null;

        // Dispose particle data
        particles.Dispose();

        // Dispose vertex data
        viualVerticesPos.Dispose();
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
        staticCollisionEdges.Dispose();

        // Extra collision data
        // latticeCollisionTriangles.Dispose();
        // latticeCollisionEdges.Dispose();
    }
#endregion
}

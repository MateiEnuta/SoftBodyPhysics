using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace SoftEngine
{
    /*
    Start of a substep: gravity, then every node moves along its velocity.
    prev keeps where the node was, the rest of the substep only changes pos.
    Nodes with invMass 0 (static bodies) never move.
    */
    [BurstCompile]
    public struct PredictJob : IJobParallelFor
    {
        public float dt;
        public float3 gravity;
        [ReadOnly] public NativeArray<float> invMass;
        public NativeArray<float3> pos, prev, vel;

        public void Execute(int i)
        {
            prev[i] = pos[i];
            if (invMass[i] <= 0f) return;

            vel[i] += gravity * dt;
            pos[i] += vel[i] * dt;
        }
    }

    // End of a substep: the velocity is whatever the node really moved
    [BurstCompile]
    public struct VelocityJob : IJobParallelFor
    {
        public float dt;
        public float damping;
        [ReadOnly] public NativeArray<float3> pos, prev;
        [WriteOnly] public NativeArray<float3> vel;
        [WriteOnly] public NativeArray<float> lastMove;     // how far, for PieceJob

        public void Execute(int i)
        {
            float3 moved = pos[i] - prev[i];
            vel[i] = moved / dt * math.max(1f - damping * dt, 0f);
            lastMove[i] = math.length(moved);
        }
    }

    /*
    Damping, one body per work item, right after the velocities are known.

    A body's nodes lose a part of their velocity, apart from the body moving and
    turning as a whole: that is the motion with the same momentum and the same
    angular momentum as the nodes have now, so neither changes. A damped body falls,
    flies and spins like any other and rests in the same shape, it only stops
    wobbling: after `damping` seconds about a third of a wobble is left.
    */
    [BurstCompile]
    public struct DampJob : IJobParallelFor
    {
        public float dt;
        [ReadOnly] public NativeArray<BodyRange> bodies;
        [ReadOnly] public NativeArray<float> bodyDamping;       // seconds, 0 = none
        [ReadOnly] public NativeArray<float> invMass;
        [ReadOnly] public NativeArray<float3> pos;
        [NativeDisableParallelForRestriction] public NativeArray<float3> vel;

        public void Execute(int body)
        {
            int start = bodies[body].nodeStart, end = start + bodies[body].nodeCount;
            if (bodyDamping[body] <= 0f || invMass[start] <= 0f) return;

            float mass = 0f;
            float3 centre = 0f, drift = 0f;
            for (int i = start; i < end; i++)
            {
                float m = 1f / invMass[i];
                mass += m;
                centre += m * pos[i];
                drift += m * vel[i];
            }
            centre /= mass;
            drift /= mass;

            float3 spin = 0f;               // angular momentum, then the spin that has it
            float3x3 inertia = default;
            for (int i = start; i < end; i++)
            {
                float m = 1f / invMass[i];
                float3 arm = pos[i] - centre;
                spin += m * math.cross(arm, vel[i] - drift);
                inertia.c0 += m * (new float3(math.lengthsq(arm), 0f, 0f) - arm * arm.x);
                inertia.c1 += m * (new float3(0f, math.lengthsq(arm), 0f) - arm * arm.y);
                inertia.c2 += m * (new float3(0f, 0f, math.lengthsq(arm)) - arm * arm.z);
            }
            float size = inertia.c0.x + inertia.c1.y + inertia.c2.z;
            spin = math.determinant(inertia) > 1e-6f * size * size * size ? math.mul(math.inverse(inertia), spin) : 0f;     // a body squashed onto a line has no spin to speak of

            float part = 1f - math.exp(-dt / bodyDamping[body]);
            for (int i = start; i < end; i++)
                vel[i] -= part * (vel[i] - drift - math.cross(spin, pos[i] - centre));
        }
    }

    /*
    Sleeping, once per step: which bodies should be asleep from now on.

    A body remembers where its nodes were (sleepRef). As long as they stay within
    sleepDistance of there on average, and none of them gets further than a few
    times that, the body is going nowhere, however much it trembles (a soft body
    pinched under a heavy one never stops doing that). After sleepTime of it the
    body falls asleep: SoftSolver then treats it like a static body until it is
    woken, which costs next to nothing and stops the trembling for good.

    A body that does get further went somewhere: it remembers the new place, starts
    counting again, and wakes every sleeping body whose box touches the box of where
    it was, where it is now and where it is heading. So whatever is about to hit a
    sleeping body wakes it, and so does whatever it was resting on or leaning
    against when that leaves.
    */
    [BurstCompile]
    public struct SleepJob : IJob
    {
        private const float Tremble = 5f;       // times sleepDistance: how far a single node may get

        public float dt, sleepTime, sleepDistance;
        public float reach;         // boxes this close count as touching
        [ReadOnly] public NativeArray<BodyRange> bodies;
        [ReadOnly] public NativeArray<bool> bodyFixed;
        [ReadOnly] public NativeArray<float3> pos;
        public NativeArray<float3> sleepRef;
        public NativeArray<float> stillTime;
        public NativeArray<bool> bodyAsleep;

        public void Execute()
        {
            int count = bodies.Length;
            NativeArray<float3> lo = new NativeArray<float3>(count, Allocator.Temp), hi = new NativeArray<float3>(count, Allocator.Temp);
            NativeArray<bool> went = new NativeArray<bool>(count, Allocator.Temp);

            for (int b = 0; b < count; b++)
            {
                if (bodyFixed[b]) continue;
                int start = bodies[b].nodeStart, end = start + bodies[b].nodeCount;

                float furthest = 0f, sum = 0f;
                float3 low = float.MaxValue, high = float.MinValue;
                for (int i = start; i < end; i++)
                {
                    float strayed = math.distance(pos[i], sleepRef[i]);
                    furthest = math.max(furthest, strayed);
                    sum += strayed;
                    low = math.min(low, pos[i]);
                    high = math.max(high, pos[i]);
                }
                if (bodyAsleep[b] || (sum <= sleepDistance * (end - start) && furthest <= Tremble * sleepDistance))
                {
                    stillTime[b] += dt;
                    lo[b] = low; hi[b] = high;
                    continue;
                }

                // Went somewhere: the box also covers where it was, and where it will be if it goes on like that
                for (int i = start; i < end; i++)
                {
                    low = math.min(low, math.min(sleepRef[i], 2f * pos[i] - sleepRef[i]));
                    high = math.max(high, math.max(sleepRef[i], 2f * pos[i] - sleepRef[i]));
                    sleepRef[i] = pos[i];
                }
                lo[b] = low; hi[b] = high;
                went[b] = true;
                stillTime[b] = 0f;
            }

            for (int b = 0; b < count; b++)
            {
                if (bodyFixed[b] || went[b]) continue;
                if (!bodyAsleep[b])
                {
                    bodyAsleep[b] = sleepTime > 0f && stillTime[b] >= sleepTime;
                    continue;
                }

                bool wake = sleepTime <= 0f;
                for (int other = 0; !wake && other < count; other++)
                    wake = went[other] && math.all(lo[b] - reach <= hi[other]) && math.all(lo[other] <= hi[b] + reach);
                if (!wake) continue;

                bodyAsleep[b] = false;
                stillTime[b] = 0f;
            }
        }
    }

    /*
    Remembers where the elastic solve left the nodes: `expected` for the rest of the
    substep, and the second pos the contact sweeps work with (ContactSweepJob).
    */
    [BurstCompile]
    public struct CopyJob : IJob
    {
        [ReadOnly] public NativeArray<float3> pos;
        [WriteOnly] public NativeArray<float3> expected, buffer;

        public void Execute()
        {
            expected.CopyFrom(pos);
            buffer.CopyFrom(pos);
        }
    }

    /*
    Elastic solve, one cell per work item, all cells of one colour at a time.

    Cells of one colour never share a node, so they run in parallel, and the fixed
    colour order makes the result the same on every run.

    Every cell is 5 tets. The material is stable neo-Hookean (Smith et al. 2018),
    with F = deformation gradient (identity at rest), I = sum of F's squared entries
    and J = det F:

        energy = mu/2 (I - 3) - mu (J - 1) + (lambda + mu)/2 (J - 1)^2

    As XPBD constraints it is usually split into |F| and J - 1 - mu/lambda (Macklin
    and Mueller 2021). Neither of those is zero at rest: one wants to shrink the tet
    to a point, the other to blow it up, and they only cancel each other. With few
    passes and stiff bodies they do not cancel well enough and the body shakes
    itself apart. Replacing them by |F| - sqrt(3) and J - 1 is calm, but those two
    measure the same thing near rest (volume), so a sheared tet feels nearly nothing.

    The same energy can be grouped into two parts that are each zero at rest:

        CD = sqrt(I + J^2 - 4J)     shape and size      stiffness mu
        CH = J - 1                  volume              stiffness lambda

    (I + J^2 - 4J is never negative, and zero only for a pure rotation.) Sheared,
    stretched, squashed and turned inside out all give CD > 0 with a gradient that
    points back to the rest shape, and to first order this is Hooke's law with
    exactly mu and lambda.

    Both are solved together as one 2x2 XPBD step (they pull on the same 4 nodes, so
    one after the other would undo part of each other).

    In double: at rest CD is the root of a difference of large numbers, in float
    that leaves about 0.0005 of strain as noise, enough to make a stiff body buzz.

    Stretch limit: after the tets, any of the cell's 12 edges longer than maxEdge is
    pulled back to it. A very soft body hit hard would otherwise be drawn out into
    long strings before its own stiffness reacts, and a stretched cell is also a huge
    box for the broad phase. It can still be squashed flat.
    */
    [BurstCompile]
    public struct ElasticJob : IJobParallelFor
    {
        public int first;           // where this colour starts in colourCells
        public int variant;         // tet layout of this colour
        public float invDt2;
        public float bigTetVolume, smallTetVolume;
        public float maxEdge;       // 0 = no limit

        [ReadOnly] public NativeArray<int> colourCells;
        [ReadOnly] public NativeArray<int> cellNodes;
        [ReadOnly] public NativeArray<float> cellMu, cellLambda;
        [ReadOnly] public NativeArray<float3x3> restInverse;    // per variant and tet, inverse of the rest edges
        [ReadOnly] public NativeArray<float> invMass;
        [NativeDisableParallelForRestriction] public NativeArray<float3> pos;

        public void Execute(int index)
        {
            int cell = colourCells[first + index];
            double mu = cellMu[cell], lambda = cellLambda[cell];

            for (int t = 0; t < 5; t++)
            {
                int4 corners = Lattice.Tet(variant, t);
                int4 nodes = new int4(cellNodes[cell * 8 + corners.x], cellNodes[cell * 8 + corners.y],
                                      cellNodes[cell * 8 + corners.z], cellNodes[cell * 8 + corners.w]);
                double volume = t == 0 ? bigTetVolume : smallTetVolume;
                SolveTet(nodes, restInverse[variant * 5 + t], invDt2 / (mu * volume), invDt2 / (lambda * volume));
            }

            for (int k = 0; maxEdge > 0f && k < 8; k++)
            for (int axis = 0; axis < 3; axis++)
            {
                if ((k >> axis & 1) != 0) continue;
                int i = cellNodes[cell * 8 + k], j = cellNodes[cell * 8 + (k | 1 << axis)];
                float3 d = pos[j] - pos[i];
                float length = math.length(d), w = invMass[i] + invMass[j];
                if (length <= maxEdge || w <= 0f) continue;

                float3 pull = d * ((length - maxEdge) / (length * w));
                pos[i] += invMass[i] * pull;
                pos[j] -= invMass[j] * pull;
            }
        }

        private void SolveTet(int4 nodes, double3x3 restInv, double alphaD, double alphaH)
        {
            double3 x0 = pos[nodes.x], x1 = pos[nodes.y], x2 = pos[nodes.z], x3 = pos[nodes.w];
            double w0 = invMass[nodes.x], w1 = invMass[nodes.y], w2 = invMass[nodes.z], w3 = invMass[nodes.w];
            double3x3 restInvT = math.transpose(restInv);

            double3x3 F = math.mul(new double3x3(x1 - x0, x2 - x0, x3 - x0), restInv);
            double3x3 cof = new double3x3(math.cross(F.c1, F.c2), math.cross(F.c2, F.c0), math.cross(F.c0, F.c1));     // gradient of det F
            double J = math.dot(F.c0, cof.c0);

            // Volume
            double cH = J - 1.0;
            double3x3 gH = math.mul(cof, restInvT);     // columns: gradients of nodes 1, 2, 3
            double3 gH0 = -(gH.c0 + gH.c1 + gH.c2);
            double kHH = w0 * math.lengthsq(gH0) + w1 * math.lengthsq(gH.c0) + w2 * math.lengthsq(gH.c1) + w3 * math.lengthsq(gH.c2) + alphaH;

            // Shape: the gradient of I + J^2 - 4J is 2F + (2J - 4) cof
            double cD = math.sqrt(math.max(math.lengthsq(F.c0) + math.lengthsq(F.c1) + math.lengthsq(F.c2) + J * J - 4.0 * J, 0.0));
            double3x3 gD = default;
            double3 gD0 = 0.0;
            double lD = 0.0, lH;

            if (cD > 1e-9)
            {
                double3x3 half = new double3x3(F.c0 + (J - 2.0) * cof.c0, F.c1 + (J - 2.0) * cof.c1, F.c2 + (J - 2.0) * cof.c2);
                gD = math.mul(half * (1.0 / cD), restInvT);
                gD0 = -(gD.c0 + gD.c1 + gD.c2);

                double kDD = w0 * math.lengthsq(gD0) + w1 * math.lengthsq(gD.c0) + w2 * math.lengthsq(gD.c1) + w3 * math.lengthsq(gD.c2) + alphaD;
                double kDH = w0 * math.dot(gD0, gH0) + w1 * math.dot(gD.c0, gH.c0) + w2 * math.dot(gD.c1, gH.c1) + w3 * math.dot(gD.c2, gH.c2);
                double det = kDD * kHH - kDH * kDH;
                if (!math.isfinite(det) || math.abs(det) < 1e-300) return;

                lD = (-cD * kHH + cH * kDH) / det;
                lH = (-cH * kDD + cD * kDH) / det;
            }
            else
            {
                if (!(kHH > 0.0)) return;       // the shape is right, only the volume is left
                lH = -cH / kHH;
            }

            pos[nodes.x] = (float3)(x0 + w0 * (lD * gD0 + lH * gH0));
            pos[nodes.y] = (float3)(x1 + w1 * (lD * gD.c0 + lH * gH.c0));
            pos[nodes.z] = (float3)(x2 + w2 * (lD * gD.c1 + lH * gH.c1));
            pos[nodes.w] = (float3)(x3 + w3 * (lD * gD.c2 + lH * gH.c2));
        }
    }

    // Where every body is (average of its nodes), the visual meshes are drawn around it
    [BurstCompile]
    public struct BodyCentreJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<BodyRange> bodies;
        [ReadOnly] public NativeArray<float3> pos;
        [WriteOnly] public NativeArray<float3> centres;

        public void Execute(int b)
        {
            float3 sum = 0f;
            for (int i = 0; i < bodies[b].nodeCount; i++) sum += pos[bodies[b].nodeStart + i];
            centres[b] = sum / math.max(bodies[b].nodeCount, 1);
        }
    }

    // Visual vertices, relative to their body's centre
    [BurstCompile]
    public struct VisualJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<EmbeddedVertex> visVerts;
        [ReadOnly] public NativeArray<int> visBody;
        [ReadOnly] public NativeArray<float3> centres;
        [ReadOnly] public NativeArray<float3> pos;
        [WriteOnly] public NativeArray<float3> visPos;

        public void Execute(int i)
        {
            visPos[i] = Geometry.Point(visVerts[i], pos) - centres[visBody[i]];
        }
    }

    // What is drawn between two steps: the vertices on their way from where they were to where they are
    [BurstCompile]
    public struct BlendJob : IJobParallelFor
    {
        public float blend;     // 0 = the step before, 1 = now
        [ReadOnly] public NativeArray<float3> before, now;
        [WriteOnly] public NativeArray<float3> drawn;

        public void Execute(int i)
        {
            drawn[i] = math.lerp(before[i], now[i], blend);
        }
    }
}

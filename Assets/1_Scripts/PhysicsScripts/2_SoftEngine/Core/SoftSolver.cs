using System;
using System.Diagnostics;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace SoftEngine
{
    /*
    The whole simulation: all data and the order the jobs run in. No GameObjects here,
    SoftWorld and SoftBody are the Unity side, the test runner uses this class directly.

    One substep:
        1. Predict          gravity and velocity move the nodes (prev -> pos)
        2. Elastic          the cells pull themselves back into shape, 8 colours
        3. Detect           pieces -> pairs -> planes -> contact list, all at prev
        4. Contacts         pushes pos apart where a plane would be crossed, a few sweeps,
                            all bodies at the same time
        5. Limit            the guarantee, takes motion away from nodes if needed
                            (after a second look at the bodies that were pushed further than expected)
        6. Velocity         from what the nodes really did, less what damping takes from a wobble

    After the substeps of a step, bodies that went nowhere for a while fall asleep
    and count as static until something wakes them (SleepJob). A step that runs
    out of time (settings.maxStepMs) does fewer substeps, see Step.

    Everything that goes through many things of the same kind (nodes, cells, pieces,
    pairs, bodies) is a parallel job. What is left on one thread only walks through
    a list once: deciding whether the pair list is still good, putting lists
    together, and the limit step.

    Index spaces:
        node        a lattice corner, the only thing that moves
        cell        8 nodes, cellNodes[cell * 8 + k]
        piece       a convex patch of the surface inside one cell, what collides
        hull point  a corner of a piece, glued into one tet of the piece's cell
        chunk       the pieces of a block of 2 x 2 x 2 cells, for the broad phase
    Bodies own contiguous ranges of all of them.
    */
    public class SoftSolver : IDisposable
    {
        public SoftSettings settings;

        // Nodes
        public NativeList<float3> pos, prev, vel;
        public NativeList<float3> expected;         // where the elastic solve left the nodes this substep
        public NativeList<float> invMass;
        private NativeList<float3> buffer;          // a second pos, for the contact sweeps and the limit step
        private NativeList<float> lastMove, lift, liftGrip, floorGrip;

        // Cells
        public NativeList<int> cellNodes;
        public NativeList<int3> cellCoords;
        public NativeList<int> cellBody;
        public NativeList<float> cellMu, cellLambda;
        public NativeList<int> colourCells;         // moving cells, sorted by colour
        public readonly int[] colourStart = new int[9];
        private NativeArray<float3x3> restInverse;

        // Bodies
        public NativeList<BodyRange> bodies;
        public NativeList<float> bodyFriction, bodyDamping;
        public NativeList<bool> bodyStatic;         // does not move now: static or asleep
        public NativeList<bool> bodyFixed;          // static, never moves
        public NativeList<bool> bodyAsleep;         // see SleepJob
        private NativeList<float> stillTime;
        private NativeList<float3> sleepRef;        // per node
        private NativeList<float> awakeInvMass;     // per node, invMass is 0 while its body sleeps
        public NativeList<float3> bodyCentre;

        // Collision: convex pieces and the hull points that stand for them
        public NativeList<Piece> pieces;
        public NativeList<int2> chunks;             // (first piece, number of pieces), the pieces of a block of cells
        public NativeList<EmbeddedVertex> hullVerts;

        // The full surface of every piece. Not used by the solver, kept to check results against
        public NativeList<EmbeddedVertex> colVerts;
        public NativeList<int3> colTris;

        // Collision, redone every substep
        public NativeList<float3> hullPos, startLo, startHi, centre, boxLo, boxHi;
        public NativeList<float> reach, need;
        public NativeList<bool> escaped, late;
        public NativeList<Contact> contacts;
        private NativeList<int2> latePairs;         // the second detection run of a substep
        private NativeList<PairPlane> latePlanes;
        private NativeList<bool> lateTouching;
        private NativeList<Contact> lateFound;
        private NativeList<int> halves, bodyStart;  // every body's own sides of the contacts, BuildContactsJob
        private NativeList<float> pushed;
        private NativeList<bool> pushedFar;         // per body, ContactSweepJob

        // Collision, kept for as long as every piece stays inside its frame box (PairsJob)
        public NativeList<float3> frameLo, frameHi;
        public NativeList<float> travel;
        public NativeList<int2> pairs;
        public NativeList<PairPlane> planes;
        private NativeList<bool> touching;
        private NativeList<Contact> found;
        private NativeArray<int> builtFor;
        private NativeList<float3> chunkLo, chunkHi, bodyLo, bodyHi;    // the boxes of the broad phase
        private NativeArray<bool> search;
        private float frameTime;
        private bool anyDamping;        // some body has damping, DampJob has something to do

        // Visual meshes
        public NativeList<EmbeddedVertex> visVerts;
        public NativeList<int> visBody;
        public NativeList<float3> visPos;
        public NativeList<float3> visDrawn;         // what BlendVisuals made of it
        private NativeList<float3> visBefore, centreBefore;     // visPos and bodyCentre of the step before
        private bool visualsKnown;                  // false after a body was added, until UpdateVisuals has run
        public bool VisualsKnown => visualsKnown;

        public NativeArray<int> stats;

        // Milliseconds per stage, only filled while profile is on (it waits for every stage, so it is slower)
        public bool profile;
        public readonly double[] stageMs = new double[StageNames.Length];
        public static readonly string[] StageNames = { "predict", "elastic", "pieces", "pairs", "planes", "list", "solve", "late", "limit", "velocity" };


        public SoftSolver(SoftSettings settings)
        {
            this.settings = settings;

            pos = new NativeList<float3>(Allocator.Persistent);
            prev = new NativeList<float3>(Allocator.Persistent);
            vel = new NativeList<float3>(Allocator.Persistent);
            expected = new NativeList<float3>(Allocator.Persistent);
            buffer = new NativeList<float3>(Allocator.Persistent);
            invMass = new NativeList<float>(Allocator.Persistent);
            lastMove = new NativeList<float>(Allocator.Persistent);
            lift = new NativeList<float>(Allocator.Persistent);
            liftGrip = new NativeList<float>(Allocator.Persistent);
            floorGrip = new NativeList<float>(Allocator.Persistent);
            cellNodes = new NativeList<int>(Allocator.Persistent);
            cellCoords = new NativeList<int3>(Allocator.Persistent);
            cellBody = new NativeList<int>(Allocator.Persistent);
            cellMu = new NativeList<float>(Allocator.Persistent);
            cellLambda = new NativeList<float>(Allocator.Persistent);
            colourCells = new NativeList<int>(Allocator.Persistent);
            bodies = new NativeList<BodyRange>(Allocator.Persistent);
            bodyFriction = new NativeList<float>(Allocator.Persistent);
            bodyDamping = new NativeList<float>(Allocator.Persistent);
            bodyStatic = new NativeList<bool>(Allocator.Persistent);
            bodyFixed = new NativeList<bool>(Allocator.Persistent);
            bodyAsleep = new NativeList<bool>(Allocator.Persistent);
            stillTime = new NativeList<float>(Allocator.Persistent);
            sleepRef = new NativeList<float3>(Allocator.Persistent);
            awakeInvMass = new NativeList<float>(Allocator.Persistent);
            bodyCentre = new NativeList<float3>(Allocator.Persistent);
            pieces = new NativeList<Piece>(Allocator.Persistent);
            chunks = new NativeList<int2>(Allocator.Persistent);
            hullVerts = new NativeList<EmbeddedVertex>(Allocator.Persistent);
            colVerts = new NativeList<EmbeddedVertex>(Allocator.Persistent);
            colTris = new NativeList<int3>(Allocator.Persistent);
            hullPos = new NativeList<float3>(Allocator.Persistent);
            frameLo = new NativeList<float3>(Allocator.Persistent);
            frameHi = new NativeList<float3>(Allocator.Persistent);
            escaped = new NativeList<bool>(Allocator.Persistent);
            builtFor = new NativeArray<int>(1, Allocator.Persistent);
            startLo = new NativeList<float3>(Allocator.Persistent);
            startHi = new NativeList<float3>(Allocator.Persistent);
            centre = new NativeList<float3>(Allocator.Persistent);
            boxLo = new NativeList<float3>(Allocator.Persistent);
            boxHi = new NativeList<float3>(Allocator.Persistent);
            reach = new NativeList<float>(Allocator.Persistent);
            need = new NativeList<float>(Allocator.Persistent);
            travel = new NativeList<float>(Allocator.Persistent);
            pairs = new NativeList<int2>(Allocator.Persistent);
            planes = new NativeList<PairPlane>(Allocator.Persistent);
            touching = new NativeList<bool>(Allocator.Persistent);
            found = new NativeList<Contact>(Allocator.Persistent);
            contacts = new NativeList<Contact>(Allocator.Persistent);
            halves = new NativeList<int>(Allocator.Persistent);
            bodyStart = new NativeList<int>(Allocator.Persistent);
            pushed = new NativeList<float>(Allocator.Persistent);
            bodyStart.Add(0);
            pushedFar = new NativeList<bool>(Allocator.Persistent);
            chunkLo = new NativeList<float3>(Allocator.Persistent);
            chunkHi = new NativeList<float3>(Allocator.Persistent);
            bodyLo = new NativeList<float3>(Allocator.Persistent);
            bodyHi = new NativeList<float3>(Allocator.Persistent);
            search = new NativeArray<bool>(1, Allocator.Persistent);
            late = new NativeList<bool>(Allocator.Persistent);
            latePairs = new NativeList<int2>(Allocator.Persistent);
            latePlanes = new NativeList<PairPlane>(Allocator.Persistent);
            lateTouching = new NativeList<bool>(Allocator.Persistent);
            lateFound = new NativeList<Contact>(Allocator.Persistent);
            visVerts = new NativeList<EmbeddedVertex>(Allocator.Persistent);
            visBody = new NativeList<int>(Allocator.Persistent);
            visPos = new NativeList<float3>(Allocator.Persistent);
            visDrawn = new NativeList<float3>(Allocator.Persistent);
            visBefore = new NativeList<float3>(Allocator.Persistent);
            centreBefore = new NativeList<float3>(Allocator.Persistent);
            stats = new NativeArray<int>(SoftStats.Count, Allocator.Persistent);

            // Rest shape of the 5 tets, the same for every cell of a variant
            restInverse = new NativeArray<float3x3>(10, Allocator.Persistent);
            for (int variant = 0; variant < 2; variant++)
            for (int t = 0; t < 5; t++)
            {
                int4 tet = Lattice.Tet(variant, t);
                float3 q0 = Lattice.Corner(tet.x), q1 = Lattice.Corner(tet.y), q2 = Lattice.Corner(tet.z), q3 = Lattice.Corner(tet.w);
                restInverse[variant * 5 + t] = math.inverse(new float3x3(q1 - q0, q2 - q0, q3 - q0) * settings.cellSize);
            }
        }

        public void Dispose()
        {
            pos.Dispose(); prev.Dispose(); vel.Dispose(); expected.Dispose(); buffer.Dispose(); invMass.Dispose();
            lastMove.Dispose(); lift.Dispose(); liftGrip.Dispose(); floorGrip.Dispose();
            cellNodes.Dispose(); cellCoords.Dispose(); cellBody.Dispose(); cellMu.Dispose(); cellLambda.Dispose(); colourCells.Dispose();
            bodies.Dispose(); bodyFriction.Dispose(); bodyDamping.Dispose(); bodyFixed.Dispose(); bodyAsleep.Dispose(); stillTime.Dispose(); sleepRef.Dispose(); awakeInvMass.Dispose(); bodyStatic.Dispose(); bodyCentre.Dispose();
            pieces.Dispose(); chunks.Dispose(); hullVerts.Dispose(); colVerts.Dispose(); colTris.Dispose();
            frameLo.Dispose(); frameHi.Dispose(); escaped.Dispose(); builtFor.Dispose();
            hullPos.Dispose(); startLo.Dispose(); startHi.Dispose(); centre.Dispose(); boxLo.Dispose(); boxHi.Dispose(); reach.Dispose(); need.Dispose(); travel.Dispose();
            pairs.Dispose(); planes.Dispose(); touching.Dispose(); found.Dispose(); contacts.Dispose(); halves.Dispose(); bodyStart.Dispose(); pushed.Dispose();
            pushedFar.Dispose(); chunkLo.Dispose(); chunkHi.Dispose(); bodyLo.Dispose(); bodyHi.Dispose(); search.Dispose();
            late.Dispose(); latePairs.Dispose(); latePlanes.Dispose(); lateTouching.Dispose(); lateFound.Dispose();
            visVerts.Dispose(); visBody.Dispose(); visPos.Dispose(); visDrawn.Dispose(); visBefore.Dispose(); centreBefore.Dispose();
            stats.Dispose(); restInverse.Dispose();
        }

        /*
        Adds a body built from a mesh placed by localToWorld. Returns its index.
        A static body never moves, other bodies collide with it like with any body.
        damping: seconds, how quickly the body stops wobbling (see DampJob), 0 = never.
        */
        public int AddBody(Vector3[] meshVerts, int[] meshTris, Matrix4x4 localToWorld,
            float mass, float youngsModulus, float poissonRatio, float friction, bool isStatic, float3 velocity, float damping = 0f)
        {
            LatticeBuilder built = new LatticeBuilder(meshVerts, meshTris, localToWorld, settings.cellSize, mass);
            int body = bodies.Length;
            BodyRange range = new BodyRange
            {
                nodeStart = pos.Length, nodeCount = built.nodePositions.Length,
                cellStart = cellCoords.Length, cellCount = built.cellCoords.Length,
                pieceStart = pieces.Length, pieceCount = built.pieces.Count,
                chunkStart = chunks.Length, chunkCount = built.chunks.Count,
                visStart = visVerts.Length, visCount = built.visVerts.Length
            };

            for (int i = 0; i < built.nodePositions.Length; i++)
            {
                pos.Add(built.nodePositions[i]);
                prev.Add(built.nodePositions[i]);
                expected.Add(built.nodePositions[i]);
                buffer.Add(built.nodePositions[i]);
                vel.Add(isStatic ? float3.zero : velocity);
                invMass.Add(isStatic ? 0f : 1f / built.nodeMass[i]);
                awakeInvMass.Add(isStatic ? 0f : 1f / built.nodeMass[i]);
                sleepRef.Add(built.nodePositions[i]);
                lastMove.Add(0f); lift.Add(0f); liftGrip.Add(0f); floorGrip.Add(0f);
            }

            float mu = youngsModulus / (2f * (1f + poissonRatio));
            float lambda = youngsModulus * poissonRatio / ((1f + poissonRatio) * (1f - 2f * poissonRatio));
            for (int i = 0; i < built.cellCoords.Length; i++)
            {
                cellCoords.Add(built.cellCoords[i]);
                cellBody.Add(body);
                cellMu.Add(mu);
                cellLambda.Add(lambda);
            }
            for (int i = 0; i < built.cellNodes.Length; i++) cellNodes.Add(built.cellNodes[i] + range.nodeStart);

            int vertOffset = colVerts.Length, triOffset = colTris.Length, hullOffset = hullVerts.Length;
            foreach (Piece piece in built.pieces)
            {
                Piece shifted = piece;
                shifted.body = body;
                shifted.cell += range.cellStart;
                shifted.hullStart += hullOffset;
                shifted.vertStart += vertOffset;
                shifted.triStart += triOffset;
                pieces.Add(shifted);
            }
            foreach (int2 chunk in built.chunks) chunks.Add(new int2(chunk.x + range.pieceStart, chunk.y));
            foreach (EmbeddedVertex v in built.hullVerts) hullVerts.Add(new EmbeddedVertex { nodes = v.nodes + range.nodeStart, w = v.w });
            foreach (EmbeddedVertex v in built.colVerts) colVerts.Add(new EmbeddedVertex { nodes = v.nodes + range.nodeStart, w = v.w });
            foreach (int3 tri in built.colTris) colTris.Add(tri + vertOffset);
            foreach (EmbeddedVertex v in built.visVerts)
            {
                visVerts.Add(new EmbeddedVertex { nodes = v.nodes + range.nodeStart, w = v.w });
                visBody.Add(body);
            }

            bodies.Add(range);
            bodyFriction.Add(friction);
            bodyDamping.Add(damping);
            anyDamping |= damping > 0f && !isStatic;
            bodyStatic.Add(isStatic);
            bodyFixed.Add(isStatic);
            bodyAsleep.Add(false);
            stillTime.Add(0f);
            bodyCentre.Add(float3.zero);
            centreBefore.Add(float3.zero);
            visualsKnown = false;
            bodyStart.Add(0);
            pushedFar.Add(false);
            bodyLo.Add(float3.zero);
            bodyHi.Add(float3.zero);
            chunkLo.ResizeUninitialized(chunks.Length);
            chunkHi.ResizeUninitialized(chunks.Length);

            hullPos.ResizeUninitialized(hullVerts.Length);
            visPos.ResizeUninitialized(visVerts.Length);
            visDrawn.ResizeUninitialized(visVerts.Length);
            visBefore.ResizeUninitialized(visVerts.Length);
            frameLo.ResizeUninitialized(pieces.Length);
            frameHi.ResizeUninitialized(pieces.Length);
            escaped.ResizeUninitialized(pieces.Length);
            startLo.ResizeUninitialized(pieces.Length);
            startHi.ResizeUninitialized(pieces.Length);
            centre.ResizeUninitialized(pieces.Length);
            boxLo.ResizeUninitialized(pieces.Length);
            boxHi.ResizeUninitialized(pieces.Length);
            reach.ResizeUninitialized(pieces.Length);
            need.ResizeUninitialized(pieces.Length);
            late.Resize(pieces.Length, NativeArrayOptions.ClearMemory);
            travel.Resize(pieces.Length, NativeArrayOptions.ClearMemory);

            SortCellsByColour();
            return body;
        }

        // colourCells = all cells of moving bodies, colour 0 first. Counting sort, keeps cell order inside a colour
        private void SortCellsByColour()
        {
            for (int c = 0; c <= 8; c++) colourStart[c] = 0;
            for (int cell = 0; cell < cellCoords.Length; cell++)
                if (!bodyStatic[cellBody[cell]]) colourStart[Lattice.Colour(cellCoords[cell]) + 1]++;
            for (int c = 0; c < 8; c++) colourStart[c + 1] += colourStart[c];

            colourCells.ResizeUninitialized(colourStart[8]);
            int[] cursor = new int[8];
            for (int cell = 0; cell < cellCoords.Length; cell++)
            {
                if (bodyStatic[cellBody[cell]]) continue;
                int colour = Lattice.Colour(cellCoords[cell]);
                colourCells[colourStart[colour] + cursor[colour]++] = cell;
            }
        }

        // How much time the last Step really simulated, less than it was asked for when settings.maxStepMs cut it short
        public float Simulated { get; private set; }

        /*
        Advances the simulation by dt, in settings.substeps equal substeps.

        With settings.maxStepMs set, a step that would take longer than that stops
        early: it does fewer substeps, each as long and as careful as always, so
        the simulation runs slower than real time for a moment (slow motion)
        instead of holding up the frame. One substep is always done.

        Before it comes to that, substeps that take more than their share save time
        on the contacts (Hurry): one sweep, and the limit step stops and freezes
        right away instead of sliding first. Nothing gets through either way, but
        bodies that are squeezed hard stick to each other while it lasts.

        Time is read from the clock, or with settings.repeatable counted (see Work).
        */
        public void Step(float dt)
        {
            Simulated = 0f;
            if (pos.Length == 0) return;

            frameTime = dt;
            Stopwatch clock = Stopwatch.StartNew();
            double spent = 0.0;
            for (int s = 0; s < settings.substeps; s++)
            {
                // Is there time for one more like the one before?
                if (s > 0 && settings.maxStepMs > 0f && spent + substepMs > settings.maxStepMs) break;

                double started = clock.Elapsed.TotalMilliseconds;
                int pairsBefore = stats[SoftStats.Pairs], contactsBefore = stats[SoftStats.PieceContacts], cutsBefore = stats[SoftStats.Cuts];
                Substep(dt / settings.substeps);
                Simulated += dt / settings.substeps;

                substepMs = settings.repeatable
                    ? Work(stats[SoftStats.Pairs] - pairsBefore, stats[SoftStats.PieceContacts] - contactsBefore, stats[SoftStats.Cuts] - cutsBefore)
                    : clock.Elapsed.TotalMilliseconds - started;
                spent += substepMs;
            }
            Sleep(Simulated);
        }

        private double substepMs;       // how long the substep before this one took

        /*
        settings.repeatable: how long a substep took, counted instead of measured.

        A clock never says the same thing twice, so with a limit that reads the
        clock no two runs are the same. This counts the work that was done, and the
        same work always gives the same number, on every run and on every machine.
        The price: it is an estimate (milliseconds per thing, measured once on an
        Apple M1 with Burst safety checks off, within about a tenth of the real
        time there), so on a slower machine the step takes longer than maxStepMs
        says instead of the simulation slowing down more.
        */
        private double Work(int pairs, int contacts, int cuts)
        {
            const double Substep = 0.16, Cell = 0.00011, Pair = 0.000035, Contact = 0.00054, Cut = 0.0014;
            return Substep + Cell * colourStart[8] + Pair * pairs + Contact * contacts + Cut * cuts;      // colourStart[8]: the cells that are not static or asleep
        }

        // The substep before took more than its share of maxStepMs: the contacts are handled the quick way until that is over
        private bool Hurry => settings.maxStepMs > 0f && substepMs * settings.substeps > settings.maxStepMs;

        // Puts bodies to sleep and wakes them, see SleepJob. A sleeping body is a static body for as long as it sleeps
        private void Sleep(float dt)
        {
            new SleepJob
            {
                dt = dt, sleepTime = settings.sleepTime, sleepDistance = settings.sleepDistance, reach = 2f * settings.skin,
                bodies = bodies.AsArray(), bodyFixed = bodyFixed.AsArray(), pos = pos.AsArray(),
                sleepRef = sleepRef.AsArray(), stillTime = stillTime.AsArray(), bodyAsleep = bodyAsleep.AsArray()
            }.Run();

            bool changed = false;
            for (int body = 0; body < bodies.Length; body++)
            {
                bool still = bodyFixed[body] || bodyAsleep[body];
                if (bodyAsleep[body]) stats[SoftStats.Asleep]++;
                if (bodyStatic[body] == still) continue;

                changed = true;
                stats[SoftStats.SleepChanges]++;
                bodyStatic[body] = still;
                for (int i = bodies[body].nodeStart; i < bodies[body].nodeStart + bodies[body].nodeCount; i++)
                {
                    invMass[i] = still ? 0f : awakeInvMass[i];
                    vel[i] = float3.zero;
                }
            }
            if (!changed) return;

            SortCellsByColour();
            builtFor[0] = -1;       // the pair list leaves out pairs of two bodies that do not move, so it has to be made again
        }

        /*
        One stage after the other, each one finished before the next starts: the jobs
        that work on many things at once are spread over all cores and waited for, the
        small ones that only walk through a list run right here.
        */
        private void Substep(float dt)
        {
            Stopwatch clock = profile ? Stopwatch.StartNew() : null;
            float cellVolume = settings.cellSize * settings.cellSize * settings.cellSize;

            // 1. Predict
            new PredictJob
            {
                dt = dt, gravity = settings.gravity, invMass = invMass.AsArray(),
                pos = pos.AsArray(), prev = prev.AsArray(), vel = vel.AsArray()
            }.Schedule(pos.Length, 256).Complete();
            Lap(clock, 0);

            // 2. Elastic
            JobHandle handle = default;
            for (int colour = 0; colour < 8; colour++)
            {
                handle = new ElasticJob
                {
                    first = colourStart[colour], variant = math.countbits(colour) & 1,
                    invDt2 = 1f / (dt * dt), bigTetVolume = cellVolume / 3f, smallTetVolume = cellVolume / 6f, maxEdge = settings.maxStretch * settings.cellSize,
                    colourCells = colourCells.AsArray(), cellNodes = cellNodes.AsArray(),
                    cellMu = cellMu.AsArray(), cellLambda = cellLambda.AsArray(), restInverse = restInverse,
                    invMass = invMass.AsArray(), pos = pos.AsArray()
                }.Schedule(colourStart[colour + 1] - colourStart[colour], 16, handle);
            }
            handle.Complete();
            new CopyJob { pos = pos.AsArray(), expected = expected.AsArray(), buffer = buffer.AsArray() }.Run();
            Lap(clock, 1);

            // 3. Detect
            new PieceJob
            {
                skin = settings.skin, regionMargin = settings.regionMargin,
                pieces = pieces.AsArray(), hullVerts = hullVerts.AsArray(), cellNodes = cellNodes.AsArray(),
                pos = pos.AsArray(), prev = prev.AsArray(), lastMove = lastMove.AsArray(), hullPos = hullPos.AsArray(),
                startLo = startLo.AsArray(), startHi = startHi.AsArray(), centre = centre.AsArray(),
                boxLo = boxLo.AsArray(), boxHi = boxHi.AsArray(), reach = reach.AsArray(), need = need.AsArray(), travel = travel.AsArray(),
                frameLo = frameLo.AsArray(), frameHi = frameHi.AsArray(), escaped = escaped.AsArray(), late = late.AsArray()
            }.Schedule(pieces.Length, 16).Complete();
            Lap(clock, 2);
            Detect(dt, pairs, planes, touching, found, false, clock);

            // 4. Contacts. The sweeps go back and forth between pos and the buffer (both hold the same now), the last one writes to pos
            bool hurry = Hurry;
            int sweeps = hurry ? 1 : math.max(settings.contactSweeps, 1);
            handle = default;
            for (int sweep = 0; sweep < sweeps; sweep++)
            {
                bool toPos = (sweeps - 1 - sweep) % 2 == 0;
                handle = new ContactSweepJob
                {
                    sweep = sweep, last = sweep == sweeps - 1, forward = toPos, skin = settings.skin, pushOut = settings.pushOutSpeed * dt, regionMargin = settings.regionMargin,
                    contacts = contacts.AsArray(), halves = halves.AsArray(), bodyStart = bodyStart.AsArray(),
                    bodies = bodies.AsArray(), pieces = pieces.AsArray(), hullVerts = hullVerts.AsArray(), cellNodes = cellNodes.AsArray(),
                    prev = prev.AsArray(), expected = expected.AsArray(), hullPos = hullPos.AsArray(),
                    from = toPos ? buffer.AsArray() : pos.AsArray(), to = toPos ? pos.AsArray() : buffer.AsArray(),
                    pushed = pushed.AsArray(), lift = lift.AsArray(), liftGrip = liftGrip.AsArray(), floorGrip = floorGrip.AsArray(), pushedFar = pushedFar.AsArray()
                }.Schedule(bodies.Length, 1, handle);
            }
            handle.Complete();
            Lap(clock, 6);

            // 5. Limit, after a second look at the pieces that were pushed further than expected
            if (pushedFar.AsArray().Contains(true))
            {
                new LatePieceJob
                {
                    skin = settings.skin, regionMargin = settings.regionMargin, pushedFar = pushedFar.AsArray(), bodies = bodies.AsArray(),
                    pieces = pieces.AsArray(), cellNodes = cellNodes.AsArray(), pos = pos.AsArray(), prev = prev.AsArray(), expected = expected.AsArray(),
                    frameLo = frameLo.AsArray(), frameHi = frameHi.AsArray(), travel = travel.AsArray(),
                    boxLo = boxLo.AsArray(), boxHi = boxHi.AsArray(), reach = reach.AsArray(), need = need.AsArray(), late = late.AsArray(), escaped = escaped.AsArray()
                }.Schedule(bodies.Length, 1).Complete();
                Detect(dt, latePairs, latePlanes, lateTouching, lateFound, true, null);
            }
            Lap(clock, 7);
            new LimitJob
            {
                hardSkin = settings.hardSkin, slidePasses = hurry ? 0 : 4, stopPasses = hurry ? 1 : 6, bodyCount = bodies.Length, contacts = contacts, pieces = pieces.AsArray(), hullVerts = hullVerts.AsArray(),
                cellNodes = cellNodes.AsArray(), prev = prev.AsArray(), pos = pos.AsArray(), goal = buffer.AsArray(), stats = stats
            }.Run();
            Lap(clock, 8);

            // 6. Velocity
            new VelocityJob
            {
                dt = dt, damping = settings.damping, pos = pos.AsArray(), prev = prev.AsArray(), vel = vel.AsArray(), lastMove = lastMove.AsArray()
            }.Schedule(pos.Length, 256).Complete();
            if (anyDamping)
                new DampJob
                {
                    dt = dt, bodies = bodies.AsArray(), bodyDamping = bodyDamping.AsArray(), invMass = invMass.AsArray(), pos = pos.AsArray(), vel = vel.AsArray()
                }.Schedule(bodies.Length, 1).Complete();
            Lap(clock, 9);
        }

        // Pairs -> planes -> contacts. The second run of a substep (onlyLate) adds to the contact list, see LatePieceJob
        private void Detect(float dt, NativeList<int2> pairList, NativeList<PairPlane> planeList, NativeList<bool> touchList, NativeList<Contact> foundList, bool onlyLate, Stopwatch clock)
        {
            new PairsJob
            {
                horizon = frameTime, grow = settings.regionMargin + 0.5f * settings.skin, slack = settings.regionMargin,
                onlyLate = onlyLate, escaped = escaped.AsArray(), late = late.AsArray(), keptPairs = onlyLate ? pairs : latePairs,
                pieces = pieces.AsArray(), bodies = bodies.AsArray(), chunks = chunks.AsArray(), cellNodes = cellNodes.AsArray(),
                pos = pos.AsArray(), vel = vel.AsArray(), boxLo = boxLo.AsArray(), boxHi = boxHi.AsArray(),
                frameLo = frameLo.AsArray(), frameHi = frameHi.AsArray(), chunkLo = chunkLo.AsArray(), chunkHi = chunkHi.AsArray(),
                bodyLo = bodyLo.AsArray(), bodyHi = bodyHi.AsArray(), builtFor = builtFor, search = search,
                pairs = pairList, planes = planeList, touching = touchList, found = foundList, stats = stats
            }.Run();
            if (search[0])
            {
                NativeStream bodyPairs = new NativeStream(bodies.Length, Allocator.TempJob);
                new PairSearchJob
                {
                    selfCollision = settings.selfCollision, onlyLate = onlyLate, late = late.AsArray(),
                    pieces = pieces.AsArray(), bodies = bodies.AsArray(), chunks = chunks.AsArray(), cellCoords = cellCoords.AsArray(), bodyStatic = bodyStatic.AsArray(),
                    boxLo = boxLo.AsArray(), boxHi = boxHi.AsArray(), frameLo = frameLo.AsArray(), frameHi = frameHi.AsArray(),
                    chunkLo = chunkLo.AsArray(), chunkHi = chunkHi.AsArray(), bodyLo = bodyLo.AsArray(), bodyHi = bodyHi.AsArray(), bodyPairs = bodyPairs.AsWriter()
                }.Schedule(bodies.Length, 1).Complete();
                new PairListJob
                {
                    onlyLate = onlyLate, bodyPairs = bodyPairs.AsReader(), travel = travel.AsArray(), need = need.AsArray(),
                    pairs = pairList, planes = planeList, touching = touchList, found = foundList, stats = stats
                }.Run();
                bodyPairs.Dispose();
            }
            Lap(clock, 3);

            new PlaneJob
            {
                skin = settings.skin, hardSkin = settings.hardSkin, pushOut = settings.pushOutSpeed * dt, cellSize = settings.cellSize,
                margin = onlyLate ? 0f : settings.regionMargin, pairs = pairList.AsArray(), pieces = pieces.AsArray(),
                hullPos = hullPos.AsArray(), startLo = startLo.AsArray(), startHi = startHi.AsArray(), centre = centre.AsArray(),
                reach = reach.AsArray(), travel = travel.AsArray(), need = need.AsArray(), cellNodes = cellNodes.AsArray(),
                invMass = invMass.AsArray(), bodyFriction = bodyFriction.AsArray(), pos = pos.AsArray(), prev = prev.AsArray(),
                planes = planeList.AsArray(), touching = touchList.AsArray(), found = foundList.AsArray()
            }.Schedule(pairList.Length, 32).Complete();
            Lap(clock, 4);

            new BuildContactsJob
            {
                skin = settings.skin, hasFloor = settings.hasFloor, floorY = settings.floorY, floorFriction = settings.floorFriction,
                onlyLate = onlyLate, late = late.AsArray(), touching = touchList, found = foundList, pieces = pieces.AsArray(), cellNodes = cellNodes.AsArray(),
                bodyFriction = bodyFriction.AsArray(), bodyStatic = bodyStatic.AsArray(),
                startLo = startLo.AsArray(), pos = pos.AsArray(), prev = prev.AsArray(), reach = reach.AsArray(),
                contacts = contacts, halves = halves, pushed = pushed, bodyStart = bodyStart.AsArray(), stats = stats
            }.Run();
            Lap(clock, 5);
        }

        private void Lap(Stopwatch clock, int stage)
        {
            if (clock == null) return;
            stageMs[stage] += clock.Elapsed.TotalMilliseconds;
            clock.Restart();
        }

        // Fills bodyCentre and visPos (visual vertices relative to their body's centre), and keeps what they were before
        public void UpdateVisuals()
        {
            if (visualsKnown) KeepVisuals();

            JobHandle handle = new BodyCentreJob
            {
                bodies = bodies.AsArray(), pos = pos.AsArray(), centres = bodyCentre.AsArray()
            }.Schedule(bodies.Length, 1);
            handle = new VisualJob
            {
                visVerts = visVerts.AsArray(), visBody = visBody.AsArray(), centres = bodyCentre.AsArray(),
                pos = pos.AsArray(), visPos = visPos.AsArray()
            }.Schedule(visVerts.Length, 256, handle);
            handle.Complete();

            if (!visualsKnown) KeepVisuals();       // the first time there is no before
            visualsKnown = true;
        }

        private void KeepVisuals()
        {
            visBefore.AsArray().CopyFrom(visPos.AsArray());
            centreBefore.AsArray().CopyFrom(bodyCentre.AsArray());
        }

        // Fills visDrawn: the visual vertices between the step before (blend 0) and the last one (blend 1)
        public void BlendVisuals(float blend)
        {
            new BlendJob
            {
                blend = blend, before = visBefore.AsArray(), now = visPos.AsArray(), drawn = visDrawn.AsArray()
            }.Schedule(visPos.Length, 1024).Complete();
        }

        // Where a body is drawn, see BlendVisuals
        public Vector3 BlendCentre(int body, float blend)
        {
            return math.lerp(centreBefore[body], bodyCentre[body], blend);
        }
    }
}

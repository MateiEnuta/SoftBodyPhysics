using System;
using Unity.Mathematics;

namespace SoftEngine
{
    /*
    Everything that can be tuned, in one place. SoftWorld shows it in the Inspector,
    the test runner overrides single fields by name.
    */
    [Serializable]
    public struct SoftSettings
    {
        public float cellSize;          // lattice cell edge, metres
        public int substeps;            // per FixedUpdate
        public float maxStepMs;         // a step that would take longer does fewer substeps: slow motion instead of a long frame. 0 = no limit
        public bool repeatable;         // maxStepMs counts the work done instead of reading the clock: every run is the same, but the limit is an estimate
        public float3 gravity;
        public float damping;           // velocity lost per second, 0 = none
        public float maxStretch;        // a cell edge never gets longer than this many times its rest length, 0 = no limit

        public float skin;              // contacts keep the pieces this far apart
        public float hardSkin;          // the limit step never lets them get closer than this (the surface itself: LatticeBuilder.HullError closer per side at most)
        public int contactSweeps;       // times the contact solve goes through the contacts
        public float pushOutSpeed;      // the fastest meshes closer than skin are moved apart, m/s
        public float regionMargin;      // how far contacts are expected to push a node off its path at most, more costs a second detection run
        public bool selfCollision;      // cells of one body that are not neighbours collide too

        public float sleepTime;         // a body that went nowhere for this many seconds falls asleep, 0 = never
        public float sleepDistance;     // "nowhere": none of its nodes got further than this from where it was

        public bool hasFloor;
        public float floorY;
        public float floorFriction;

        public static SoftSettings Default => new SoftSettings
        {
            cellSize = 0.25f,
            substeps = 4,
            maxStepMs = 0f,
            repeatable = true,
            gravity = new float3(0f, -9.81f, 0f),
            damping = 0f,
            maxStretch = 2f,
            skin = 0.025f,
            hardSkin = 0.0125f,
            contactSweeps = 3,
            pushOutSpeed = 0.2f,
            regionMargin = 0.025f,
            selfCollision = false,
            sleepTime = 0.5f,
            sleepDistance = 0.01f,
            hasFloor = true,
            floorY = 0f,
            floorFriction = 0.5f
        };
    }

    /*
    A point glued into the lattice: position = sum of w[k] * node[nodes[k]].
    The 4 nodes are one tet of a cell and the weights are barycentric (>= 0, sum 1),
    so the point always stays inside the tet. Hull points, surface and visual vertices use it.
    */
    public struct EmbeddedVertex
    {
        public int4 nodes;
        public float4 w;
    }

    /*
    A convex patch of a body's surface inside one lattice cell (a cell whose surface
    bends inwards holds several).

    What collides is the convex hull of a few of its vertices, the hull points:
    enough of them that no vertex of the patch sticks out of their hull by more than
    LatticeBuilder.HullError, also when the cell is deformed. The full patch
    (vertStart, triStart) is kept as well, the solver itself never reads it, it is
    there to check results against.
    */
    public struct Piece
    {
        public int cell;
        public int body;
        public int hullStart, hullCount;
        public int vertStart, vertCount;
        public int triStart, triCount;
    }

    public struct BodyRange
    {
        public int nodeStart, nodeCount;
        public int cellStart, cellCount;
        public int pieceStart, pieceCount;
        public int chunkStart, chunkCount;
        public int visStart, visCount;
    }

    public enum ContactKind : byte
    {
        Pieces,     // the hull points of piece a against the hull points of piece b
        Floor       // the hull points of piece a against a fixed plane
    }

    /*
    One plane that keeps two sets of hull points apart for the whole substep, fixed at its start.

        n        unit, points from side B to side A
        slab     (lowest A point along n) - (highest B point along n) at the start

    If A's points end above B's along n, the two hulls were apart all the way.
    slab can be zero or negative: the hulls touch or overlap (bodies created inside
    each other). The plane is then the best direction to get them apart again.

    Pieces      the solver pushes at one spot per side, the middle of where the two
                pieces face each other at the start. A spot is a fixed place in its
                cell, given by the share each of the cell's 8 nodes has in it.
                firm: for pieces that are heading for the limit. Measured at the
                hull points, and the spot is the middle of the cell.
                idle: nothing but a, b, n and slab is filled in.
    Floor       B is the plane n.x = planeD.
    */
    public struct Contact
    {
        public ContactKind kind;
        public int a, b;                // pieces
        public double3 n;
        public float slab;
        public float planeD;
        public float friction;
        public bool idle;               // heading far enough apart that there is nothing to push
        public bool firm;

        // Pieces only
        public float target;            // how far apart along n the two spots have to end
        public float4 a0, a1, b0, b1;   // shares of nodes 0-3 and 4-7 of cell A and of cell B in their spot
        public float spreadA, spreadB;  // sum of the squared shares
        public float shareA;            // the part of a push that side A takes (by mass), B takes the rest
    }

    /*
    What is kept for every pair of the broad phase, see PlaneJob.

        n           the direction that kept the two pieces furthest apart when it was last looked for
        budget      far pairs: the slab found then, plus how far both pieces had travelled by then.
                    Negative infinity for a pair that can touch
    */
    public struct PairPlane
    {
        public float3 n;
        public float budget;
    }

    // Counters the jobs add to, SoftSolver.stats. Reset by whoever reads them
    public static class SoftStats
    {
        public const int Pairs = 0;             // piece pairs from the broad phase
        public const int PieceContacts = 1;
        public const int ActiveContacts = 2;    // contacts of either kind that had something to push
        public const int FloorContacts = 3;
        public const int Cuts = 4;              // contacts the limit step had to act on
        public const int LimitPasses = 5;
        public const int Freezes = 6;           // contacts frozen because cutting did not settle
        public const int Asleep = 7;            // bodies asleep after the step
        public const int SleepChanges = 8;      // bodies that fell asleep or woke up in the step
        public const int Count = 9;
    }

    public static class Lattice
    {
        // Corner k of a cell sits at (k & 1, k >> 1 & 1, k >> 2 & 1)
        public static float3 Corner(int k) => new float3(k & 1, (k >> 1) & 1, (k >> 2) & 1);

        /*
        The 5 tets of a cell: one in the middle plus one per remaining corner.
        Variant 1 is the mirror image, used by every second cell (odd x + y + z) so
        the diagonals on a face shared by two cells are the same edge. A point on that
        face then moves the same way for both cells and the surface has no cracks.
        */
        public static int4 Tet(int variant, int t)
        {
            int4 tet;
            switch (t)
            {
                case 0: tet = new int4(0, 3, 6, 5); break;
                case 1: tet = new int4(0, 5, 6, 4); break;
                case 2: tet = new int4(0, 6, 3, 2); break;
                case 3: tet = new int4(5, 3, 6, 7); break;
                default: tet = new int4(0, 3, 5, 1); break;
            }
            return variant == 0 ? tet : tet ^ 1;
        }

        public static int Colour(int3 coord) => (coord.x & 1) | ((coord.y & 1) << 1) | ((coord.z & 1) << 2);
        public static int Variant(int3 coord) => (coord.x + coord.y + coord.z) & 1;
    }
}

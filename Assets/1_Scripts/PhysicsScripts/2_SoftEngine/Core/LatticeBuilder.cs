using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace SoftEngine
{
    /*
    Turns one mesh into lattice data. Runs once when a body is added, plain managed code.

        1. Box of cells around the mesh, in the body's own frame (rotated with it, world sized)
        2. Every triangle is cut by the cells it passes through, each cell keeps its part
           of the surface
        3. Cells without surface are inside the mesh when a ray from their centre crosses
           the surface an odd number of times
        4. Nodes for the corners of all kept cells, shared between neighbours
        5. The surface in a cell is split into convex patches, the pieces. Each gets its
           hull points (what collides), and every vertex is glued into one tet of its cell.
           Pieces are listed chunk by chunk (blocks of 2 x 2 x 2 cells)

    Why convex patches: two convex things are apart exactly when a plane fits between
    them, and that plane is cheap to find and to check. A patch that bends inwards
    (the inside of a bowl, an inner corner) is not convex, its hull would fill the
    hollow and keep other bodies out of it, so it is cut up until every part is
    convex enough (ConvexTolerance). A flat or outward bending cell stays one piece.

    Why hull points: only the corners of a hull matter. A flat patch with 30 vertices
    collides exactly like its 4 corners, so the rest is dropped (HullError). They are
    picked tet by tet: the vertices in one tet all move the same way, the tet is
    deformed as a whole, so a vertex that is inside the hull of its own tet's hull
    points stays inside it however the cell is bent or crushed. Picked for the whole
    cell at once, a dropped vertex can come out of the hull when the cell folds.

    All indices in the result are local to this body, SoftSolver shifts them.
    */
    public class LatticeBuilder
    {
        public float3[] nodePositions;      // world space
        public float[] nodeMass;
        public int[] cellNodes;             // 8 per cell
        public int3[] cellCoords;           // position of the cell in the body's lattice

        public List<Piece> pieces = new List<Piece>();
        public List<int2> chunks = new List<int2>();            // (first piece, number of pieces) of every block of ChunkCells^3 cells that has any
        public List<EmbeddedVertex> hullVerts = new List<EmbeddedVertex>();
        public List<EmbeddedVertex> colVerts = new List<EmbeddedVertex>();      // the full surface, per piece
        public List<int3> colTris = new List<int3>();
        public EmbeddedVertex[] visVerts;   // one per mesh vertex, same order

        // Both in cell sizes, and both should stay well below the skins (0.1 and 0.05 cells by default)
        public const double ConvexTolerance = 0.04;     // a patch may bend inwards this far and still be one piece: bodies hover that much further out there
        public const double HullError = 0.005;          // no vertex sticks out of the hull of its tet's hull points further than this (times how far the tet is stretched)

        public const int ChunkCells = 2;                // per axis, the broad phase looks at chunks before it looks at pieces (PairsJob)

        private const double BoundsPad = 1e-4;
        private const double WeldCells = 1e-4;      // vertices of a piece closer than this (in cells) are one vertex

        private readonly double h;
        private readonly int3 res;
        private readonly double3 latticeMin;
        private int[] cellOfGrid;
        private int[] nodeOfGrid;

        private class PieceBuild
        {
            public List<double3> verts = new List<double3>();
            public Dictionary<long, int> weld = new Dictionary<long, int>();
            public List<int3> tris = new List<int3>();
        }


        public LatticeBuilder(Vector3[] meshVerts, int[] meshTris, Matrix4x4 localToWorld, float cellSize, float mass)
        {
            h = cellSize;
            Quaternion rotation = localToWorld.rotation;
            Quaternion invRotation = Quaternion.Inverse(rotation);
            Vector3 origin = localToWorld.GetColumn(3);

            // 1. Mesh in the body frame, and the lattice around it
            double3[] p = new double3[meshVerts.Length];
            double3 lo = double.MaxValue, hi = double.MinValue;
            for (int i = 0; i < p.Length; i++)
            {
                Vector3 local = invRotation * (localToWorld.MultiplyPoint3x4(meshVerts[i]) - origin);
                p[i] = new double3(local.x, local.y, local.z);
                lo = math.min(lo, p[i]);
                hi = math.max(hi, p[i]);
            }
            double3 size = hi - lo + 2.0 * BoundsPad;
            res = math.max((int3)math.ceil(size / h), 1);
            latticeMin = 0.5 * (lo + hi) - 0.5 * h * (double3)res;

            int gridCount = res.x * res.y * res.z;
            bool[] keep = new bool[gridCount];

            // 2. Cut the triangles
            Dictionary<int, PieceBuild> builds = new Dictionary<int, PieceBuild>();
            List<double3> polygon = new List<double3>(12), scratch = new List<double3>(12);
            for (int t = 0; t < meshTris.Length; t += 3)
            {
                double3 a = p[meshTris[t]], b = p[meshTris[t + 1]], c = p[meshTris[t + 2]];
                int3 first = GridCoord(math.min(a, math.min(b, c)));
                int3 last = GridCoord(math.max(a, math.max(b, c)));

                for (int z = first.z; z <= last.z; z++)
                for (int y = first.y; y <= last.y; y++)
                for (int x = first.x; x <= last.x; x++)
                {
                    double3 cellMin = latticeMin + h * new double3(x, y, z);
                    polygon.Clear();
                    polygon.Add(a); polygon.Add(b); polygon.Add(c);
                    for (int axis = 0; axis < 3 && polygon.Count >= 3; axis++)
                    {
                        Clip(polygon, scratch, axis, cellMin[axis], 1.0);
                        Clip(scratch, polygon, axis, cellMin[axis] + h, -1.0);
                    }
                    if (polygon.Count < 3) continue;

                    int grid = GridIndex(new int3(x, y, z));
                    if (!builds.TryGetValue(grid, out PieceBuild build)) builds[grid] = build = new PieceBuild();
                    AddPolygon(build, polygon, cellMin);
                }
            }
            foreach (KeyValuePair<int, PieceBuild> build in builds)
                if (build.Value.tris.Count > 0) keep[build.Key] = true;

            // 3. Cells inside the mesh
            MarkInside(p, meshTris, keep);

            // Every mesh vertex needs a cell to live in
            int[] visGrid = new int[p.Length];
            for (int i = 0; i < p.Length; i++)
            {
                visGrid[i] = GridIndex(GridCoord(p[i]));
                keep[visGrid[i]] = true;
            }

            // 4. Cells and nodes, in grid order so the result never depends on dictionary order
            cellOfGrid = new int[gridCount];
            nodeOfGrid = new int[(res.x + 1) * (res.y + 1) * (res.z + 1)];
            for (int i = 0; i < nodeOfGrid.Length; i++) nodeOfGrid[i] = -1;

            List<int> nodes = new List<int>();
            List<int3> coords = new List<int3>();
            List<double3> positions = new List<double3>();
            for (int grid = 0; grid < gridCount; grid++)
            {
                cellOfGrid[grid] = -1;
                if (!keep[grid]) continue;

                int3 coord = GridCoordOf(grid);
                cellOfGrid[grid] = coords.Count;
                coords.Add(coord);
                for (int k = 0; k < 8; k++)
                {
                    int3 corner = coord + (int3)Lattice.Corner(k);
                    int nodeGrid = corner.x + (res.x + 1) * (corner.y + (res.y + 1) * corner.z);
                    if (nodeOfGrid[nodeGrid] == -1)
                    {
                        nodeOfGrid[nodeGrid] = positions.Count;
                        positions.Add(latticeMin + h * (double3)corner);
                    }
                    nodes.Add(nodeOfGrid[nodeGrid]);
                }
            }
            cellNodes = nodes.ToArray();
            cellCoords = coords.ToArray();

            nodePositions = new float3[positions.Count];
            nodeMass = new float[positions.Count];
            for (int i = 0; i < positions.Count; i++)
            {
                Vector3 local = new Vector3((float)positions[i].x, (float)positions[i].y, (float)positions[i].z);
                nodePositions[i] = rotation * local + origin;
            }
            float cornerMass = mass / cellCoords.Length / 8f;
            for (int i = 0; i < cellNodes.Length; i++) nodeMass[cellNodes[i]] += cornerMass;

            // 5. Pieces. Triangle normals point out of the mesh when its signed volume is positive
            double volume = 0.0;
            for (int t = 0; t < meshTris.Length; t += 3)
                volume += math.dot(p[meshTris[t]], math.cross(p[meshTris[t + 1]], p[meshTris[t + 2]]));
            double outward = volume < 0.0 ? -1.0 : 1.0;

            // Chunk after chunk, so the pieces of a chunk are next to each other in the list
            int3 chunkRes = (res + ChunkCells - 1) / ChunkCells;
            for (int chunk = 0; chunk < chunkRes.x * chunkRes.y * chunkRes.z; chunk++)
            {
                int3 first = ChunkCells * new int3(chunk % chunkRes.x, chunk / chunkRes.x % chunkRes.y, chunk / (chunkRes.x * chunkRes.y));
                int start = pieces.Count;

                for (int z = first.z; z < math.min(first.z + ChunkCells, res.z); z++)
                for (int y = first.y; y < math.min(first.y + ChunkCells, res.y); y++)
                for (int x = first.x; x < math.min(first.x + ChunkCells, res.x); x++)
                {
                    int grid = GridIndex(new int3(x, y, z));
                    if (!builds.TryGetValue(grid, out PieceBuild build) || build.tris.Count == 0) continue;
                    foreach (List<int> patch in ConvexPatches(build, outward)) AddPiece(build, patch, grid);
                }
                if (pieces.Count > start) chunks.Add(new int2(start, pieces.Count - start));
            }

            visVerts = new EmbeddedVertex[p.Length];
            for (int i = 0; i < p.Length; i++) visVerts[i] = Embed(p[i], visGrid[i]);
        }

        // Keeps the part of the polygon where sign * (x[axis] - value) >= 0
        private static void Clip(List<double3> input, List<double3> output, int axis, double value, double sign)
        {
            output.Clear();
            for (int i = 0; i < input.Count; i++)
            {
                double3 current = input[i], next = input[(i + 1) % input.Count];
                double dCurrent = sign * (current[axis] - value), dNext = sign * (next[axis] - value);

                if (dCurrent >= 0.0) output.Add(current);
                if ((dCurrent > 0.0 && dNext < 0.0) || (dCurrent < 0.0 && dNext > 0.0))
                {
                    double3 cut = current + (next - current) * (dCurrent / (dCurrent - dNext));
                    cut[axis] = value;
                    output.Add(cut);
                }
            }
        }

        // Adds a convex polygon to a piece as a fan of triangles, welding its vertices with the ones already there
        private void AddPolygon(PieceBuild build, List<double3> polygon, double3 cellMin)
        {
            int first = -1, previous = -1;
            foreach (double3 v in polygon)
            {
                int3 q = (int3)math.round((v - cellMin) / (h * WeldCells));
                long key = q.x + 100003L * (q.y + 100003L * q.z);
                if (!build.weld.TryGetValue(key, out int index))
                {
                    build.weld[key] = index = build.verts.Count;
                    build.verts.Add(v);
                }

                if (first == -1) first = index;
                else if (previous != -1 && index != previous && index != first)
                {
                    double3 normal = math.cross(build.verts[previous] - build.verts[first], build.verts[index] - build.verts[first]);
                    if (math.lengthsq(normal) > 1e-24) build.tris.Add(new int3(first, previous, index));
                }
                if (index != first) previous = index;
            }
        }

        /*
        Splits the triangles of one cell into patches that are convex seen from outside:
        no vertex of a patch lies in front of the plane of any of its triangles (by more
        than the tolerance). Greedy: start a patch with the first free triangle and add
        every triangle that fits.
        */
        private List<List<int>> ConvexPatches(PieceBuild build, double outward)
        {
            int count = build.tris.Count;
            double tolerance = ConvexTolerance * h;
            double3[] normal = new double3[count];
            for (int t = 0; t < count; t++)
            {
                int3 tri = build.tris[t];
                normal[t] = outward * math.normalizesafe(math.cross(build.verts[tri.y] - build.verts[tri.x], build.verts[tri.z] - build.verts[tri.x]));
            }

            List<List<int>> patches = new List<List<int>>();
            bool[] taken = new bool[count];
            for (int seed = 0; seed < count; seed++)
            {
                if (taken[seed]) continue;
                List<int> patch = new List<int> { seed };
                HashSet<int> verts = new HashSet<int> { build.tris[seed].x, build.tris[seed].y, build.tris[seed].z };
                taken[seed] = true;

                // Twice: a triangle that did not fit can fit once the patch has grown around it
                for (int pass = 0; pass < 2; pass++)
                for (int t = 0; t < count; t++)
                {
                    if (taken[t]) continue;
                    int3 tri = build.tris[t];

                    bool fits = true;
                    foreach (int other in patch)
                    {
                        double3 on = build.verts[build.tris[other].x];
                        for (int k = 0; k < 3 && fits; k++) fits = math.dot(normal[other], build.verts[tri[k]] - on) <= tolerance;
                        if (!fits) break;
                    }
                    if (fits)
                        foreach (int v in verts)
                            if (math.dot(normal[t], build.verts[v] - build.verts[tri.x]) > tolerance) { fits = false; break; }
                    if (!fits) continue;

                    taken[t] = true;
                    patch.Add(t);
                    verts.Add(tri.x); verts.Add(tri.y); verts.Add(tri.z);
                }
                patches.Add(patch);
            }
            return patches;
        }

        private void AddPiece(PieceBuild build, List<int> patch, int grid)
        {
            Dictionary<int, int> local = new Dictionary<int, int>();
            List<double3> points = new List<double3>();
            Piece piece = new Piece { cell = cellOfGrid[grid], vertStart = colVerts.Count, triStart = colTris.Count, triCount = patch.Count, hullStart = hullVerts.Count };

            foreach (int t in patch)
            {
                int3 tri = build.tris[t];
                for (int k = 0; k < 3; k++)
                {
                    if (!local.TryGetValue(tri[k], out int index))
                    {
                        local[tri[k]] = index = points.Count;
                        points.Add(build.verts[tri[k]]);
                    }
                    tri[k] = index + piece.vertStart;
                }
                colTris.Add(tri);
            }
            foreach (double3 v in points) colVerts.Add(Embed(v, grid));
            piece.vertCount = points.Count;

            // Hull points, picked among the vertices that share a tet (see the comment of the class)
            bool[] grouped = new bool[points.Count];
            List<int> group = new List<int>();
            List<double3> groupPoints = new List<double3>();
            for (int first = 0; first < points.Count; first++)
            {
                if (grouped[first]) continue;
                group.Clear();
                groupPoints.Clear();
                for (int i = first; i < points.Count; i++)
                {
                    if (grouped[i] || math.any(colVerts[piece.vertStart + i].nodes != colVerts[piece.vertStart + first].nodes)) continue;
                    grouped[i] = true;
                    group.Add(i);
                    groupPoints.Add(points[i]);
                }
                foreach (int i in HullPoints(groupPoints)) hullVerts.Add(colVerts[piece.vertStart + group[i]]);
            }
            piece.hullCount = hullVerts.Count - piece.hullStart;
            pieces.Add(piece);
        }

        /*
        A few of the points whose hull covers all of them to within HullError. Starts
        with the furthest points along the axes and keeps adding the point that sticks
        out of the hull most, until none sticks out further than allowed.
        */
        private List<int> HullPoints(List<double3> points)
        {
            List<int> chosen = new List<int>();
            List<double3> hull = new List<double3>();
            double[] sticksOut = new double[points.Count];
            for (int i = 0; i < points.Count; i++) sticksOut[i] = double.MaxValue;

            for (int axis = 0; axis < 6; axis++)
            {
                int furthest = 0;
                for (int i = 1; i < points.Count; i++)
                    if ((axis < 3 ? points[i][axis] : -points[i][axis - 3]) > (axis < 3 ? points[furthest][axis] : -points[furthest][axis - 3])) furthest = i;
                Choose(furthest);
            }

            double allowed = HullError * h;
            while (true)
            {
                // A point can only get closer to the hull as it grows, so one that was close enough stays that way
                int worst = -1;
                for (int i = 0; i < points.Count; i++)
                {
                    if (sticksOut[i] <= allowed) continue;
                    sticksOut[i] = DistanceToHull(points[i], hull);
                    if (sticksOut[i] > allowed && (worst == -1 || sticksOut[i] > sticksOut[worst])) worst = i;
                }
                if (worst == -1) return chosen;
                Choose(worst);
            }

            void Choose(int i)
            {
                if (sticksOut[i] == 0.0) return;
                sticksOut[i] = 0.0;
                chosen.Add(i);
                hull.Add(points[i]);
            }
        }

        // Distance from a point to the convex hull of others (0 inside). GJK, the same search PlaneJob does for two pieces
        private static double DistanceToHull(double3 point, List<double3> hull)
        {
            double3x4 simplex = default;
            int count = 0;
            double3 v = hull[0] - point;

            for (int step = 0; step < 64; step++)
            {
                double lengthSq = math.lengthsq(v);
                if (lengthSq < 1e-24) return 0.0;

                double3 nearest = hull[0] - point;      // the hull point furthest towards the point
                for (int i = 1; i < hull.Count; i++)
                    if (math.dot(v, hull[i] - point) < math.dot(v, nearest)) nearest = hull[i] - point;
                if (lengthSq - math.dot(v, nearest) <= 1e-12 * lengthSq) break;

                bool known = false;
                for (int k = 0; k < count; k++) known |= math.all(simplex[k] == nearest);
                if (known) break;

                simplex[count] = nearest;
                v = Geometry.ClosestOnSimplex(simplex, count + 1, out double4 share);
                int kept = 0;
                for (int k = 0; k <= count; k++)
                    if (share[k] > 0.0) simplex[kept++] = simplex[k];
                count = kept;
                if (count == 0) return 0.0;
            }
            return math.length(v);
        }

        /*
        A ray along +x through every row of cell centres. A centre is inside when an odd
        number of crossings lie before it. The rays are nudged off centre so they do not
        run exactly through the vertices and edges of symmetric meshes.
        */
        private void MarkInside(double3[] p, int[] tris, bool[] keep)
        {
            const double NudgeY = 0.0137, NudgeZ = 0.0071;
            List<double>[] rows = new List<double>[res.y * res.z];

            for (int t = 0; t < tris.Length; t += 3)
            {
                double3 a = p[tris[t]], b = p[tris[t + 1]], c = p[tris[t + 2]];
                double area = Cross2(b.yz - a.yz, c.yz - a.yz);
                if (math.abs(area) < 1e-18) continue;

                double3 triMin = math.min(a, math.min(b, c)), triMax = math.max(a, math.max(b, c));
                int y0 = math.max((int)math.ceil((triMin.y - latticeMin.y) / h - 0.5 - NudgeY), 0);
                int y1 = math.min((int)math.floor((triMax.y - latticeMin.y) / h - 0.5 - NudgeY), res.y - 1);
                int z0 = math.max((int)math.ceil((triMin.z - latticeMin.z) / h - 0.5 - NudgeZ), 0);
                int z1 = math.min((int)math.floor((triMax.z - latticeMin.z) / h - 0.5 - NudgeZ), res.z - 1);

                for (int z = z0; z <= z1; z++)
                for (int y = y0; y <= y1; y++)
                {
                    double2 q = new double2(latticeMin.y + (y + 0.5 + NudgeY) * h, latticeMin.z + (z + 0.5 + NudgeZ) * h);
                    double wc = Cross2(b.yz - a.yz, q - a.yz) / area;
                    double wa = Cross2(c.yz - b.yz, q - b.yz) / area;
                    double wb = Cross2(a.yz - c.yz, q - c.yz) / area;
                    if (wa < 0.0 || wb < 0.0 || wc < 0.0) continue;

                    int row = y + res.y * z;
                    if (rows[row] == null) rows[row] = new List<double>();
                    rows[row].Add(wa * a.x + wb * b.x + wc * c.x);
                }
            }

            for (int row = 0; row < rows.Length; row++)
            {
                if (rows[row] == null) continue;
                rows[row].Sort();

                int crossed = 0;
                for (int x = 0; x < res.x; x++)
                {
                    double centre = latticeMin.x + (x + 0.5) * h;
                    while (crossed < rows[row].Count && rows[row][crossed] < centre) crossed++;
                    if ((crossed & 1) == 1) keep[x + res.x * row] = true;
                }
            }
        }

        // The tet of the cell that holds the point best, and the point's weights in it
        private EmbeddedVertex Embed(double3 point, int grid)
        {
            int3 coord = GridCoordOf(grid);
            double3 u = math.saturate((point - (latticeMin + h * (double3)coord)) / h);
            int variant = Lattice.Variant(coord);
            int cell = cellOfGrid[grid];

            double bestInside = double.MinValue;
            int4 bestTet = 0;
            double4 bestWeights = 0;
            for (int t = 0; t < 5; t++)
            {
                int4 tet = Lattice.Tet(variant, t);
                double3 a = Lattice.Corner(tet.x), b = Lattice.Corner(tet.y), c = Lattice.Corner(tet.z), d = Lattice.Corner(tet.w);
                double3 abc = math.mul(math.inverse(new double3x3(a - d, b - d, c - d)), u - d);
                double4 weights = new double4(abc, 1.0 - math.csum(abc));

                double inside = math.cmin(weights);
                if (inside > bestInside) { bestInside = inside; bestTet = tet; bestWeights = weights; }
            }

            bestWeights = math.max(bestWeights, 0.0);
            bestWeights /= math.csum(bestWeights);
            return new EmbeddedVertex
            {
                nodes = new int4(cellNodes[cell * 8 + bestTet.x], cellNodes[cell * 8 + bestTet.y],
                                 cellNodes[cell * 8 + bestTet.z], cellNodes[cell * 8 + bestTet.w]),
                w = (float4)bestWeights
            };
        }

        private static double Cross2(double2 a, double2 b) => a.x * b.y - a.y * b.x;

        private int3 GridCoord(double3 point) => math.clamp((int3)math.floor((point - latticeMin) / h), 0, res - 1);
        private int GridIndex(int3 coord) => coord.x + res.x * (coord.y + res.y * coord.z);
        private int3 GridCoordOf(int grid) => new int3(grid % res.x, grid / res.x % res.y, grid / (res.x * res.y));
    }
}

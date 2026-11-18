# SoftEngine handoff

State of 16 November 2026. For whoever continues this work, human or AI. Read this first, then `HowItWorks.pdf` in this folder for the full explanation of the design.

## 1. What this is and where it stands

A soft body physics engine for Unity 6000.3.8f1 (Burst, Jobs, Collections, Mathematics), written as a full rewrite next to the owner's earlier engine.

- New engine: `Assets/1_Scripts/PhysicsScripts/2_SoftEngine/`, namespace `SoftEngine`.
- Earlier engine: `Assets/1_Scripts/PhysicsScripts/1_PhysicsSolver/`. It still works and must stay untouched.
- The owner considers the current state the final version of the solver. Nothing is committed by the assistant; the owner commits through GitHub Desktop.

Goals, in the owner's order: zero penetration, fast enough for a game, believable behaviour, code simple enough to keep extending. Momentum loss is acceptable. Weird artefacts are not.

What is verified and how:

| Claim | How it was checked |
|---|---|
| No crossings, real surface distance never under 12.5 mm | Independent checker (`Editor/SoftChecker.cs`) on the full triangle surface, in all 18 scenarios and the owner's scene |
| Two runs are identical | Byte comparison of the per-body output of two headless runs, with the limit off, and with the limit on and `repeatable` on |
| Sleeping bodies wake | Scenario `wake` |
| No job safety errors | One pass with `safety=1` over pile, tower, wake and the owner's scene |
| Play mode | By the owner only. The assistant never ran Play mode after the first day. `SoftWorld.Update` (smooth drawing) has only run on the owner's side |

## 2. Rules the owner set

- Do not modify, move or delete the owner's scripts. New code goes in its own folder and namespace.
- Scenes may be changed for testing. Do not build scenes or enter Play mode while the open scene has unsaved changes (`ping` answers `unsaved=True`).
- Simpler code is better. One small change at a time. No debug or temporary code left behind (`grep -rn TEMP` must find nothing).
- Comments in plain language, explaining why. Every job has a header comment; keep them true when the code changes.
- Try several approaches for important parts, measure in Unity, keep the best, and report what was tried and why.
- Do not commit.

## 3. File map

```
2_SoftEngine/
  SoftWorld.cs          one per scene: owns the solver, steps it in FixedUpdate, draws in Update
  SoftBody.cs           per object: mass, stiffness, friction, damping, static
  Core/
    SoftSolver.cs       all data, AddBody, Step / Substep (the order of the stages), Sleep, Work
    SoftTypes.cs        SoftSettings, Contact, PairPlane, Piece, BodyRange, SoftStats, Lattice
    LatticeBuilder.cs   mesh -> lattice, tets, pieces, hull points, chunks
    MotionJobs.cs       PredictJob, VelocityJob, DampJob, SleepJob, CopyJob, ElasticJob,
                        BodyCentreJob, VisualJob, BlendJob
    DetectionJobs.cs    PieceJob, LatePieceJob, PairsJob, PairSearchJob, PairListJob,
                        PlaneJob, BuildContactsJob
    ContactJobs.cs      ContactSweepJob, LimitJob
    Geometry.cs         small helpers (points in cells, least motion, closest on simplex)
  Editor/
    SoftTestBridge.cs   headless test runner, driven by a request file
    SoftScenarios.cs    the 18 test scenes, and "open" = the scene open in the Editor
    SoftChecker.cs      independent distance / crossing / inversion / energy check
    SoftSceneBuilder.cs turns a scenario into GameObjects for either engine
  Testing/
    OldEngineStartVelocity.cs   gives a body of the earlier engine a start velocity, for side by side scenes
Docs/SoftEngine/
  HowItWorks.pdf / .html   the explanation (the .html is the source, see section 9)
  HANDOFF.md               this file
  tools/                   the test scripts described below
```

## 4. One substep

`SoftSolver.Substep`, each stage finished before the next starts:

1. `PredictJob`: gravity, velocity, `prev` = start of substep.
2. `ElasticJob` x 8 colours, then `CopyJob` (`expected` = where elastic left the nodes, `buffer` = second copy of `pos`).
3. `PieceJob` -> `Detect()`: `PairsJob` (decide, boxes) -> `PairSearchJob` (per body, NativeStream) -> `PairListJob` -> `PlaneJob` -> `BuildContactsJob`.
4. `ContactSweepJob` x `contactSweeps`, one body per work item, ping-pong between `pos` and `buffer`; the last sweep writes `pos`, does friction and sets `pushedFar`.
5. If any `pushedFar`: `LatePieceJob` + `Detect(onlyLate)`. Then `LimitJob`.
6. `VelocityJob`, then `DampJob` if any body has damping.

After the substeps of a step: `Sleep()` (`SleepJob`, then apply changes on the main thread).

## 5. Invariants: what must stay true

Breaking any of these breaks either the no-penetration guarantee or repeatability.

1. **Nodes move in straight lines within a substep, and `LimitJob` is the last thing that moves them.** The guarantee is: plane fixed at `prev`, hull points checked at `pos`.
2. **The limit step only ever takes motion away.** Slide is clamped to the box of the node's own path (`prev`..`goal`); the broad phase boxes count on that.
3. **Every pair that could come within `hardSkin` must be in the contact list when `LimitJob` runs.** That is what the "needed" test in `PlaneJob`, `regionMargin`, `pushedFar` and the second detection run are for. Anything that lets a node move further than its piece's box and reach assumed needs its own answer here.
4. **The travel budget is a proof, not a heuristic.** `travel[p]` must grow by at least the largest node move of the piece's cell each substep, and is re-based only in `PairListJob` together with every stored budget.
5. **Hull points are picked per tet** in `LatticeBuilder`. Per cell is unsafe under crush.
6. **A body writes only its own nodes in `ContactSweepJob`** and reads the other side from `from`. Shared per-node arrays (`lift`, `liftGrip`, `floorGrip`) are only touched for own nodes.
7. **The pair list is sorted by (first piece, second piece)** and built in body order. Warm starts and repeatability depend on it.
8. **Nothing in `Step` may depend on the clock when `repeatable` is on.** The only clock use is in `Step`, behind that flag.
9. **Sleeping = `invMass` 0 + `bodyStatic` true.** On any change: `SortCellsByColour()` and `builtFor[0] = -1` (the pair list leaves out pairs of two bodies that do not move). `bodyFixed` is "really static", `bodyStatic` is "does not move now".

## 6. Testing

The Editor must be open with the project. Tests run headless inside it (no Play mode) through a request file, `Temp/SoftBridge/request.txt`. The scripts in `Docs/SoftEngine/tools/` wrap that. They contain this machine's absolute project path.

```
python3 uc.py -t 240 refresh             recompile (uses the older ClaudeBridge), prints "ok compiled" or the errors
python3 sb.py ping                       which scene is open, unsaved changes?
python3 sb.py -t 300 "run pile 300 check=10 safety=0"
python3 suite.py safety=0                11 scenarios, one line each
./stack.sh stack 300                     resting quality: node speeds per body
./prof.sh "pile 300" "open 300"          per-stage milliseconds
python3 sleepy.py "open 750"             how sleeping develops, who is still awake
python3 twitch.py "open 400" body,body   node speed of the named bodies over the last 100 steps
python3 fit.py                           refits the constants of SoftSolver.Work(), see section 7
```

`run <scenario> <steps> key=value ...`. Useful keys: `check=N` (run the checker every N steps), `safety=0|1`, `profile=1`, `shot=N` (PNG every N steps), `tag=name` (output files `Temp/SoftBridge/<tag>_steps.csv`, `_bodies.csv`), `maxSeconds=`, `bodyDamping=`, `youngScale=`, `massScale=`, `speedScale=`, and any `SoftSettings` field by name (`contactSweeps=6`, `maxStepMs=6`, `sleepTime=0`).

Scenarios: drop, rest, slide, tower, midair, headon, fast, smash, pile, squeeze, coils, ramp, stack, onslab, float, bowl, edge, wake, and `open` (the scene open in the Editor with its own settings).

What a healthy run looks like: `crossings=0`, `minDistance` at least 0.0125, `midair` cube vy near -17.85, `slide` x near 0.917, `stack` asleep and in place, `wake` ends with the upper cube on the floor.

### Traps when measuring

- **Unity in the background is throttled by macOS**: every stage reads 4 to 7 times slower, uniformly (elastic 5 to 9 ms instead of 1.3 on the pile). Check the elastic number before believing any timing. Ask for Unity to be in the foreground.
- The first run after a recompile is slow (Burst compiling). Time the second.
- Burst safety checks roughly double every time. The owner's Editor currently has them off; pass `safety=1` to test job safety.
- The owner edits and saves their scene while work goes on (it grew from 24 to 44 bodies in one afternoon). `open` timings are only comparable within minutes. A newly added serialized field is saved with whatever default it had at that moment (existing bodies have `damping: 0`).
- `check=` on a tangled mesh (`smash`) can take minutes; the client reports TIMEOUT while Unity is still busy.
- Resting quality must be tested with both odd and even `contactSweeps`, and on `stack` (two moving bodies), not only on `rest` or `onslab` (one side static hides two-sided solver bugs).

## 7. Decisions and why (short form)

Full reasoning is in `HowItWorks.pdf` sections 4 to 8 and 10.

- **Elastic**: stable neo-Hookean regrouped as `CD = sqrt(I + J^2 - 4J)` and `CH = J - 1`, solved as one 2x2 XPBD step in double. The pair `|F| - sqrt(3)`, `J - 1` has no shear stiffness; the textbook split explodes for stiff bodies with one pass per substep.
- **Collision on hull points of convex pieces**, no triangles at runtime.
- **Broad phase**: bodies -> chunks -> pieces box tests, list kept while every piece stays in its frame box. A hashed grid and partial rebuilds were both slower.
- **Contact solve parallel by body** with a snapshot. No give-back of over-pushes (it made the result depend on odd or even sweep counts). Friction is target-based (each side takes its share of the slide at sweep start, counting what its spot already moved). Firm contacts move whole cells (per-point pushes added energy).
- **Limit step**: slide 4 passes, stop 6, then freeze. In a hurry: stop 1, then freeze.
- **3 sweeps** by default: as calm as 6 and cheaper.
- **Damping** on velocities per body, relative to the body's rigid motion. Damping inside the elastic constraints softens stiff bodies; per-cell damping does not touch a bounce.
- **Sleep test on the average node displacement** (plus a cap on the worst node), because pinched soft bodies tremble forever and would otherwise never sleep.
- **Step-time limit**: hurry first, then fewer substeps. `repeatable` replaces the clock by `SoftSolver.Work()`.

### The constants in `Work()`

`0.16 + 0.00011 * moving cells + 0.000035 * pairs + 0.00054 * piece contacts + 0.0014 * cuts` milliseconds per substep. Fitted by least squares to per-step data of drop, midair, headon, tower, smash, fast and squeeze plus the pile average, M1, safety checks off, without hurry mode. `tools/fit.py` redoes the fit from `Temp/SoftBridge/suite_*_steps.csv` (run `suite.py safety=0` first, with Unity in the foreground). To recalibrate for another machine, rerun and paste the five numbers.

## 8. Measured (M1, headless, per physics step)

- Owner's scene when it had 24 bodies, safety off: 4.4 avg / 11 worst before the parallel round, 2.8 / 6.7 after. Collision part 3.1 -> 1.43 ms, elastic 1.26. Safety on: 8.6 / 23 -> 5.1 / 13.
- Pile (27 bodies): 3.1 -> 2.3 ms, collision 1.67 -> 0.89. Safety on 4.1.
- Sleeping: stack 1.2 -> 0.23 ms; owner's scene (44 bodies, damping 0.1) 0.85 ms with 38 asleep.
- Limit 6 ms: worst step tower 22.8 -> 6.4, smash 45.2 -> 5.3, squeeze 15.4 -> 6.2, at about 90% speed (clock version). With `repeatable`: tower 75%, smash 89%, squeeze 91%.
- Earlier engine, Play mode, 13 and 14 November: pile 62 avg / 178 worst, tower 121 / 598, stack 17, rest 13.5.

## 9. Open issues, most useful first

1. **Narrow phase** is the largest collision cost (0.44 of 1.43 ms). GJK runs every substep for every pair within about 6 cm. Reusing a pair's plane for all substeps of a frame keeps the guarantee (any separating plane is valid); the risk is lagging normals on rolling contacts. Untried.
2. **Sleeping bodies still get hull points, boxes and floor contacts computed** every substep. `PieceJob` could skip pieces of bodies that have been still for more than one substep (all its outputs persist).
3. **`Work()` overestimates in hurry mode** (cost per cut was fitted without it), so `repeatable` slows more than needed in crush scenes.
4. **Bodies wake each other while a pile settles** (a slow creeper wakes its neighbours each time it has moved 1 cm). Harmless, costs a pair-list rebuild per change.
5. **Pinched soft bodies tremble until they sleep.** Cause: the contact solve shuffles a body that has nowhere to go. Velocity damping cannot fix it.
6. **Momentum is not conserved** by the one-sided solve (8% in `headon`). Accepted by the owner.
7. **No restitution, no static friction.** A restitution pass would be a second, velocity-level contact sweep.
8. Bodies created overlapping stay overlapping. `smash` leaves the ball tangled. Self collision exists but is barely tested.
9. Not built, by the owner's choice: Unity colliders and rigidbodies, removing bodies, forces, grabbing, raycasts, events, a standalone build.

## 10. Rebuilding the PDF

`HowItWorks.html` is the source (plain HTML with inline SVG diagrams). After editing it:

```
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" --headless --disable-gpu \
  --no-pdf-header-footer --print-to-pdf="$PWD/HowItWorks.pdf" "file://$PWD/HowItWorks.html"
```

Any Chromium-based browser works the same way.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using static System.FormattableString;

namespace SoftEngine.Testing
{
    /*
    Runs the new engine without Play mode, on request, and writes what happened.
    Requests arrive as a file, like ClaudeBridge:

        Temp/SoftBridge/request.txt     "<id> run <scenario> <steps> [key=value ...]"      (scenario "open": the open scene)
        Temp/SoftBridge/response.txt    "<id>" then a summary

    Other commands: ping, mesh <name>, open <scene path>, scene <scenario> <old|new|both> [save]
    (the last one is SoftSceneBuilder).

    key can be any field of SoftSettings (skin=0.05, selfCollision=true, gravity=0,-10,0) or
        tag=name        file name for the output, default the scenario
        check=n         measure distances, crossings and energy every n steps (default 1, 0 = never)
        shot=n          PNG every n steps
        cam=px,py,pz,tx,ty,tz
        profile=1       time per stage
        fixedDt=0.02
        maxSeconds=60   stop early when the run takes longer than this, the Editor is blocked while it runs
        safety=0 / 1    Burst safety checks and the jobs debugger off (for timing) or on (to find job mistakes) while it runs
        youngScale, massScale, speedScale   multiply every body's stiffness, mass or start velocity
        bodyDamping                         every body's damping

    Output: <tag>_steps.csv (one row per step) and <tag>_bodies.csv (one row per step and body).
    Nothing runs unless a request file appears.
    */
    [InitializeOnLoad]
    public static class SoftTestBridge
    {
        private const string Dir = "Temp/SoftBridge";
        private static double nextPoll;

        static SoftTestBridge()
        {
            Directory.CreateDirectory(Dir);
            EditorApplication.update += Tick;
        }

        private static void Tick()
        {
            if (EditorApplication.timeSinceStartup < nextPoll) return;
            nextPoll = EditorApplication.timeSinceStartup + 0.1;

            string requestPath = Dir + "/request.txt";
            if (!File.Exists(requestPath)) return;
            string[] parts = File.ReadAllText(requestPath).Trim().Split(' ');
            File.Delete(requestPath);
            if (parts.Length < 2) return;

            string answer;
            try
            {
                if (parts[1] == "ping") answer = "ok scene=" + UnityEngine.SceneManagement.SceneManager.GetActiveScene().path + " unsaved=" + UnityEngine.SceneManagement.SceneManager.GetActiveScene().isDirty;
                else if (parts[1] == "open")
                {
                    if (EditorApplication.isPlaying) answer = "error stop Play mode first";
                    else if (SoftSceneBuilder.UnsavedWork()) answer = "error the open scene has unsaved changes, save it first";
                    else
                    {
                        UnityEditor.SceneManagement.EditorSceneManager.OpenScene(parts[2]);
                        answer = "ok opened " + parts[2];
                    }
                }
                else if (parts[1] == "run") answer = Run(parts);
                else if (parts[1] == "mesh")
                {
                    Mesh mesh = SoftScenarios.FindMesh(parts[2]);
                    answer = $"ok {mesh.name} verts={mesh.vertexCount} tris={mesh.triangles.Length / 3} bounds={mesh.bounds}";
                }
                else if (parts[1] == "scene") answer = SoftSceneBuilder.Build(parts);
                else answer = "error unknown command " + parts[1];
            }
            catch (Exception e)
            {
                answer = "error " + e;
            }

            File.WriteAllText(Dir + "/response.tmp", parts[0] + "\n" + answer + "\n");
            File.Delete(Dir + "/response.txt");
            File.Move(Dir + "/response.tmp", Dir + "/response.txt");
        }

        private static string Run(string[] parts)
        {
            string scenario = parts[2];
            int steps = int.Parse(parts[3]);

            // Bodies and settings, "open" is the scene that is open in the Editor
            SoftSettings settings = SoftSettings.Default;
            List<TestBody> bodies = scenario == "open" ? SoftScenarios.FromOpenScene(ref settings) : SoftScenarios.Build(scenario);
            if (bodies == null) return "error unknown scenario " + scenario;
            SoftScenarios.Adjust(scenario, ref settings);
            Dictionary<string, string> options = new Dictionary<string, string>();
            object boxed = settings;
            for (int i = 4; i < parts.Length; i++)
            {
                string[] pair = parts[i].Split('=');
                if (pair.Length != 2) continue;
                FieldInfo field = typeof(SoftSettings).GetField(pair[0]);
                if (field == null) options[pair[0]] = pair[1];
                else field.SetValue(boxed, Parse(field.FieldType, pair[1]));
            }
            settings = (SoftSettings)boxed;

            string tag = options.TryGetValue("tag", out string t) ? t : scenario;
            int checkEvery = options.TryGetValue("check", out string c) ? int.Parse(c) : 1;
            int shotEvery = options.TryGetValue("shot", out string s) ? int.Parse(s) : 0;
            float fixedDt = options.TryGetValue("fixedDt", out string f) ? Number(f) : 0.02f;
            float maxSeconds = options.TryGetValue("maxSeconds", out string m) ? Number(m) : 60f;
            Stopwatch wall = Stopwatch.StartNew();

            bool wasSynchronous = BurstCompiler.Options.EnableBurstCompileSynchronously;
            bool wasSafe = BurstCompiler.Options.EnableBurstSafetyChecks, wasDebugged = JobsUtility.JobDebuggerEnabled;
            BurstCompiler.Options.EnableBurstCompileSynchronously = true;
            if (options.TryGetValue("safety", out string safety))
            {
                BurstCompiler.Options.EnableBurstSafetyChecks = safety != "0";
                JobsUtility.JobDebuggerEnabled = safety != "0";
            }

            SoftSolver solver = new SoftSolver(settings) { profile = options.ContainsKey("profile") };
            NativeArray<double> check = new NativeArray<double>(CheckJob.Count, Allocator.Persistent);
            StringBuilder summary = new StringBuilder();
            try
            {
                Stopwatch clock = Stopwatch.StartNew();
                foreach (TestBody body in bodies)
                {
                    TestBody scaled = body;
                    if (options.TryGetValue("youngScale", out string ys)) scaled.young *= Number(ys);
                    if (options.TryGetValue("massScale", out string ms)) scaled.mass *= Number(ms);
                    if (options.TryGetValue("speedScale", out string ss)) scaled.velocity *= Number(ss);
                    if (options.TryGetValue("bodyDamping", out string bd)) scaled.damping = Number(bd);
                    SoftScenarios.AddTo(solver, scaled);
                }
                summary.AppendLine(Invariant($"ok {scenario} steps={steps} bodies={bodies.Count} nodes={solver.pos.Length} cells={solver.cellCoords.Length} pieces={solver.pieces.Length} hullPoints={solver.hullVerts.Length} verts={solver.colVerts.Length} tris={solver.colTris.Length} buildMs={clock.ElapsedMilliseconds} safetyChecks={BurstCompiler.Options.EnableBurstSafetyChecks}"));

                using StreamWriter stepsCsv = new StreamWriter($"{Dir}/{tag}_steps.csv");
                using StreamWriter bodiesCsv = new StreamWriter($"{Dir}/{tag}_bodies.csv");
                stepsCsv.WriteLine("step,ms,pairs,pieceContacts,activeContacts,floorContacts,cuts,passes,freezes,asleep,sleepChanges,minDistance,crossings,selfMin,selfCrossings,lowest,inverted,minJ,maxEdge,kinetic,potential,elastic,px,py,pz");
                bodiesCsv.WriteLine("step,body,x,y,z,minY,maxY,maxSpeed,vx,vy,vz");

                double msTotal = 0.0, msMax = 0.0, firstMs = 0.0, simulated = 0.0;
                int worstStep = 0;
                double[] worstStages = new double[solver.stageMs.Length];
                double minDistance = double.MaxValue, selfMin = double.MaxValue, lowest = double.MaxValue, minJ = double.MaxValue, maxEdge = 0.0, maxEnergy = 0.0, firstEnergy = 0.0;
                int minDistanceStep = 0, crossings = 0, selfCrossings = 0, maxInverted = 0, maxPairs = 0, maxPasses = 0, badNodes = 0;
                long[] totals = new long[SoftStats.Count];

                for (int step = 1; step <= steps; step++)
                {
                    if (wall.Elapsed.TotalSeconds > maxSeconds)
                    {
                        summary.AppendLine($"STOPPED after step {step - 1}, over {maxSeconds} s");
                        steps = step - 1;
                        break;
                    }
                    for (int i = 0; i < SoftStats.Count; i++) solver.stats[i] = 0;

                    double[] stagesBefore = (double[])solver.stageMs.Clone();
                    clock.Restart();
                    solver.Step(fixedDt);
                    double ms = clock.Elapsed.TotalMilliseconds;
                    if (step == 1) firstMs = ms;
                    else if (ms > msMax)
                    {
                        msMax = ms;
                        worstStep = step;
                        for (int i = 0; i < worstStages.Length; i++) worstStages[i] = solver.stageMs[i] - stagesBefore[i];
                    }
                    if (step > 1) msTotal += ms;
                    simulated += solver.Simulated;

                    for (int i = 0; i < SoftStats.Count; i++) totals[i] += solver.stats[i];
                    maxPairs = math.max(maxPairs, solver.stats[SoftStats.Pairs] / settings.substeps);
                    maxPasses = math.max(maxPasses, solver.stats[SoftStats.LimitPasses]);

                    bool checkNow = checkEvery > 0 && (step % checkEvery == 0 || step == steps);
                    if (checkNow)
                    {
                        new CheckJob
                        {
                            nearDistance = 2f * settings.skin, cellSize = settings.cellSize, gravity = settings.gravity,
                            pieces = solver.pieces.AsArray(), colVerts = solver.colVerts.AsArray(), colTris = solver.colTris.AsArray(),
                            cellNodes = solver.cellNodes.AsArray(), cellBody = solver.cellBody.AsArray(), cellCoords = solver.cellCoords.AsArray(),
                            cellMu = solver.cellMu.AsArray(), cellLambda = solver.cellLambda.AsArray(), invMass = solver.invMass.AsArray(),
                            bodyStatic = solver.bodyStatic.AsArray(), pos = solver.pos.AsArray(), vel = solver.vel.AsArray(), result = check
                        }.Run();

                        if (check[CheckJob.MinDistance] < minDistance) { minDistance = check[CheckJob.MinDistance]; minDistanceStep = step; }
                        selfMin = math.min(selfMin, check[CheckJob.SelfMinDistance]);
                        lowest = math.min(lowest, check[CheckJob.LowestVertex]);
                        minJ = math.min(minJ, check[CheckJob.MinJ]);
                        maxEdge = math.max(maxEdge, check[CheckJob.MaxEdge]);
                        crossings += (int)check[CheckJob.Crossings];
                        selfCrossings += (int)check[CheckJob.SelfCrossings];
                        maxInverted = math.max(maxInverted, (int)check[CheckJob.Inverted]);
                        badNodes = math.max(badNodes, (int)check[CheckJob.BadNodes]);
                        double energy = check[CheckJob.Kinetic] + check[CheckJob.Potential] + check[CheckJob.Elastic];
                        if (firstEnergy == 0.0) firstEnergy = energy;
                        maxEnergy = math.max(maxEnergy, energy);
                    }

                    NativeArray<int> st = solver.stats;
                    stepsCsv.WriteLine(Invariant($"{step},{ms:F2},{st[SoftStats.Pairs]},{st[SoftStats.PieceContacts]},{st[SoftStats.ActiveContacts]},{st[SoftStats.FloorContacts]},{st[SoftStats.Cuts]},{st[SoftStats.LimitPasses]},{st[SoftStats.Freezes]},{st[SoftStats.Asleep]},{st[SoftStats.SleepChanges]},")
                        + (checkNow ? Invariant($"{Short(check[CheckJob.MinDistance])},{check[CheckJob.Crossings]},{Short(check[CheckJob.SelfMinDistance])},{check[CheckJob.SelfCrossings]},{check[CheckJob.LowestVertex]:F4},{check[CheckJob.Inverted]},{check[CheckJob.MinJ]:F3},{check[CheckJob.MaxEdge]:F3},{check[CheckJob.Kinetic]:G6},{check[CheckJob.Potential]:G6},{check[CheckJob.Elastic]:G6},{check[CheckJob.MomentumX]:G6},{check[CheckJob.MomentumY]:G6},{check[CheckJob.MomentumZ]:G6}") : ",,,,,,,,,,,,,"));

                    for (int b = 0; b < bodies.Count; b++) bodiesCsv.WriteLine(Invariant($"{step},{bodies[b].name},") + BodyLine(solver, b));
                    if (shotEvery > 0 && step % shotEvery == 0) Shot(solver, bodies, $"{Dir}/{tag}_{step:D4}.png", options);
                }

                if (steps < 1) steps = 1;
                int timed = math.max(steps - 1, 1);
                summary.AppendLine(Invariant($"ms avg={msTotal / timed:F2} max={msMax:F2} total={msTotal:F0} first={firstMs:F0} timeScale={simulated / (steps * fixedDt):F3}"));
                if (solver.profile)
                {
                    summary.Append("stageMs");
                    for (int i = 0; i < solver.stageMs.Length; i++) summary.Append(Invariant($" {SoftSolver.StageNames[i]}={solver.stageMs[i] / steps:F2}"));
                    summary.AppendLine();
                    summary.Append(Invariant($"worstStep={worstStep}"));
                    for (int i = 0; i < worstStages.Length; i++) summary.Append(Invariant($" {SoftSolver.StageNames[i]}={worstStages[i]:F2}"));
                    summary.AppendLine();
                }
                summary.AppendLine(Invariant($"minDistance={Short(minDistance)} atStep={minDistanceStep} crossings={crossings} selfMin={Short(selfMin)} selfCrossings={selfCrossings} lowestVertex={lowest:F4}"));
                summary.AppendLine(Invariant($"invertedMax={maxInverted} invertedEnd={check[CheckJob.Inverted]} minJ={minJ:F3} maxEdge={maxEdge:F3} nanNodes={badNodes}"));
                summary.AppendLine(Invariant($"perStep pairsMax={maxPairs} pieceContacts={totals[SoftStats.PieceContacts] / steps} activeContacts={totals[SoftStats.ActiveContacts] / steps} floorContacts={totals[SoftStats.FloorContacts] / steps}"));
                int asleep = 0;
                for (int i = 0; i < solver.bodyAsleep.Length; i++) asleep += solver.bodyAsleep[i] ? 1 : 0;
                summary.AppendLine(Invariant($"total cuts={totals[SoftStats.Cuts]} freezes={totals[SoftStats.Freezes]} passesMaxPerStep={maxPasses} asleepAtEnd={asleep}"));
                summary.AppendLine(Invariant($"energy first={firstEnergy:G5} max={maxEnergy:G5} end={check[CheckJob.Kinetic] + check[CheckJob.Potential] + check[CheckJob.Elastic]:G5} (kinetic={check[CheckJob.Kinetic]:G5} elastic={check[CheckJob.Elastic]:G5}) momentum=({check[CheckJob.MomentumX]:G5},{check[CheckJob.MomentumY]:G5},{check[CheckJob.MomentumZ]:G5})"));
                for (int b = 0; b < bodies.Count; b++) summary.AppendLine(bodies[b].name + " " + BodyLine(solver, b));
            }
            finally
            {
                solver.Dispose();
                check.Dispose();
                BurstCompiler.Options.EnableBurstCompileSynchronously = wasSynchronous;
                BurstCompiler.Options.EnableBurstSafetyChecks = wasSafe;
                JobsUtility.JobDebuggerEnabled = wasDebugged;
            }
            return summary.ToString();
        }

        // x,y,z,minY,maxY,maxSpeed,vx,vy,vz of one body
        private static string BodyLine(SoftSolver solver, int body)
        {
            BodyRange range = solver.bodies[body];
            float3 centre = 0f, velocity = 0f;
            float minY = float.MaxValue, maxY = float.MinValue, maxSpeed = 0f;
            for (int i = range.nodeStart; i < range.nodeStart + range.nodeCount; i++)
            {
                centre += solver.pos[i];
                velocity += solver.vel[i];
                minY = math.min(minY, solver.pos[i].y);
                maxY = math.max(maxY, solver.pos[i].y);
                maxSpeed = math.max(maxSpeed, math.length(solver.vel[i]));
            }
            centre /= range.nodeCount;
            velocity /= range.nodeCount;
            return Invariant($"{centre.x:F3},{centre.y:F3},{centre.z:F3},{minY:F3},{maxY:F3},{maxSpeed:F2},{velocity.x:F3},{velocity.y:F3},{velocity.z:F3}");
        }

        // Draws the bodies as they are now into a PNG, with objects that only exist for the picture
        private static void Shot(SoftSolver solver, List<TestBody> bodies, string path, Dictionary<string, string> options)
        {
            const int Width = 960, Height = 540, ShotLayer = 31;     // an unused layer, so the open scene is not in the picture
            solver.UpdateVisuals();

            List<UnityEngine.Object> temporary = new List<UnityEngine.Object>();
            Bounds all = new Bounds(solver.bodyCentre[0], Vector3.zero);
            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            for (int b = 0; b < bodies.Count; b++)
            {
                BodyRange range = solver.bodies[b];
                Mesh mesh = UnityEngine.Object.Instantiate(SoftScenarios.FindMesh(bodies[b].mesh));
                mesh.SetVertices(solver.visPos.AsArray(), range.visStart, range.visCount);
                mesh.RecalculateNormals();
                mesh.RecalculateBounds();

                GameObject go = new GameObject("shot body") { hideFlags = HideFlags.HideAndDontSave, layer = ShotLayer };
                go.transform.position = solver.bodyCentre[b];
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                Material material = new Material(shader) { color = Color.HSVToRGB(b * 0.618f % 1f, 0.55f, 0.95f) };
                go.AddComponent<MeshRenderer>().sharedMaterial = material;
                all.Encapsulate(new Bounds(go.transform.position, mesh.bounds.size));
                temporary.Add(go); temporary.Add(mesh); temporary.Add(material);
            }

            GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floor.hideFlags = HideFlags.HideAndDontSave;
            floor.layer = ShotLayer;
            floor.transform.position = new Vector3(all.center.x, solver.settings.floorY, all.center.z);
            floor.transform.localScale = Vector3.one * 5f;
            floor.SetActive(solver.settings.hasFloor);
            temporary.Add(floor);

            GameObject holder = new GameObject("shot camera") { hideFlags = HideFlags.HideAndDontSave };
            Camera camera = holder.AddComponent<Camera>();
            Light light = new GameObject("shot light") { hideFlags = HideFlags.HideAndDontSave }.AddComponent<Light>();
            light.type = LightType.Directional;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
            temporary.Add(holder); temporary.Add(light.gameObject);

            Vector3 target = all.center, position = all.center + new Vector3(0.3f, 0.35f, -1f).normalized * (all.extents.magnitude * 2.2f + 2f);
            if (options.TryGetValue("cam", out string cam))
            {
                string[] v = cam.Split(',');
                position = new Vector3(Number(v[0]), Number(v[1]), Number(v[2]));
                target = new Vector3(Number(v[3]), Number(v[4]), Number(v[5]));
            }
            holder.transform.SetPositionAndRotation(position, Quaternion.LookRotation(target - position));
            camera.cullingMask = 1 << ShotLayer;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.16f, 0.17f, 0.2f);

            RenderTexture texture = new RenderTexture(Width, Height, 24);
            RenderTexture previous = RenderTexture.active;
            camera.targetTexture = texture;
            camera.Render();
            RenderTexture.active = texture;
            Texture2D image = new Texture2D(Width, Height, TextureFormat.RGB24, false);
            image.ReadPixels(new Rect(0, 0, Width, Height), 0, 0);
            image.Apply();
            File.WriteAllBytes(path, image.EncodeToPNG());
            RenderTexture.active = previous;
            camera.targetTexture = null;

            temporary.Add(image); temporary.Add(texture);
            foreach (UnityEngine.Object o in temporary) UnityEngine.Object.DestroyImmediate(o);
        }

        private static object Parse(Type type, string text)
        {
            if (type == typeof(int)) return int.Parse(text, CultureInfo.InvariantCulture);
            if (type == typeof(bool)) return text == "1" || text.ToLowerInvariant() == "true";
            if (type == typeof(float3))
            {
                string[] v = text.Split(',');
                return v.Length == 1 ? new float3(Number(v[0])) : new float3(Number(v[0]), Number(v[1]), Number(v[2]));
            }
            return Number(text);
        }

        private static float Number(string text) => float.Parse(text, CultureInfo.InvariantCulture);

        private static string Short(double distance) => distance > 1e30 ? "none" : distance.ToString("F4", CultureInfo.InvariantCulture);
    }
}

using System.Collections.Generic;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace SoftEngine.Testing
{
    // One body of a test scene, enough to build it for either engine
    public struct TestBody
    {
        public string name, mesh;
        public Vector3 position, euler, scale, velocity;
        public float mass, young, poisson, friction, damping;
        public bool isStatic;
        public bool placed;             // matrix is used as it is, instead of position, euler and scale
        public Matrix4x4 matrix;
    }

    /*
    The test scenes, as plain lists of bodies. SoftTestBridge runs them headless in a
    SoftSolver, SoftSceneBuilder turns them into GameObjects for either engine.
    */
    public static class SoftScenarios
    {
        public static readonly string[] Names =
        {
            "drop", "rest", "slide", "tower", "midair", "headon", "fast", "smash", "pile", "squeeze", "coils", "ramp", "stack", "onslab", "float", "bowl", "edge", "wake"
        };

        public static List<TestBody> Build(string name)
        {
            List<TestBody> b = new List<TestBody>();
            switch (name)
            {
                // A very soft ball lands on the edge of a stiff cube: it should wrap around the edge and come off again, not hang there
                case "edge":
                    b.Add(Body("cube", "Cube", new Vector3(0f, 0.475f, 0f), 0.9f, 5f, 1e6f));
                    b.Add(Body("ball", "Sphere", new Vector3(0.55f, 2.2f, 0.1f), 1.4f, 5f, 67f, velocity: new Vector3(0f, -5f, 0f)));
                    break;

                // Two cubes on top of each other fall asleep, then a ball knocks the lower one away: both have to wake, the upper one has to fall
                case "wake":
                    b.Add(Body("lower", "Cube", new Vector3(0f, 0.53f, 0f), 1f, 5f, 1e6f));
                    b.Add(Body("upper", "Cube", new Vector3(0f, 1.57f, 0f), 1f, 5f, 1e6f));
                    b.Add(Body("ball", "Sphere", new Vector3(-9f, 0.53f, 0f), 1f, 40f, 1e6f, velocity: new Vector3(10f, 0f, 0f), friction: 0.1f));
                    break;

                // A soft ball dropped on the floor
                case "drop":
                    b.Add(Body("ball", "Sphere", new Vector3(0f, 3f, 0f), 2f, 1f, 1e5f));
                    break;

                // A stiff cube lying on the floor, should stay still
                case "rest":
                    b.Add(Body("cube", "Cube", new Vector3(0f, 0.55f, 0f), 1f, 10f, 1e7f));
                    break;

                // A cube sliding at 3 m/s
                case "slide":
                    b.Add(Body("cube", "Cube", new Vector3(0f, 0.53f, 0f), 1f, 10f, 1e7f, velocity: new Vector3(3f, 0f, 0f)));
                    break;

                // Test1: four very soft balls stacked under a 10000 ton cylinder
                case "tower":
                    b.Add(Body("ball13", "Sphere", new Vector3(-11.4f, 8.21f, 6.59f), 2f, 1f, 67f, 0.48f));
                    b.Add(Body("ball14", "Sphere", new Vector3(-11.4f, 11.13f, 6.59f), 2f, 1f, 67f, 0.48f));
                    b.Add(Body("ball15", "Sphere", new Vector3(-11.4f, 13.97f, 6.59f), 2f, 1f, 67f, 0.48f));
                    b.Add(Body("ball16", "Sphere", new Vector3(-11.4f, 16.67f, 6.59f), 2f, 1f, 67f, 0.48f));
                    b.Add(Body("cylinder", "Cylinder", new Vector3(-11.34f, 19.13f, 6.58f), 1f, 1e7f, 1e12f));
                    break;

                // A heavy cube catches up with a light ball in the air, it should not be slowed
                case "midair":
                    b.Add(Body("ball", "Sphere", new Vector3(0f, 20f, 0f), 2f, 1f, 1e4f));
                    b.Add(Body("cube", "Cube", new Vector3(0f, 22.2f, 0f), 1f, 1000f, 1e7f, velocity: new Vector3(0f, -10f, 0f)));
                    break;

                // Two equal balls meet at 5 m/s each, no gravity
                case "headon":
                    b.Add(Body("left", "Sphere", new Vector3(-1.5f, 5f, 0f), 2f, 10f, 1e5f, velocity: new Vector3(5f, 0f, 0f)));
                    b.Add(Body("right", "Sphere", new Vector3(1.5f, 5f, 0.1f), 2f, 10f, 1e5f, velocity: new Vector3(-5f, 0f, 0f)));
                    break;

                // A stiff ball into the floor at 140 m/s
                case "fast":
                    b.Add(Body("ball", "Sphere", new Vector3(0f, 6f, 0f), 2f, 1f, 1e7f, velocity: new Vector3(0f, -140f, 0f)));
                    break;

                // A heavy cube at 80 m/s onto a light soft ball that lies on the floor
                case "smash":
                    b.Add(Body("ball", "Sphere", new Vector3(0f, 1.05f, 0f), 2f, 2f, 1e4f));
                    b.Add(Body("cube", "Cube", new Vector3(0f, 6f, 0f), 1f, 1000f, 1e7f, velocity: new Vector3(0f, -80f, 0f)));
                    break;

                // 27 mixed bodies falling into a heap
                case "pile":
                    for (int i = 0; i < 27; i++)
                    {
                        Vector3 p = new Vector3((i % 3 - 1) * 1.7f + 0.13f * (i / 9), 1.5f + (i / 9) * 1.9f, (i / 3 % 3 - 1) * 1.7f + 0.1f * (i % 3));
                        string mesh = i % 3 == 0 ? "Sphere" : i % 3 == 1 ? "Cube" : "Cylinder";
                        float scale = mesh == "Sphere" ? 1.4f : mesh == "Cube" ? 0.9f : 0.7f;
                        b.Add(Body(mesh.ToLower() + i, mesh, p, scale, 5f, i % 2 == 0 ? 1e5f : 1e6f, euler: new Vector3(17f * i, 31f * i, 0f)));
                    }
                    break;

                // A heavy stiff cube left lying on a soft ball: a pinch that never ends
                case "squeeze":
                    b.Add(Body("ball", "Sphere", new Vector3(0f, 1.05f, 0f), 2f, 1f, 1e3f, 0.45f));
                    b.Add(Body("cube", "Cube", new Vector3(0f, 2.9f, 0f), 1.5f, 5000f, 1e8f));
                    break;

                // The coil mesh: thin, concave, many triangles per cell
                case "coils":
                    b.Add(Body("coil0", "Coil", new Vector3(0f, 2f, 0f), 1f, 1f, 1e7f, euler: new Vector3(20f, 0f, 35f)));
                    b.Add(Body("coil1", "Coil", new Vector3(0.2f, 5f, 0.1f), 1f, 1f, 1e7f, euler: new Vector3(-30f, 50f, 0f)));
                    b.Add(Body("coil2", "Coil", new Vector3(-0.1f, 8f, 0.2f), 1f, 500f, 1e6f, euler: new Vector3(70f, 10f, 0f)));
                    b.Add(Body("ball", "Sphere", new Vector3(0.1f, 11f, 0f), 2f, 1f, 1e4f));
                    break;

                // A static slab tilted by 20 degrees: the cube should stay where it is (friction 0.5 holds up to 26 degrees), the ball should roll off
                case "ramp":
                    TestBody slab = Body("slab", "Cube", new Vector3(0f, 2f, 0f), 1f, 1f, 1e7f, euler: new Vector3(0f, 0f, 20f), isStatic: true);
                    slab.scale = new Vector3(8f, 0.5f, 4f);
                    b.Add(slab);
                    b.Add(Body("cube", "Cube", new Vector3(1.5f, 3.46f, -0.8f), 1f, 10f, 1e6f, euler: new Vector3(0f, 0f, 20f)));
                    b.Add(Body("ball", "Sphere", new Vector3(1.5f, 3.5f, 0.9f), 1f, 5f, 1e6f));
                    break;

                // Three cubes on top of each other, should stay still
                case "stack":
                    for (int i = 0; i < 3; i++)
                        b.Add(Body("cube" + i, "Cube", new Vector3(0f, 0.55f + 1.05f * i, 0f), 1f, 10f, 1e6f));
                    break;

                // A cube lying on a static slab, should stay still
                case "onslab":
                    TestBody ground = Body("slab", "Cube", new Vector3(0f, 1f, 0f), 1f, 1f, 1e7f, isStatic: true);
                    ground.scale = new Vector3(4f, 0.5f, 4f);
                    b.Add(ground);
                    b.Add(Body("cube", "Cube", new Vector3(0.07f, 1.8f, 0.04f), 1f, 10f, 1e6f));
                    break;

                // Nothing acts on these: no gravity, no floor, no contact. They should not move or change shape at all
                case "float":
                    b.Add(Body("softBall", "Sphere", new Vector3(-11.4f, 8.2f, 6.6f), 2f, 1f, 67f, 0.48f, euler: new Vector3(13f, 41f, 7f)));
                    b.Add(Body("jellyCube", "Cube", new Vector3(-7.3f, 8.1f, 6.3f), 1f, 10f, 1e4f, euler: new Vector3(17f, 31f, 5f)));
                    b.Add(Body("stiffCube", "Cube", new Vector3(-4.1f, 8.3f, 6.7f), 1f, 10f, 1e7f, euler: new Vector3(23f, 11f, 37f)));
                    b.Add(Body("lightStiffBall", "Sphere", new Vector3(-1.2f, 8.4f, 6.2f), 1f, 1f, 1e8f, euler: new Vector3(5f, 19f, 29f)));
                    break;

                // A ball dropped into a static bowl: the inside of the bowl bends inwards everywhere.
                // At rest the ball's centre should be 0.575 + skin above the bowl's lowest inner point (y = 0.55)
                case "bowl":
                    b.Add(Body("bowl", "Bowl", new Vector3(0f, 1.5f, 0f), 1f, 1f, 1e7f, isStatic: true));
                    b.Add(Body("ball", "Sphere", new Vector3(0.12f, 1.6f, 0.07f), 1.15f, 5f, 1e6f));
                    break;

                default:
                    return null;
            }
            return b;
        }

        /*
        The scene that is open in the Editor, as it would start in Play mode: every active
        SoftBody, and the settings of its SoftWorld.
        */
        public static List<TestBody> FromOpenScene(ref SoftSettings settings)
        {
            SoftWorld world = Object.FindFirstObjectByType<SoftWorld>();
            if (world != null) settings = world.settings;

            List<TestBody> b = new List<TestBody>();
            foreach (SoftBody body in Object.FindObjectsByType<SoftBody>(FindObjectsSortMode.InstanceID))
            {
                Mesh mesh = body.GetComponent<MeshFilter>().sharedMesh;
                meshes[mesh.name] = mesh;
                b.Add(new TestBody
                {
                    name = body.name.Replace(' ', '_'), mesh = mesh.name,
                    position = body.transform.position, euler = body.transform.eulerAngles, scale = body.transform.lossyScale, velocity = body.startVelocity,
                    mass = body.mass, young = body.youngsModulus, poisson = body.poissonRatio, friction = body.friction, damping = body.damping, isStatic = body.isStatic,
                    placed = true, matrix = body.transform.localToWorldMatrix
                });
            }
            return b;
        }

        private static TestBody Body(string name, string mesh, Vector3 position, float scale, float mass, float young,
            float poisson = 0.4f, Vector3 velocity = default, Vector3 euler = default, float friction = 0.5f, bool isStatic = false)
        {
            return new TestBody
            {
                name = name, mesh = mesh, position = position, euler = euler, scale = Vector3.one * scale, velocity = velocity,
                mass = mass, young = young, poisson = poisson, friction = friction, isStatic = isStatic
            };
        }

        // Scene-wide settings a scenario needs changed
        public static void Adjust(string name, ref SoftSettings settings)
        {
            if (name == "headon" || name == "float") { settings.gravity = float3.zero; settings.hasFloor = false; }
        }

        private static readonly Dictionary<string, Mesh> meshes = new Dictionary<string, Mesh>();

        // The same meshes the scenes use: Unity's primitives (Sphere, Cube, Cylinder, Capsule), the coil asset and the bowl made below
        public static Mesh FindMesh(string name)
        {
            if (meshes.TryGetValue(name, out Mesh found) && found != null) return found;

            if (name == "Coil")
            {
                string path = AssetDatabase.GUIDToAssetPath("22126b7ccf9d3b14580eeebc7a42bd64");
                foreach (Object asset in AssetDatabase.LoadAllAssetsAtPath(path))
                    if (asset is Mesh mesh) found = mesh;
            }
            else if (name == "Bowl") found = Bowl(1.2f, 0.95f, 32, 10);
            else
            {
                GameObject primitive = GameObject.CreatePrimitive((PrimitiveType)System.Enum.Parse(typeof(PrimitiveType), name));
                found = primitive.GetComponent<MeshFilter>().sharedMesh;
                Object.DestroyImmediate(primitive);
            }
            return meshes[name] = found;
        }

        /*
        Half a hollow ball, open at the top, the rim at y = 0. A line from the bottom of the
        outside up to the rim and back down the inside, turned around the y axis.
        */
        private static Mesh Bowl(float outer, float inner, int around, int along)
        {
            List<Vector2> line = new List<Vector2>();      // (distance from the axis, height)
            for (int i = 0; i <= along; i++)
            {
                float angle = 0.5f * Mathf.PI * i / along;
                line.Add(new Vector2(outer * Mathf.Sin(angle), -outer * Mathf.Cos(angle)));
            }
            for (int i = along; i >= 0; i--)
            {
                float angle = 0.5f * Mathf.PI * i / along;
                line.Add(new Vector2(inner * Mathf.Sin(angle), -inner * Mathf.Cos(angle)));
            }

            List<Vector3> verts = new List<Vector3>();
            List<int> tris = new List<int>();
            for (int i = 0; i < line.Count; i++)
            for (int k = 0; k < around; k++)
            {
                float turn = 2f * Mathf.PI * k / around;
                verts.Add(new Vector3(line[i].x * Mathf.Cos(turn), line[i].y, line[i].x * Mathf.Sin(turn)));
            }
            for (int i = 0; i + 1 < line.Count; i++)
            for (int k = 0; k < around; k++)
            {
                int a = i * around + k, b = i * around + (k + 1) % around, c = a + around, d = b + around;
                tris.AddRange(new[] { a, b, c, b, d, c });
            }

            Mesh mesh = new Mesh { name = "Bowl", hideFlags = HideFlags.HideAndDontSave };
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            mesh.RecalculateNormals();
            return mesh;
        }

        public static int AddTo(SoftSolver solver, TestBody body)
        {
            Mesh mesh = FindMesh(body.mesh);
            Matrix4x4 matrix = body.placed ? body.matrix : Matrix4x4.TRS(body.position, Quaternion.Euler(body.euler), body.scale);
            return solver.AddBody(mesh.vertices, mesh.triangles, matrix, body.mass, body.young, body.poisson, body.friction, body.isStatic, body.velocity, body.damping);
        }
    }
}

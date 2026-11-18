using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace SoftEngine.Testing
{
    /*
    Turns a test scenario into a scene with real GameObjects, for the old engine
    (PhysicsSolver + LatticeBody), the new one (SoftWorld + SoftBody) or both next to
    each other. Request, through SoftTestBridge:

        "<id> scene <scenario> <old|new|both> [save]"

    both: the old engine's copy is moved Spacing to the left, the new one to the right.
    save: written to Assets/Scenes/SoftEngine_<scenario>.unity, otherwise it stays an
    unsaved scene.

    The open scene is replaced. That is refused while it has unsaved changes, unless
    it is an unsaved scene this builder made itself (nothing is saved or thrown away
    on your behalf, and no dialog pops up).
    */
    public static class SoftSceneBuilder
    {
        private const float Spacing = 10f;
        private const string Folder = "Assets/Scenes/SoftEngine";
        private const string OldName = "PhysicsSolver (old engine)", NewName = "SoftWorld (new engine)";

        public static string Build(string[] parts)
        {
            string scenario = parts[2], which = parts.Length > 3 ? parts[3] : "both";
            bool save = parts.Length > 4 && parts[4] == "save";
            List<TestBody> bodies = SoftScenarios.Build(scenario);
            if (bodies == null) return "error unknown scenario " + scenario;
            if (EditorApplication.isPlaying) return "error stop Play mode first";
            if (UnsavedWork()) return "error the open scene has unsaved changes, save it first";

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

            SoftSettings settings = SoftSettings.Default;
            SoftScenarios.Adjust(scenario, ref settings);
            bool old = which != "new", fresh = which != "old";
            float shift = old && fresh ? Spacing : 0f;

            Bounds all = new Bounds(bodies[0].position, Vector3.zero);
            foreach (TestBody body in bodies) all.Encapsulate(new Bounds(body.position, body.scale * 2f));

            if (settings.hasFloor)
            {
                GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
                floor.name = "Floor (visual only)";
                floor.transform.position = new Vector3(all.center.x, 0f, all.center.z);
                floor.transform.localScale = Vector3.one * 6f;
                Object.DestroyImmediate(floor.GetComponent<Collider>());
            }

            if (old)
            {
                PhysicsSolver solver = new GameObject(OldName).AddComponent<PhysicsSolver>();
                solver.substeps = settings.substeps;
                solver.targetLatticeDimensions = settings.cellSize;
                solver.gravity = settings.gravity;

                Material material = MaterialAsset("OldEngine", new Color(0.95f, 0.6f, 0.2f));
                foreach (TestBody body in bodies)
                {
                    GameObject go = Create(body, "old " + body.name, -shift, material);
                    LatticeBody lattice = go.AddComponent<LatticeBody>();
                    lattice.mass = body.mass;
                    lattice.youngsModulus = body.young;
                    lattice.poissonRatio = body.poisson;
                    if (body.velocity != Vector3.zero) go.AddComponent<OldEngineStartVelocity>().velocity = body.velocity;
                }
            }

            if (fresh)
            {
                SoftWorld world = new GameObject(NewName).AddComponent<SoftWorld>();
                world.settings = settings;

                Material material = MaterialAsset("NewEngine", new Color(0.3f, 0.65f, 1f));
                foreach (TestBody body in bodies)
                {
                    GameObject go = Create(body, "new " + body.name, shift, material);
                    SoftBody soft = go.AddComponent<SoftBody>();
                    soft.mass = body.mass;
                    soft.youngsModulus = body.young;
                    soft.poissonRatio = body.poisson;
                    soft.friction = body.friction;
                    soft.isStatic = body.isStatic;
                    soft.startVelocity = body.velocity;
                }
            }

            Camera camera = Camera.main;
            if (camera != null)
            {
                Vector3 target = new Vector3(all.center.x, all.center.y * 0.5f, all.center.z);
                camera.transform.position = target + new Vector3(0f, all.extents.y * 0.6f + 2f, -(all.extents.magnitude * 1.6f + shift * 2f + 6f));
                camera.transform.LookAt(target);
            }

            string path = "unsaved";
            if (save)
            {
                path = $"Assets/Scenes/SoftEngine_{scenario}.unity";
                EditorSceneManager.SaveScene(scene, path);
            }
            return $"ok {scenario} {which} bodies={bodies.Count} scene={path}";
        }

        // True when replacing the open scenes would lose something: unsaved changes in anything but a scene built here and never saved
        public static bool UnsavedWork()
        {
            for (int i = 0; i < SceneManager.sceneCount; i++)
            {
                Scene scene = SceneManager.GetSceneAt(i);
                if (!scene.isDirty) continue;

                bool builtHere = false;
                foreach (GameObject root in scene.GetRootGameObjects())
                    builtHere |= root.name == OldName || root.name == NewName;
                if (!builtHere || !string.IsNullOrEmpty(scene.path)) return true;
            }
            return false;
        }

        private static GameObject Create(TestBody body, string name, float shiftX, Material material)
        {
            GameObject go = new GameObject(name);
            go.transform.SetPositionAndRotation(body.position + new Vector3(shiftX, 0f, 0f), Quaternion.Euler(body.euler));
            go.transform.localScale = body.scale;
            go.AddComponent<MeshFilter>().sharedMesh = SoftScenarios.FindMesh(body.mesh);
            go.AddComponent<MeshRenderer>().sharedMaterial = material;
            return go;
        }

        // A plain coloured material, kept as an asset so saved scenes can use it
        private static Material MaterialAsset(string name, Color colour)
        {
            string path = $"{Folder}/{name}.mat";
            Material material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material != null) return material;

            Directory.CreateDirectory(Folder);
            material = new Material(GraphicsSettings.currentRenderPipeline != null
                ? GraphicsSettings.currentRenderPipeline.defaultShader : Shader.Find("Standard")) { color = colour };
            AssetDatabase.CreateAsset(material, path);
            return material;
        }
    }
}

using System.Collections.Generic;
using UnityEngine;

namespace SoftEngine
{
    /*
    One per scene. Owns the SoftSolver, steps it every FixedUpdate and writes the
    result back into the meshes of the SoftBody components that registered.
    */
    [DefaultExecutionOrder(-100)]
    public class SoftWorld : MonoBehaviour
    {
        public SoftSettings settings = SoftSettings.Default;
        [Tooltip("Draw the bodies in between two physics steps")]
        public bool smooth = true;

        public SoftSolver Solver { get; private set; }
        private readonly List<SoftBody> bodies = new List<SoftBody>();
        private bool stepped;       // there was a physics step since the bodies were last drawn

        private void Awake()
        {
            Solver = new SoftSolver(settings);
        }

        private void OnDestroy()
        {
            Solver?.Dispose();
            Solver = null;
        }

        // Called by SoftBody.Start
        public void Add(SoftBody body)
        {
            Mesh mesh = body.GetComponent<MeshFilter>().mesh;
            mesh.MarkDynamic();

            Transform t = body.transform;
            body.Index = Solver.AddBody(mesh.vertices, mesh.triangles, t.localToWorldMatrix,
                body.mass, body.youngsModulus, body.poissonRatio, body.friction, body.isStatic, body.startVelocity, body.damping);
            body.Mesh = mesh;
            bodies.Add(body);

            // From here on the vertices are written in world axes around the body's centre (also under a scaled parent)
            t.rotation = Quaternion.identity;
            t.localScale = Vector3.one;
            Vector3 left = t.lossyScale;
            t.localScale = new Vector3(1f / left.x, 1f / left.y, 1f / left.z);
        }

        private void FixedUpdate()
        {
            float cellSize = Solver.settings.cellSize;      // the lattices are already built with it
            Solver.settings = settings;
            Solver.settings.cellSize = cellSize;

            Solver.Step(Time.fixedDeltaTime);
            Solver.UpdateVisuals();
            stepped = true;
        }

        /*
        Draws the bodies. With smooth on, every frame shows them on their way from
        the physics step before to the last one, so they also move evenly when the
        screen is faster than the physics (and one step behind, which nobody sees).
        */
        private void Update()
        {
            if (Solver == null || bodies.Count == 0) return;
            if (!Solver.VisualsKnown)       // a body was added and there has been no step yet
            {
                Solver.UpdateVisuals();
                stepped = true;
            }
            if (!smooth && !stepped) return;

            float blend = smooth ? Mathf.Clamp01((Time.time - Time.fixedTime) / Time.fixedDeltaTime) : 1f;
            Solver.BlendVisuals(blend);

            foreach (SoftBody body in bodies)
            {
                // A sleeping body does not change any more once it has been drawn where it fell asleep
                bool asleep = Solver.bodyAsleep[body.Index];
                if (stepped) body.StepsAsleep = asleep ? body.StepsAsleep + 1 : 0;
                if (body.StepsAsleep > 2) continue;

                BodyRange range = Solver.bodies[body.Index];
                body.transform.position = Solver.BlendCentre(body.Index, blend);
                body.Mesh.SetVertices(Solver.visDrawn.AsArray(), range.visStart, range.visCount);
                body.Mesh.RecalculateBounds();
                body.Mesh.RecalculateNormals();
            }
            stepped = false;
        }

        private void OnDrawGizmosSelected()
        {
            if (!settings.hasFloor) return;
            Gizmos.color = new Color(0.3f, 0.8f, 1f, 0.5f);
            Gizmos.DrawWireCube(new Vector3(0f, settings.floorY, 0f), new Vector3(50f, 0f, 50f));
        }
    }
}

#if UNITY_EDITOR
using System.Collections.Generic;
using System.Reflection;
using SolverStructs;
using Unity.Collections;
using UnityEngine;

namespace SoftEngine.Testing
{
    /*
    Test scenes only: gives a LatticeBody of the old engine a start velocity, so both
    engines can run the same scene. The old solver keeps velocity as the difference
    between a particle's current and previous position, so that is what gets set,
    through reflection, PhysicsSolver itself is not touched.
    */
    [DefaultExecutionOrder(1000)]
    public class OldEngineStartVelocity : MonoBehaviour
    {
        public Vector3 velocity;

        private void Start()
        {
            PhysicsSolver solver = FindFirstObjectByType<PhysicsSolver>();
            if (solver == null) return;

            const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;
            List<BodyWraper> wrappers = (List<BodyWraper>)typeof(PhysicsSolver).GetField("bodyWrapers", Private).GetValue(solver);
            NativeList<Particle> particles = (NativeList<Particle>)typeof(PhysicsSolver).GetField("particles", Private).GetValue(solver);
            float dt = Time.fixedDeltaTime / solver.substeps;

            foreach (BodyWraper wrapper in wrappers)
            {
                if (wrapper.transform != transform) continue;
                for (int i = wrapper.particlesRange.x; i < wrapper.particlesRange.y; i++)
                {
                    Particle p = particles[i];
                    p.previousPosition = p.currentPosition - (Unity.Mathematics.float3)(velocity * dt);
                    particles[i] = p;
                }
            }
        }
    }
}
#endif

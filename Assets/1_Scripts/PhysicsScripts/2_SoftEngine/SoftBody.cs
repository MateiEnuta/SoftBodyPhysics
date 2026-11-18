using UnityEngine;

namespace SoftEngine
{
    /*
    Makes the object's mesh a soft body of the scene's SoftWorld. The mesh is both
    what is drawn and what collides.
    */
    [RequireComponent(typeof(MeshFilter))]
    public class SoftBody : MonoBehaviour
    {
        public float mass = 1f;
        public float youngsModulus = 100000f;
        [Range(0f, 0.49f)] public float poissonRatio = 0.4f;
        public float friction = 0.5f;
        [Tooltip("Seconds until most of a wobble is gone: 0 = never, 0.5 = jelly, 0.05 = settles at once. Does not slow falling or spinning.")]
        public float damping = 0.1f;
        public bool isStatic = false;
        public Vector3 startVelocity;

        public int Index { get; set; } = -1;
        public Mesh Mesh { get; set; }
        public int StepsAsleep { get; set; }

        private void Start()
        {
            SoftWorld world = FindFirstObjectByType<SoftWorld>();
            if (world != null) world.Add(this);
        }
    }
}

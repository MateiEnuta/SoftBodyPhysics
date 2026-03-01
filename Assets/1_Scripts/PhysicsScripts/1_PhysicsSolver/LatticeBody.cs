using UnityEngine;

public class LatticeBody : MonoBehaviour
{
    public Mesh collisionMesh;
    public float mass = 1;

    private PhysicsSolver physicsSolver;
    [HideInInspector] public Mesh visualMesh { get; private set; }

    private void Start()
    {
        visualMesh = GetComponent<MeshFilter>().sharedMesh;

        physicsSolver = FindFirstObjectByType<PhysicsSolver>();
        physicsSolver.ProcessLatticeBody(this);
    }
}

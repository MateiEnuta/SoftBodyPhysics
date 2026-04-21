using UnityEngine;

public class LatticeBody : MonoBehaviour
{
    public Mesh collisionMesh;
    public float mass = 1;
    public float distanceCompliance = 0f;
    public float volumeCompliance = 0f;

    private PhysicsSolver physicsSolver;
    [HideInInspector] public Mesh visualMesh { get; private set; }

    private void Start()
    {
        Initialize();
    }

    public void Initialize()
    {
        visualMesh = GetComponent<MeshFilter>().mesh;

        physicsSolver = FindFirstObjectByType<PhysicsSolver>();
        if (physicsSolver == null) return;
        physicsSolver.ProcessLatticeBody(this);
    }
}

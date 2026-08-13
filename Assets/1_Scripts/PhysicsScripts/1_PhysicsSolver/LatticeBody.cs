using UnityEngine;

public class LatticeBody : MonoBehaviour
{
    public Mesh collisionMesh;
    public float mass = 1;
    public float youngsModulus = 1f;
    public float poissonRatio = 0.4f;

    private PhysicsSolver physicsSolver;
    [HideInInspector] public Mesh visualMesh { get; private set; }
    [HideInInspector] public float mu;
    [HideInInspector] public float lambda;

    private void Start()
    {
        Initialize();
    }

    public void Initialize()
    {
        mu     = youngsModulus / (2f * (1f + poissonRatio));
        lambda = (youngsModulus * poissonRatio) /
                    ((1f + poissonRatio) * (1f - 2f * poissonRatio));

        visualMesh = GetComponent<MeshFilter>().mesh;

        physicsSolver = FindFirstObjectByType<PhysicsSolver>();
        if (physicsSolver == null) return;
        physicsSolver.ProcessLatticeBody(this);
    }
}

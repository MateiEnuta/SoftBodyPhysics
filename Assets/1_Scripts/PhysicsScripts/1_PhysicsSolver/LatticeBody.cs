using UnityEngine;

public class LatticeBody : MonoBehaviour
{
    public Mesh collisionMesh;

    private PhysicsSolver physicsSolver;

    [HideInInspector] public Mesh visualMesh { get; private set; }

    private void Start()
    {
        visualMesh = GetComponent<MeshFilter>().sharedMesh;

        physicsSolver = FindFirstObjectByType<PhysicsSolver>();
        physicsSolver.latticePreprocessor.ProcessLatticeBody(this);
    }
}

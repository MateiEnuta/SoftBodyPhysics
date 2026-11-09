using System;
using System.IO;
using UnityEditor;
using UnityEditor.Compilation;
using UnityEngine;

/*
Lets Claude drive the open Editor through two files in Temp/ClaudeBridge:

    request.txt     "<id> <command> <args>", deleted when it is read
    response.txt    "<id>" on the first line, then the answer

Commands:
    ping                     Editor state
    refresh                  import changed files and recompile, answers with compile errors
    play                     enter Play mode paused, with a ClaudeRecorder in the scene
    step <n>                 advance until n more FixedUpdates ran, stops early on NaN
    snapshot <name>          every particle and cell to <name>_particles.csv / _cells.csv
    check                    real distance between the meshes of different bodies, crossings
    shot <name> [px py pz tx ty tz]   PNG from the main camera or from p looking at t
    stop                     leave Play mode

Only these fixed commands exist. Nothing happens unless a request file appears,
and Play mode started by hand is left alone.
*/
[InitializeOnLoad]
public static class ClaudeBridge
{
    public const string Dir = "Temp/ClaudeBridge";
    private const string RequestPath = Dir + "/request.txt";
    private const string ResponsePath = Dir + "/response.txt";
    private const string ErrorsPath = Dir + "/compile_errors.txt";
    private const string ConsolePath = Dir + "/console.txt";

    // Survive the domain reloads of recompiling and entering Play mode
    private const string PendingId = "ClaudeBridge.PendingId";
    private const string PendingKind = "ClaudeBridge.PendingKind";
    private const string PendingSince = "ClaudeBridge.PendingSince";
    private const string RecordKey = "ClaudeBridge.Record";

    private const double PollSeconds = 0.05;
    private const double RefreshSettleSeconds = 1.0;
    private const int MaxConsoleLines = 300;

    private static double nextPoll;
    private static string stepId;
    private static int stepGoal;
    private static int stepFrame = -1;
    private static int consoleLines;
    private static readonly object consoleLock = new object();


    static ClaudeBridge()
    {
        Directory.CreateDirectory(Dir);
        EditorApplication.update += Tick;
        EditorApplication.playModeStateChanged += OnPlayModeChanged;
        CompilationPipeline.assemblyCompilationFinished += OnAssemblyCompiled;
        Application.logMessageReceivedThreaded += OnLog;
    }

    private static void Tick()
    {
        if (EditorApplication.timeSinceStartup < nextPoll) return;
        nextPoll = EditorApplication.timeSinceStartup + PollSeconds;

        try
        {
            FinishPending();
            AdvanceSteps();
            ReadRequest();
        }
        catch (Exception e)
        {
            Respond(SessionState.GetString(PendingId, "0"), "error " + e.Message);
            SessionState.EraseString(PendingKind);
        }
    }

    private static void ReadRequest()
    {
        if (!File.Exists(RequestPath)) return;

        string[] parts = File.ReadAllText(RequestPath).Trim().Split(' ');
        File.Delete(RequestPath);
        if (parts.Length < 2) return;

        string id = parts[0];
        if (stepId != null || SessionState.GetString(PendingKind, "") != "")
        {
            Respond(id, "error busy");
            return;
        }

        ClaudeRecorder recorder = ClaudeRecorder.Instance;
        switch (parts[1])
        {
            case "ping":
                Respond(id, $"ok playing={EditorApplication.isPlaying} paused={EditorApplication.isPaused} compiling={EditorApplication.isCompiling} focused={UnityEditorInternal.InternalEditorUtility.isApplicationActive}");
                break;

            case "refresh":
                if (EditorApplication.isPlaying) { Respond(id, "error stop Play mode first"); break; }
                File.Delete(ErrorsPath);
                SetPending(id, "refresh");
                AssetDatabase.Refresh();
                break;

            case "play":
                if (EditorApplication.isPlaying) { Respond(id, "error already playing"); break; }
                if (EditorApplication.isCompiling) { Respond(id, "error compiling"); break; }
                File.Delete(ConsolePath);
                SessionState.SetBool(RecordKey, true);
                SetPending(id, "play");
                EditorApplication.isPaused = true;
                EditorApplication.isPlaying = true;
                break;

            case "step":
                if (recorder == null) { Respond(id, "error not playing"); break; }
                stepId = id;
                stepGoal = recorder.FixedSteps + int.Parse(parts[2]);
                stepFrame = -1;
                break;

            case "snapshot":
                if (recorder == null) { Respond(id, "error not playing"); break; }
                Respond(id, "ok " + recorder.Snapshot(parts[2]));
                break;

            case "check":
                if (recorder == null) { Respond(id, "error not playing"); break; }
                Respond(id, "ok " + recorder.Check());
                break;

            case "shot":
                if (recorder == null) { Respond(id, "error not playing"); break; }
                Respond(id, "ok " + recorder.Shot(parts[2], parts));
                break;

            case "stop":
                if (!EditorApplication.isPlaying) { Respond(id, "ok already stopped"); break; }
                SetPending(id, "stop");
                EditorApplication.isPlaying = false;
                break;

            default:
                Respond(id, "error unknown command " + parts[1]);
                break;
        }
    }

    private static void AdvanceSteps()
    {
        if (stepId == null) return;

        ClaudeRecorder recorder = ClaudeRecorder.Instance;
        if (recorder == null || !EditorApplication.isPlaying)
        {
            Respond(stepId, "error Play mode ended");
            stepId = null;
            return;
        }

        if (stepFrame != -1)
        {
            if (Time.frameCount == stepFrame) return;     // the requested frame has not run yet
            if (recorder.FixedSteps >= stepGoal || recorder.NanCount > 0)
            {
                Respond(stepId, "ok " + recorder.LastLine);
                stepId = null;
                return;
            }
        }

        ClaudeRecorder.StepClock.Restart();
        stepFrame = Time.frameCount;
        EditorApplication.isPaused = true;
        EditorApplication.Step();
    }

    private static void FinishPending()
    {
        string kind = SessionState.GetString(PendingKind, "");
        if (kind == "") return;

        if (kind == "refresh")
        {
            double since = EditorApplication.timeSinceStartup - SessionState.GetFloat(PendingSince, 0f);
            if (since < RefreshSettleSeconds || EditorApplication.isCompiling || EditorApplication.isUpdating) return;
            FinishWith(File.Exists(ErrorsPath) ? "error compile\n" + File.ReadAllText(ErrorsPath) : "ok compiled");
        }
        else if (kind == "play")
        {
            ClaudeRecorder recorder = ClaudeRecorder.Instance;
            if (!EditorApplication.isPlaying || recorder == null || !EditorApplication.isPaused) return;
            FinishWith("ok " + recorder.Describe());
        }
        else if (kind == "stop")
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;
            FinishWith("ok stopped");
        }
    }

    private static void OnPlayModeChanged(PlayModeStateChange state)
    {
        if (state == PlayModeStateChange.EnteredPlayMode && SessionState.GetBool(RecordKey, false))
        {
            Application.runInBackground = true;
            EditorApplication.isPaused = true;
            new GameObject("ClaudeRecorder").AddComponent<ClaudeRecorder>();
        }
        else if (state == PlayModeStateChange.EnteredEditMode)
        {
            SessionState.EraseBool(RecordKey);
        }
    }

    private static void OnAssemblyCompiled(string assembly, CompilerMessage[] messages)
    {
        foreach (CompilerMessage message in messages)
            if (message.type == CompilerMessageType.Error)
                File.AppendAllText(ErrorsPath, message.message + "\n");
    }

    private static void OnLog(string text, string stack, LogType type)
    {
        if (type == LogType.Log || type == LogType.Warning) return;

        lock (consoleLock)
        {
            if (consoleLines++ >= MaxConsoleLines) return;
            File.AppendAllText(ConsolePath, $"[{type}] {text}\n{stack}\n");
        }
    }

    private static void SetPending(string id, string kind)
    {
        SessionState.SetString(PendingId, id);
        SessionState.SetString(PendingKind, kind);
        SessionState.SetFloat(PendingSince, (float)EditorApplication.timeSinceStartup);
    }

    private static void FinishWith(string message)
    {
        Respond(SessionState.GetString(PendingId, "0"), message);
        SessionState.EraseString(PendingKind);
    }

    private static void Respond(string id, string message)
    {
        File.WriteAllText(ResponsePath + ".tmp", id + "\n" + message + "\n");
        File.Delete(ResponsePath);
        File.Move(ResponsePath + ".tmp", ResponsePath);
    }
}

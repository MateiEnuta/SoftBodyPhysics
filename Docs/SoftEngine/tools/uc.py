#!/usr/bin/env python3
"""Client for the Unity Editor bridge: python3 uc.py [-t seconds] command args..."""
import os, sys, time
D = "/Users/enutamatei/GitHub/SoftBodyPhysics/Temp/ClaudeBridge"
REQ, RES = D + "/request.txt", D + "/response.txt"

def call(cmd, timeout=60):
    rid = str(int(time.time() * 1000) % 100000000)
    try: os.remove(RES)
    except FileNotFoundError: pass
    with open(REQ + ".tmp", "w") as f: f.write(rid + " " + cmd)
    os.replace(REQ + ".tmp", REQ)
    end = time.time() + timeout
    while time.time() < end:
        try:
            with open(RES) as f: text = f.read()
            lines = text.split("\n", 1)
            if lines[0].strip() == rid: return lines[1].strip() if len(lines) > 1 else ""
        except FileNotFoundError: pass
        time.sleep(0.05)
    return "TIMEOUT"

if __name__ == "__main__":
    a = sys.argv[1:]
    t = 60
    if a and a[0] == "-t": t = float(a[1]); a = a[2:]
    print(call(" ".join(a), t))

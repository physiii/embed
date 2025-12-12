#!/usr/bin/env python3
"""
GPU / VRAM audit helper.

What it does:
  - shows which GPUs exist and their current VRAM usage
  - shows which processes (and docker containers, if applicable) are using VRAM
  - optionally enriches with embedding-service weight/parameter size via /gpu_report

Usage:
  python3 gpu_audit.py
  python3 gpu_audit.py --json
"""

import argparse
import json
import os
import re
import subprocess
import urllib.request
from typing import Optional, Dict, Any, List


def sh(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def try_sh(cmd: list[str]) -> Optional[str]:
    try:
        return sh(cmd)
    except Exception:
        return None


def parse_csv_lines(s: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([p.strip() for p in line.split(",")])
    return rows


def get_gpu_index_by_uuid() -> Dict[str, Dict[str, Any]]:
    out = sh(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"])
    m: Dict[str, Dict[str, Any]] = {}
    for idx, uuid, name, total, used in parse_csv_lines(out):
        m[uuid] = {
            "gpu_index": int(idx),
            "gpu_uuid": uuid,
            "gpu_name": name,
            "mem_total_mib": int(total),
            "mem_used_mib": int(used),
        }
    return m


def get_compute_apps() -> List[Dict[str, Any]]:
    out = sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    apps: List[Dict[str, Any]] = []
    for gpu_uuid, pid, pname, used in parse_csv_lines(out):
        apps.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": int(pid),
                "process_name": pname,
                "used_mib": int(used),
            }
        )
    return apps


def docker_id_from_pid(pid: int) -> Optional[str]:
    try:
        cg = open(f"/proc/{pid}/cgroup", "r", encoding="utf-8").read()
    except Exception:
        return None
    m = re.search(r"docker-([0-9a-f]{12,64})\.scope", cg)
    return m.group(1) if m else None


def docker_id_to_name() -> Dict[str, str]:
    out = try_sh(["docker", "ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}"])
    if not out:
        return {}
    m: Dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        cid, name = line.split(" ", 1)
        m[cid] = name.strip()
    return m


def fetch_json(url: str, timeout_s: float = 2.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    ap.add_argument("--embed-url", default="http://localhost:8000/gpu_report", help="Embedding service /gpu_report URL")
    args = ap.parse_args()

    gpu_by_uuid = get_gpu_index_by_uuid()
    apps = get_compute_apps()
    cid_to_name = docker_id_to_name()

    enriched = []
    for a in apps:
        gpu = gpu_by_uuid.get(a["gpu_uuid"], {})
        pid = a["pid"]
        docker_id = docker_id_from_pid(pid)
        container = None
        if docker_id:
            # docker ps shows full IDs; handle short IDs by prefix match just in case
            container = cid_to_name.get(docker_id)
            if not container:
                for full, name in cid_to_name.items():
                    if full.startswith(docker_id):
                        container = name
                        break
        enriched.append({**a, **gpu, "docker_id": docker_id, "container": container})

    embed_report = fetch_json(args.embed_url)

    report = {
        "host_pid": os.getpid(),
        "gpus": list(gpu_by_uuid.values()),
        "compute_apps": enriched,
        "embedding_service_gpu_report": embed_report,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    # Text output
    print("GPUs:")
    for g in sorted(report["gpus"], key=lambda x: x["gpu_index"]):
        print(
            f"  GPU{g['gpu_index']}: {g['gpu_name']}  used={g['mem_used_mib']}MiB / total={g['mem_total_mib']}MiB  uuid={g['gpu_uuid']}"
        )

    print("\nProcesses using VRAM:")
    for a in sorted(enriched, key=lambda x: (x.get("gpu_index", 999), -x["used_mib"])):
        who = a.get("container") or a["process_name"]
        print(f"  GPU{a.get('gpu_index','?')}: {who}  pid={a['pid']}  used={a['used_mib']}MiB")

    if embed_report:
        w = embed_report.get("weights", {})
        cuda = embed_report.get("cuda", {})
        print("\nEmbedding service (/gpu_report):")
        print(f"  model_id={embed_report.get('model_id')}  device={embed_report.get('device')}  quantization={embed_report.get('quantization')}")
        print(
            f"  weights: params={w.get('param_mib')}MiB buffers={w.get('buffer_mib')}MiB"
        )
        if cuda.get("available"):
            print(
                f"  torch: allocated={cuda.get('torch_allocated_mib')}MiB reserved={cuda.get('torch_reserved_mib')}MiB device_index={cuda.get('device_index')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



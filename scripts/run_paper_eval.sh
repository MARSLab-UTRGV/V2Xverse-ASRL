#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_PATH="${REPO_ROOT}/experiments/paper_eval_town05_main.yaml"
DRY_RUN=0
FORCE=0
METHOD_FILTER=""
START_PASS=1
CONTINUE_MODE=0

usage() {
  cat <<'USAGE'
Usage: scripts/run_paper_eval.sh [options]

Options:
  --config PATH      Run-matrix config yaml (default: experiments/paper_eval_town05_main.yaml)
  --method NAME      Run only one method name from config
  --start-pass N     Start from pass N (default: 1)
  --continue         Continue mode: skip finished passes and resume unfinished pass checkpoints
  --dry-run          Print resolved commands and manifests, do not execute evaluator
  --force            Overwrite existing pass results for this experiment/method
  -h, --help         Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --method)
      METHOD_FILTER="$2"
      shift 2
      ;;
    --start-pass)
      START_PASS="$2"
      shift 2
      ;;
    --continue)
      CONTINUE_MODE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

python3 - "$REPO_ROOT" "$CONFIG_PATH" "$METHOD_FILTER" "$DRY_RUN" "$FORCE" "$START_PASS" "$CONTINUE_MODE" <<'PY'
import copy
import datetime
import json
import os
import shutil
import subprocess
import sys

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml (import yaml failed)") from exc


repo_root, config_path, method_filter, dry_run_s, force_s, start_pass_s, continue_mode_s = sys.argv[1:]
dry_run = dry_run_s == "1"
force = force_s == "1"
continue_mode = continue_mode_s == "1"
start_pass = int(start_pass_s)
if start_pass <= 0:
    raise ValueError("--start-pass must be >= 1")


def _abspath(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root, path))


def _require(cfg, key):
    if key not in cfg:
        raise ValueError("Missing required key: {}".format(key))
    return cfg[key]


def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)


def _read_entry_status(results_json_path):
    try:
        with open(results_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("entry_status", "")).strip()
    except Exception:
        return ""


with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not isinstance(cfg, dict):
    raise ValueError("Top-level config must be a mapping")

exp_cfg = _require(cfg, "experiment")
runtime_cfg = _require(cfg, "runtime")
methods_cfg = _require(cfg, "methods")

exp_name = _require(exp_cfg, "name")
output_root_rel = _require(exp_cfg, "output_root")
reference_method = exp_cfg.get("reference_method", "")

if not isinstance(methods_cfg, list) or not methods_cfg:
    raise ValueError("methods must be a non-empty list")

required_runtime_keys = [
    "host", "port", "traffic_manager_port", "timeout", "track", "repetitions",
    "best_every", "reuse_agent", "carla_seed", "traffic_seed", "route_id_start",
    "num_routes", "route_passes", "route_id_max", "routes_dir", "routes_pattern",
    "routes_file", "scenarios_file", "scenario_parameter_file", "team_agent",
    "debug", "resume", "skip_existed",
]
for key in required_runtime_keys:
    _require(runtime_cfg, key)

route_passes = int(runtime_cfg["route_passes"])
if route_passes <= 0:
    raise ValueError("runtime.route_passes must be >= 1")
if start_pass > route_passes:
    raise ValueError("--start-pass={} is greater than runtime.route_passes={}".format(start_pass, route_passes))

exp_root = _abspath(os.path.join(output_root_rel, exp_name))
os.makedirs(exp_root, exist_ok=True)

manifest = {
    "experiment": exp_name,
    "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "config_path": os.path.abspath(config_path),
    "reference_method": reference_method,
    "runtime": runtime_cfg,
    "start_pass": start_pass,
    "continue_mode": continue_mode,
    "methods": [],
}

carla_root = os.environ.get("CARLA_ROOT", "external_paths/carla_root")
leaderboard_root = os.environ.get("LEADERBOARD_ROOT", "simulation/leaderboard")
scenario_runner_root = os.environ.get("SCENARIO_RUNNER_ROOT", "simulation/scenario_runner")
data_root = os.environ.get("DATA_ROOT", "external_paths/data_root")

for method in methods_cfg:
    method_name = _require(method, "name")
    if method_filter and method_name != method_filter:
        continue

    base_cfg_rel = _require(method, "base_agent_config")
    overrides = method.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("method overrides must be a mapping for {}".format(method_name))

    base_cfg_path = _abspath(base_cfg_rel)
    if not os.path.exists(base_cfg_path):
        raise FileNotFoundError("Base agent config not found: {}".format(base_cfg_path))

    with open(base_cfg_path, "r", encoding="utf-8") as f:
        agent_cfg = yaml.safe_load(f)
    if not isinstance(agent_cfg, dict):
        raise ValueError("Agent config must be a mapping: {}".format(base_cfg_path))

    deep_update(agent_cfg, overrides)

    method_root = os.path.join(exp_root, method_name)
    method_manifest_dir = os.path.join(method_root, "manifest")
    os.makedirs(method_manifest_dir, exist_ok=True)

    # Ensure deterministic RL log location in this method tree.
    if isinstance(agent_cfg.get("rl"), dict):
        agent_cfg["rl"]["log_dir"] = os.path.join(method_root, "rl")

    # Guard eval-only usage against random-policy runs.
    rl_cfg = agent_cfg.get("rl", {}) if isinstance(agent_cfg.get("rl"), dict) else {}
    if rl_cfg.get("enabled", False) and rl_cfg.get("eval_only", False):
        resume_path = rl_cfg.get("resume_path", None)
        if not resume_path:
            raise ValueError(
                "Method '{}' has rl.eval_only=true but no rl.resume_path".format(method_name)
            )
        resume_abs = _abspath(str(resume_path))
        if not os.path.exists(resume_abs):
            raise FileNotFoundError(
                "Method '{}' resume checkpoint not found: {}".format(method_name, resume_abs)
            )

    agent_cfg_out = os.path.join(method_manifest_dir, "agent_config_eval.yaml")
    with open(agent_cfg_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(agent_cfg, f, sort_keys=False)

    shutil.copy2(config_path, os.path.join(method_manifest_dir, "matrix_snapshot.yaml"))

    method_entry = {
        "name": method_name,
        "base_agent_config": base_cfg_path,
        "effective_agent_config": agent_cfg_out,
        "pass_runs": [],
    }

    for pass_idx in range(start_pass, route_passes + 1):
        pass_tag = "pass{:02d}".format(pass_idx)
        run_root = os.path.join(method_root, pass_tag)
        run_eval_root = os.path.join(run_root, "eval")
        run_images_root = os.path.join(run_root, "images")
        run_rl_root = os.path.join(run_root, "rl")
        run_manifest_root = os.path.join(run_root, "manifest")
        os.makedirs(run_eval_root, exist_ok=True)
        os.makedirs(run_images_root, exist_ok=True)
        os.makedirs(run_rl_root, exist_ok=True)
        os.makedirs(run_manifest_root, exist_ok=True)

        checkpoint_endpoint = os.path.join(run_eval_root, "results.json")
        method_results = os.path.join(run_eval_root, "ego_vehicle_0", "results.json")
        resume_flag = int(runtime_cfg["resume"])
        existing_entry_status = ""

        if continue_mode:
            if os.path.exists(method_results):
                existing_entry_status = _read_entry_status(method_results)
                if existing_entry_status == "Finished" and not force:
                    print("[paper-eval] {} {} already finished, skipping.".format(method_name, pass_tag))
                    method_entry["pass_runs"].append(
                        {
                            "pass_index": pass_idx,
                            "run_root": run_root,
                            "checkpoint_endpoint": checkpoint_endpoint,
                            "results_json": method_results,
                            "status": "skipped_finished",
                        }
                    )
                    continue
                resume_flag = 1
            elif os.path.exists(checkpoint_endpoint):
                resume_flag = 1

        if os.path.exists(method_results) and not force and not continue_mode:
            raise FileExistsError(
                "Existing results found (use --force to overwrite): {}".format(method_results)
            )

        cmd = [
            "python",
            os.path.join(leaderboard_root, "leaderboard", "leaderboard_evaluator_rl.py"),
            "--host={}".format(runtime_cfg["host"]),
            "--port={}".format(runtime_cfg["port"]),
            "--trafficManagerPort={}".format(runtime_cfg["traffic_manager_port"]),
            "--trafficManagerSeed={}".format(runtime_cfg["traffic_seed"]),
            "--carlaProviderSeed={}".format(runtime_cfg["carla_seed"]),
            "--debug={}".format(runtime_cfg["debug"]),
            "--record=",
            "--timeout={}".format(runtime_cfg["timeout"]),
            "--routes={}".format(runtime_cfg["routes_file"]),
            "--route-id-start={}".format(runtime_cfg["route_id_start"]),
            "--num-routes={}".format(runtime_cfg["num_routes"]),
            "--route-passes=1",
            "--routes-dir={}".format(runtime_cfg["routes_dir"]),
            "--routes-pattern={}".format(runtime_cfg["routes_pattern"]),
            "--reuse-agent={}".format(runtime_cfg["reuse_agent"]),
            "--route-id-max={}".format(runtime_cfg["route_id_max"]),
            "--best-every={}".format(runtime_cfg["best_every"]),
            "--scenarios={}".format(runtime_cfg["scenarios_file"]),
            "--scenario_parameter={}".format(runtime_cfg["scenario_parameter_file"]),
            "--repetitions={}".format(runtime_cfg["repetitions"]),
            "--agent={}".format(runtime_cfg["team_agent"]),
            "--agent-config={}".format(agent_cfg_out),
            "--track={}".format(runtime_cfg["track"]),
            "--resume={}".format(resume_flag),
            "--checkpoint={}".format(checkpoint_endpoint),
            "--ego-num=1",
            "--skip_existed={}".format(runtime_cfg["skip_existed"]),
        ]

        env = os.environ.copy()
        env["CARLA_ROOT"] = carla_root
        env["LEADERBOARD_ROOT"] = leaderboard_root
        env["SCENARIO_RUNNER_ROOT"] = scenario_runner_root
        env["DATA_ROOT"] = data_root
        env["RESULT_ROOT"] = run_root
        env["CHECKPOINT_ENDPOINT"] = checkpoint_endpoint
        env["SAVE_PATH"] = run_images_root
        env["RL_LOG_DIR"] = run_rl_root

        py_parts = [
            env.get("PYTHONPATH", ""),
            os.path.join(carla_root, "PythonAPI"),
            os.path.join(carla_root, "PythonAPI", "carla"),
            os.path.join(carla_root, "PythonAPI", "carla", "dist", "carla-0.9.10-py3.7-linux-x86_64.egg"),
            leaderboard_root,
            os.path.join(leaderboard_root, "team_code"),
            scenario_runner_root,
            os.path.join(scenario_runner_root, "srunner"),
        ]
        env["PYTHONPATH"] = ":".join([p for p in py_parts if p])

        command_txt = " ".join(cmd)
        with open(os.path.join(run_manifest_root, "command.sh"), "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\nset -euo pipefail\n")
            f.write(command_txt + "\n")

        pass_entry = {
            "pass_index": pass_idx,
            "run_root": run_root,
            "checkpoint_endpoint": checkpoint_endpoint,
            "results_json": method_results,
            "resume": resume_flag,
            "existing_entry_status": existing_entry_status,
            "command": cmd,
        }
        method_entry["pass_runs"].append(pass_entry)

        print("[paper-eval] {} {}".format(method_name, pass_tag))
        if dry_run:
            print("[dry-run] {}".format(command_txt))
        else:
            log_path = os.path.join(run_manifest_root, "eval.log")
            with open(log_path, "w", encoding="utf-8") as log_file:
                proc = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Evaluation failed for {} {} (exit={}). See {}".format(
                        method_name, pass_tag, proc.returncode, log_path
                    )
                )

    with open(os.path.join(method_manifest_dir, "method_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(method_entry, f, indent=2)
    manifest["methods"].append(method_entry)

if not manifest["methods"]:
    raise ValueError("No methods selected. Check --method filter.")

manifest_path = os.path.join(exp_root, "run_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("[paper-eval] Manifest: {}".format(manifest_path))
PY

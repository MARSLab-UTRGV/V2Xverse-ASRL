#!/bin/bash

# $1, route id start
# $2, Carla port
# $3, exp_name
# $4, run id (e.g., 1 or run001)
# $5, agent config
# $6, scenario config
# $7, num routes (optional)
# $8, repeat passes (optional; full-pass repeats)
# $9, max route id (optional; stop a pass after this id)
# $10, reuse agent across routes (optional; 1 = reuse)

export CARLA_ROOT=external_paths/carla_root
export LEADERBOARD_ROOT=simulation/leaderboard
export SCENARIO_RUNNER_ROOT=simulation/scenario_runner
export DATA_ROOT=external_paths/data_root
export SAVE_DIR=results

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}/team_code
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=${2:-40000} # IMPORTANT: same as the carla server port
export TM_PORT=`expr $PORT + 5` # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export TRAFFIC_SEED=2000
export CARLA_SEED=2000
export REPETITIONS=1 # multiple evaluation runs
export ROUTE_ID_START=${1:-0}
export NUM_ROUTES=${7:-0}
export ROUTE_PASSES=${8:-0}
export ROUTE_ID_MAX=${9:-331}
export REUSE_AGENT=${10:-1}
export ROUTES=${LEADERBOARD_ROOT}/data/evaluation_routes/town05_short_r${ROUTE_ID_START}.xml
# verify the evaluation route, including start point and end point.
export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/town05_all_scenarios_2.json
export SCENARIOS_PARAMETER=${LEADERBOARD_ROOT}/leaderboard/scenarios/scenario_parameter$6.yaml

RUN_ID_RAW=${4:-1}
if [[ "${RUN_ID_RAW}" == run* ]]; then
  RUN_TAG="${RUN_ID_RAW}"
elif [[ "${RUN_ID_RAW}" =~ ^[0-9]+$ ]]; then
  RUN_TAG=$(printf "run%03d" "${RUN_ID_RAW}")
else
  RUN_TAG="run${RUN_ID_RAW}"
fi

export RESULT_ROOT=${SAVE_DIR}/rl_runs/${RUN_TAG}
export CHECKPOINT_ENDPOINT=${RESULT_ROOT}/eval/results.json
export SAVE_PATH=${RESULT_ROOT}/images
export RL_LOG_DIR=${RESULT_ROOT}/rl

export TEAM_AGENT=simulation/leaderboard/team_code/pnp_agent_e2e.py
# V2X agent with BEV input to indicate the drivable area.
export TEAM_CONFIG=simulation/leaderboard/team_code/agent_config/pnp_config_$5.yaml
# model config file!

export RESUME=0
export EGO_NUM=1
export SKIP_EXISTED=0

mkdir -p "${SAVE_PATH}"
mkdir -p "${RESULT_ROOT}/eval"
mkdir -p "${RL_LOG_DIR}"


python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_rl.py \
--scenarios=${SCENARIOS}  \
--scenario_parameter=${SCENARIOS_PARAMETER}  \
--routes=${ROUTES} \
--route-id-start=${ROUTE_ID_START} \
--num-routes=${NUM_ROUTES} \
--routes-dir=${LEADERBOARD_ROOT}/data/evaluation_routes \
--routes-pattern=town05_short_r{}.xml \
--route-id-max=${ROUTE_ID_MAX} \
--route-passes=${ROUTE_PASSES} \
--reuse-agent=${REUSE_AGENT} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--carlaProviderSeed=${CARLA_SEED} \
--trafficManagerSeed=${TRAFFIC_SEED} \
--ego-num=${EGO_NUM} \
--timeout 600 \
--skip_existed=${SKIP_EXISTED}

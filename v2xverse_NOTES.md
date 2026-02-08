# V2Xverse Evaluation and Control Map


## 1) Entry points and top-level execution

- Evaluation script:
  - `V2Xverse-ASRL/scripts/eval_driving_e2e.sh`
  - Sets env vars for CARLA, Leaderboard, ScenarioRunner, data paths, and selects route/scenario files.
  - Calls: `V2Xverse-ASRL/simulation/leaderboard/leaderboard/leaderboard_evaluator_parameter.py`


## 2) World sync and tick loop

- Evaluator sets synchronous CARLA mode:
  - `V2Xverse-ASRL/simulation/leaderboard/leaderboard/leaderboard_evaluator_parameter.py`
  - Sets `fixed_delta_seconds = 1/20` and seeds for determinism.

- Main tick loop:
  - `V2Xverse-ASRL/simulation/leaderboard/leaderboard/scenarios/scenario_manager.py`
  - Sequence per tick:
    1) `GameTime.on_carla_tick(...)`
    2) `CarlaDataProvider.on_carla_tick()`
    3) Agent called via `AgentWrapper` (blocks on sensors)
    4) Apply controls to ego vehicles
    5) Tick scenario behavior trees
    6) `world.tick()` to advance simulation

- Decision cadence detail:
  - CARLA still advances every sync tick (20 Hz), but the PnP agent computes new control every `skip_frames`.
  - On skipped frames it returns previous control (`pnp_agent_e2e.py`), so "tick rate" and "new decision rate" are different.

- Cached world state:
  - `V2Xverse-ASRL/simulation/scenario_runner/srunner/scenariomanager/carla_data_provider.py`
  - Provides actor velocities, transforms, traffic lights, and hero lookup `get_hero_actor(hero_id)`.

- Sensor buffering:
  - `V2Xverse-ASRL/simulation/leaderboard/leaderboard/envs/sensor_interface.py`
  - Asynchronous callbacks feed a queue; `get_data()` blocks until all sensors report.

## 3) Routes, scenarios, and triggers

- Route list / execution order:
  - `V2Xverse-ASRL/simulation/leaderboard/leaderboard/utils/route_indexer.py`

- Parsing route XML and scenario JSON:
  - `V2Xverse-ASRL/simulation/leaderboard/leaderboard/utils/route_parser.py`
  - `parse_routes_file(...)` reads `town05_short_r#.xml` (coarse anchors).
  - `parse_annotations_file(...)` reads `town05_all_scenarios_2.json` (scenario trigger positions).
  - `scan_route_for_scenarios(...)` matches triggers to dense route waypoints.

- Route + scenarios construction:
  - `V2Xverse-ASRL/simulation/leaderboard/leaderboard/scenarios/route_scenario.py`
  - `_cal_multi_routes(...)` creates per-ego coarse routes (different start positions).
  - `interpolate_trajectory(...)` densifies each route using CARLA GlobalRoutePlanner.
  - `self.route` becomes `List[List[(carla.Transform, RoadOption)]]`.
  - `ScenarioTriggerer` activates scenarios when ego reaches trigger locations.

- Scenario class mapping:
  - `RouteScenario.NUMBER_CLASS_TRANSLATION` in `route_scenario.py`:
    - Scenario1 -> ControlLoss
    - Scenario2 -> FollowLeadingVehicle
    - Scenario3 -> DynamicObjectCrossing
    - Scenario4 -> VehicleTurningRoute
    - Scenario5 -> OtherLeadingVehicle
    - Scenario6 -> ManeuverOppositeDirection
    - Scenario7/8/9 -> SignalJunctionCrossingRoute
    - Scenario10 -> NoSignalJunctionCrossingRoute

- Scenario parameter YAML:
  - `V2Xverse-ASRL/simulation/leaderboard/leaderboard/scenarios/scenario_parameter_*.yaml`
  - Used by `RouteScenario._build_scenario_parameter_instances(...)` to select scenario variants.

## 4) Agent integration and global plan

- Global plan injection:
  - `RouteScenario._update_route(...)` calls `config.agent.set_global_plan(gps_route, self.route)`.
  - `V2Xverse-ASRL/simulation/leaderboard/leaderboard/autoagents/autonomous_agent.py`
    - `set_global_plan(...)` downsamples the dense route using `target_point_distance`.
    - Default `target_point_distance = 50` meters unless overridden by config.

- PnP agent entry:
  - `V2Xverse-ASRL/simulation/leaderboard/team_code/pnp_agent_e2e.py`
  - In `run_step(...)`:
    - Builds local target point from `RoutePlanner`.
    - Feeds target + perception into `PnP_infer`.

- Route planner for target point:
  - `V2Xverse-ASRL/simulation/leaderboard/team_code/planner_pnp.py`
  - `RoutePlanner(min_distance=2.0, max_distance=10.0)` meters.
  - Scans route up to 10 m ahead and selects the next target point.

## 5) Perception and CoDriving planning

- Agent models:
  - `pnp_agent_e2e.py` loads perception (OpenCOOD) and planning (CoDriving) models.

- Perception + planning pipeline:
  - `V2Xverse-ASRL/simulation/leaderboard/team_code/pnp_infer_action_e2e.py`
  - Perception: `inference_utils.inference_intermediate_fusion_multiclass(...)`.
  - Planning input: occupancy stack + target point + fused features.
  - Planning output: `predicted_waypoints = planning_model(...)["future_waypoints"]`.
  - Output is typically 10 future waypoints (local BEV frame).

- Target point usage:
  - Target point is from global route planner, not predicted by the model.
  - CoDriving uses it as a goal cue to generate local future waypoints.

## 6) Low-level control

- Controller used by PnP:
  - `V2Xverse-ASRL/simulation/leaderboard/team_code/v2x_controller.py`
  - Called from `PnP_infer.generate_action_from_model_output(...)`.
  - Uses predicted waypoints + target to compute steering/throttle/brake.

## 7) Evaluation and criteria

- RouteScenario criteria:
  - `RouteScenario._create_test_criteria()` in `route_scenario.py`.
  - Builds `CollisionTest`, `InRouteTest`, `RouteCompletionTest`, `OutsideRouteLanesTest`, `RunningRedLightTest`, `RunningStopTest`, `ActorSpeedAboveThresholdTest`.

- Base scenario integration:
  - `V2Xverse-ASRL/simulation/scenario_runner/srunner/scenarios/basic_scenario.py`
  - Base class used by RouteScenario for behavior tree and criteria handling.

- Scoring:
  - `V2Xverse-ASRL/simulation/leaderboard/leaderboard/utils/statistics_manager.py`
  - Consumes traffic events and writes `results.json`.

## 8) Scenario trigger files and meaning

- Route anchors (coarse):
  - `V2Xverse-ASRL/simulation/leaderboard/data/evaluation_routes/town05_short_r#.xml`
  - Usually start/end anchors; densified into full route.

- Scenario triggers (events along the route):
  - `V2Xverse-ASRL/simulation/leaderboard/data/scenarios/town05_all_scenarios_2.json`
  - Locations and scenario types that are matched to the route.

## 9) Units and distance correlations (meters)

- `target_point_distance` in config (e.g., `pnp_config_codriving_5_10.yaml`):
  - Used by `AutonomousAgent.set_global_plan` to downsample global route.

- `RoutePlanner(min_distance=2.0, max_distance=10.0)`:
  - Selects target point within 10 m look-ahead; pops points within 2 m.

- Perception ranges:
  - Detection ranges and BEV extents are in meters; target points are clipped to those ranges in `pnp_agent_e2e.py`.

- Example hazard checks (meters):
  - Traffic light search `min_dis=8` in `pnp_infer_action_e2e.py`.
  - Bike hazard ignores if `distance > 20`.
  - Vehicle hazard thresholds are computed from speed in meters.

## 10) Key relationships (compact view)

- `eval_driving_e2e.sh` -> `leaderboard_evaluator_parameter.py` -> `ScenarioManager` -> `AgentWrapper` -> `PnP_Agent` -> `PnP_infer` -> `V2X_Controller` -> (optional RL gamma) -> `CBFQPFilter` -> throttle/brake.
- `town05_short_r#.xml` -> `RouteParser` -> `RouteScenario` -> `interpolate_trajectory` -> `set_global_plan` -> `RoutePlanner` -> target point.
- `town05_all_scenarios_2.json` -> `RouteParser.scan_route_for_scenarios` -> `ScenarioTriggerer` -> scenario class instance.
- Criteria and scoring flow: `RouteScenario` -> `BasicScenario` -> `TrafficEventType` -> `StatisticsManager`.

## 11) How to run

- Running carla simulation:
  - CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=40000
  
- Running eval_driving_e2e.sh
  - CUDA_VISIBLE_DEVICES=0 bash scripts/eval_driving_e2e.sh 24 40002 codriving 0 codriving_5_10 _2 with arguments:

- `Route_id` (`$1`): route file id. Route path is `simulation/leaderboard/data/evaluation_routes/town05_short_r${route_id}.xml` (currently 105 routes in this folder).
- `Carla_port` (`$2`): port used by python client to connect to CARLA. Keep it consistent with the CARLA server world port.
- `exp_name` (`$3`): experiment/output name segment used in `results/results_driving_${exp_name}/...`.
- `repeat_tag` (`$4`): output naming tag used in `.../r${route_id}_repeat${repeat_tag}`. It does **not** control evaluator repetitions; `scripts/eval_driving_e2e.sh` currently sets `REPETITIONS=1`.
- `Agent_config` (`$5`): agent config file `simulation/leaderboard/team_code/agent_config/pnp_config_${Agent_config}.yaml`. This file controls model and controller settings.
- `Scenario_config` (`$6`): scenario parameter file suffix, loaded as `simulation/leaderboard/leaderboard/scenarios/scenario_parameter${Scenario_config}.yaml` (for example `_2` -> `scenario_parameter_2.yaml`). Available variants should be checked in `simulation/leaderboard/leaderboard/scenarios/scenario_parameter*.yaml`.

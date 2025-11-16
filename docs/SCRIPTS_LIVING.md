# Scripts Living Document

_Latest snapshot for each script — auto-generated._


## Registry (latest per script)

| Script | Session | Last Run (IST) | Purpose | Outputs | Artifacts | git |

|---|---|---|---|---|---|---|

| `scripts\w11_build_targets.py` | S-W11 | 2025-11-09T15:59:03+05:30 | Build target weights from capped base × blended schedule | reports\wk11_blend_targets.csv | reports\wk11_blend_targets.csv | c3fb3184 |

| `scripts\w12_size_and_orders.py` | S-W12 | 2025-11-09T15:59:06+05:30 | Create order intents; basic capacity checks | reports\wk12_orders_lastday.csv; reports\wk12_orders_schedule.csv; reports\wk12_orders_validation.csv | reports\wk12_orders_lastday.csv; reports\wk12_orders_schedule.csv; reports\wk12_orders_validation.csv | c3fb3184 |

| `scripts\w13_dry_run_fills.py` | S-W13 | 2025-11-02T21:35:22+05:30 |  | reports\wk13_dryrun_fills.csv; reports\wk13_tca_summary.csv | reports\wk13_dryrun_fills.csv; reports\wk13_tca_summary.csv | 97ed5f36 |

| `scripts\w21_purgedcv_pbo.py` | S-W21 | 2025-11-06T22:56:51+05:30 |  | reports\w21_summary.json; reports\wk21_purgedcv_pbo.csv | reports\w21_summary.json; reports\wk21_purgedcv_pbo.csv | 97ed5f36 |

| `scripts\w25_exec_engineering.py` | S-W25 | 2025-11-08T15:07:23+05:30 |  |  |  | d9d5fad7 |

| `scripts\w4_voltarget_stops.py` | S-W4 | 2025-11-02T22:14:27+05:30 |  | reports\dd_throttle_map.csv; reports\kill_switch.yaml; reports\w4_vol_diag.json; reports\wk4_voltarget_stops.csv | reports\dd_throttle_map.csv; reports\kill_switch.yaml; reports\w4_vol_diag.json; reports\wk4_voltarget_stops.csv | 97ed5f36 |

| `scripts\w7_apply_gates.py` | S-W7 | 2025-11-09T15:58:54+05:30 | Apply macro/vol/DD gates to build daily risk schedule |  |  | c3fb3184 |

| `scripts\w7_compute_regimes.py` | S-W7 | 2025-11-09T15:58:51+05:30 | Compute regimes & macro gate timeline | reports\macro_gates_eval.csv; reports\regime_timeline.csv | reports\macro_gates_eval.csv; reports\regime_timeline.csv | c3fb3184 |

| `scripts\w8_apply_event_rules.py` | S-W8 | 2025-11-09T15:58:57+05:30 | Expand event rules to per-day/ticker flags | reports\events_position_flags.csv | reports\events_position_flags.csv | c3fb3184 |

| `scripts\w8_combine_schedules.py` | S-W8 | 2025-11-09T15:59:00+05:30 | Blend macro/DD and events multipliers | reports\risk_schedule_blended.csv | reports\risk_schedule_blended.csv | c3fb3184 |


---

## Details by Script


### `scripts\w11_build_targets.py`

- **Session**: S-W11

- **Last run**: 2025-11-09T15:59:03+05:30

- **Purpose**: Build target weights from capped base × blended schedule

- **Inputs**: config\.env.sample; config\aggressive.json; config\baseline.json; config\capacity_policy.backup.yaml; config\capacity_policy.yaml; config\change_control.yaml; config\config.yaml; config\config_commodities.yaml; config\config_fx.yaml; config\event_rules.yaml; config\kill_switch.yaml; config\macro_gates.yaml; config\my_universe.csv; config\policy_w1.yaml; config\policy_w2.yaml; config\reporting.toml; config\rolling.json; config\run.yaml; config\script_info_map.yaml; config\sector_mapping.csv; config\strategy.yaml; config\universe.csv; config\w4_atr.yaml

- **Outputs**: reports\wk11_blend_targets.csv

- **Artifacts**: reports\wk11_blend_targets.csv

- **Params**: `{"return_code": 0, "log_file": "reports\\logs\\w11_build_targets_2025-11-09_15-59-02.log"}`

- **git**: c3fb3184


### `scripts\w12_size_and_orders.py`

- **Session**: S-W12

- **Last run**: 2025-11-09T15:59:06+05:30

- **Purpose**: Create order intents; basic capacity checks

- **Inputs**: config\.env.sample; config\aggressive.json; config\baseline.json; config\capacity_policy.backup.yaml; config\capacity_policy.yaml; config\change_control.yaml; config\config.yaml; config\config_commodities.yaml; config\config_fx.yaml; config\event_rules.yaml; config\kill_switch.yaml; config\macro_gates.yaml; config\my_universe.csv; config\policy_w1.yaml; config\policy_w2.yaml; config\reporting.toml; config\rolling.json; config\run.yaml; config\script_info_map.yaml; config\sector_mapping.csv; config\strategy.yaml; config\universe.csv; config\w4_atr.yaml

- **Outputs**: reports\wk12_orders_lastday.csv; reports\wk12_orders_schedule.csv; reports\wk12_orders_validation.csv

- **Artifacts**: reports\wk12_orders_lastday.csv; reports\wk12_orders_schedule.csv; reports\wk12_orders_validation.csv

- **Params**: `{"return_code": 0, "log_file": "reports\\logs\\w12_size_and_orders_2025-11-09_15-59-04.log"}`

- **git**: c3fb3184


### `scripts\w13_dry_run_fills.py`

- **Session**: S-W13

- **Last run**: 2025-11-02T21:35:22+05:30

- **Purpose**: 

- **Inputs**: config\.env.sample; config\aggressive.json; config\baseline.json; config\capacity_policy.backup.yaml; config\capacity_policy.yaml; config\config.yaml; config\config_commodities.yaml; config\config_fx.yaml; config\event_rules.yaml; config\kill_switch.yaml; config\macro_gates.yaml; config\my_universe.csv; config\policy_w1.yaml; config\policy_w2.yaml; config\reporting.toml; config\rolling.json; config\run.yaml; config\script_info_map.yaml; config\sector_mapping.csv; config\strategy.yaml; config\universe.csv; config\w4_atr.yaml

- **Outputs**: reports\wk13_dryrun_fills.csv; reports\wk13_tca_summary.csv

- **Artifacts**: reports\wk13_dryrun_fills.csv; reports\wk13_tca_summary.csv

- **Params**: `{"return_code": 0, "log_file": "reports\\logs\\w13_dry_run_fills_2025-11-02_21-35-21.log"}`

- **git**: 97ed5f36


### `scripts\w21_purgedcv_pbo.py`

- **Session**: S-W21

- **Last run**: 2025-11-06T22:56:51+05:30

- **Purpose**: 

- **Inputs**: config\.env.sample; config\aggressive.json; config\baseline.json; config\capacity_policy.backup.yaml; config\capacity_policy.yaml; config\change_control.yaml; config\config.yaml; config\config_commodities.yaml; config\config_fx.yaml; config\event_rules.yaml; config\kill_switch.yaml; config\macro_gates.yaml; config\my_universe.csv; config\policy_w1.yaml; config\policy_w2.yaml; config\reporting.toml; config\rolling.json; config\run.yaml; config\script_info_map.yaml; config\sector_mapping.csv; config\strategy.yaml; config\universe.csv; config\w4_atr.yaml

- **Outputs**: reports\w21_summary.json; reports\wk21_purgedcv_pbo.csv

- **Artifacts**: reports\w21_summary.json; reports\wk21_purgedcv_pbo.csv

- **Params**: `{"return_code": 0, "log_file": "reports\\logs\\w21_purgedcv_pbo_2025-11-06_22-56-50.log"}`

- **git**: 97ed5f36


### `scripts\w25_exec_engineering.py`

- **Session**: S-W25

- **Last run**: 2025-11-08T15:07:23+05:30

- **Purpose**: 

- **Inputs**: config\.env.sample; config\aggressive.json; config\baseline.json; config\capacity_policy.backup.yaml; config\capacity_policy.yaml; config\change_control.yaml; config\config.yaml; config\config_commodities.yaml; config\config_fx.yaml; config\event_rules.yaml; config\kill_switch.yaml; config\macro_gates.yaml; config\my_universe.csv; config\policy_w1.yaml; config\policy_w2.yaml; config\reporting.toml; config\rolling.json; config\run.yaml; config\script_info_map.yaml; config\sector_mapping.csv; config\strategy.yaml; config\universe.csv; config\w4_atr.yaml

- **Outputs**: 

- **Artifacts**: 

- **Params**: `{"return_code": 0, "log_file": "reports\\logs\\w25_exec_engineering_2025-11-08_15-07-23.log"}`

- **git**: d9d5fad7


### `scripts\w4_voltarget_stops.py`

- **Session**: S-W4

- **Last run**: 2025-11-02T22:14:27+05:30

- **Purpose**: 

- **Inputs**: config\.env.sample; config\aggressive.json; config\baseline.json; config\capacity_policy.backup.yaml; config\capacity_policy.yaml; config\config.yaml; config\config_commodities.yaml; config\config_fx.yaml; config\event_rules.yaml; config\kill_switch.yaml; config\macro_gates.yaml; config\my_universe.csv; config\policy_w1.yaml; config\policy_w2.yaml; config\reporting.toml; config\rolling.json; config\run.yaml; config\script_info_map.yaml; config\sector_mapping.csv; config\strategy.yaml; config\universe.csv; config\w4_atr.yaml

- **Outputs**: reports\dd_throttle_map.csv; reports\kill_switch.yaml; reports\w4_vol_diag.json; reports\wk4_voltarget_stops.csv

- **Artifacts**: reports\dd_throttle_map.csv; reports\kill_switch.yaml; reports\w4_vol_diag.json; reports\wk4_voltarget_stops.csv

- **Params**: `{"return_code": 0, "log_file": "reports\\logs\\w4_voltarget_stops_2025-11-02_22-14-26.log"}`

- **git**: 97ed5f36


### `scripts\w7_apply_gates.py`

- **Session**: S-W7

- **Last run**: 2025-11-09T15:58:54+05:30

- **Purpose**: Apply macro/vol/DD gates to build daily risk schedule

- **Inputs**: config\.env.sample; config\aggressive.json; config\baseline.json; config\capacity_policy.backup.yaml; config\capacity_policy.yaml; config\change_control.yaml; config\config.yaml; config\config_commodities.yaml; config\config_fx.yaml; config\event_rules.yaml; config\kill_switch.yaml; config\macro_gates.yaml; config\my_universe.csv; config\policy_w1.yaml; config\policy_w2.yaml; config\reporting.toml; config\rolling.json; config\run.yaml; config\script_info_map.yaml; config\sector_mapping.csv; config\strategy.yaml; config\universe.csv; config\w4_atr.yaml

- **Outputs**: 

- **Artifacts**: 

- **Params**: `{"return_code": 1, "log_file": "reports\\logs\\w7_apply_gates_2025-11-09_15-58-53.log"}`

- **git**: c3fb3184


### `scripts\w7_compute_regimes.py`

- **Session**: S-W7

- **Last run**: 2025-11-09T15:58:51+05:30

- **Purpose**: Compute regimes & macro gate timeline

- **Inputs**: config\.env.sample; config\aggressive.json; config\baseline.json; config\capacity_policy.backup.yaml; config\capacity_policy.yaml; config\change_control.yaml; config\config.yaml; config\config_commodities.yaml; config\config_fx.yaml; config\event_rules.yaml; config\kill_switch.yaml; config\macro_gates.yaml; config\my_universe.csv; config\policy_w1.yaml; config\policy_w2.yaml; config\reporting.toml; config\rolling.json; config\run.yaml; config\script_info_map.yaml; config\sector_mapping.csv; config\strategy.yaml; config\universe.csv; config\w4_atr.yaml

- **Outputs**: reports\macro_gates_eval.csv; reports\regime_timeline.csv

- **Artifacts**: reports\macro_gates_eval.csv; reports\regime_timeline.csv

- **Params**: `{"return_code": 0, "log_file": "reports\\logs\\w7_compute_regimes_2025-11-09_15-58-49.log"}`

- **git**: c3fb3184


### `scripts\w8_apply_event_rules.py`

- **Session**: S-W8

- **Last run**: 2025-11-09T15:58:57+05:30

- **Purpose**: Expand event rules to per-day/ticker flags

- **Inputs**: config\.env.sample; config\aggressive.json; config\baseline.json; config\capacity_policy.backup.yaml; config\capacity_policy.yaml; config\change_control.yaml; config\config.yaml; config\config_commodities.yaml; config\config_fx.yaml; config\event_rules.yaml; config\kill_switch.yaml; config\macro_gates.yaml; config\my_universe.csv; config\policy_w1.yaml; config\policy_w2.yaml; config\reporting.toml; config\rolling.json; config\run.yaml; config\script_info_map.yaml; config\sector_mapping.csv; config\strategy.yaml; config\universe.csv; config\w4_atr.yaml

- **Outputs**: reports\events_position_flags.csv

- **Artifacts**: reports\events_position_flags.csv

- **Params**: `{"return_code": 0, "log_file": "reports\\logs\\w8_apply_event_rules_2025-11-09_15-58-56.log"}`

- **git**: c3fb3184


### `scripts\w8_combine_schedules.py`

- **Session**: S-W8

- **Last run**: 2025-11-09T15:59:00+05:30

- **Purpose**: Blend macro/DD and events multipliers

- **Inputs**: config\.env.sample; config\aggressive.json; config\baseline.json; config\capacity_policy.backup.yaml; config\capacity_policy.yaml; config\change_control.yaml; config\config.yaml; config\config_commodities.yaml; config\config_fx.yaml; config\event_rules.yaml; config\kill_switch.yaml; config\macro_gates.yaml; config\my_universe.csv; config\policy_w1.yaml; config\policy_w2.yaml; config\reporting.toml; config\rolling.json; config\run.yaml; config\script_info_map.yaml; config\sector_mapping.csv; config\strategy.yaml; config\universe.csv; config\w4_atr.yaml

- **Outputs**: reports\risk_schedule_blended.csv

- **Artifacts**: reports\risk_schedule_blended.csv

- **Params**: `{"return_code": 0, "log_file": "reports\\logs\\w8_combine_schedules_2025-11-09_15-58-59.log"}`

- **git**: c3fb3184

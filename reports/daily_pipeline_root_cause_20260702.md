# Daily Pipeline Root Cause Audit

## Proven failure

Production run `2026-07-02T11:13:28Z` executed the first ten stages with
exit code `0`. `shadow_monitor.py` returned exit code `1` because
`snapshot_coverage_pass=False` with 146 allegedly missed races.

The 146 races were:

- 125 unsupported foreign races
- 15 domestic races whose final prediction window had not closed
- 6 historical domestic misses from the earlier broken freeze period

The validator mixed unsupported, not-yet-due, and historical races into the
current operational coverage gate. The first two groups were not prediction
candidates. The last group was immutable historical evidence and caused every
future run to fail permanently even after the producer was fixed.

## Execution graph

```text
run_daily_pipeline.py
  -> pipeline_runner.runner_lock("daily_pipeline")
  -> update_race_programs.py
     -> pedigreeall_core.APIClient
     -> snapshot_store.insert_program_capture
  -> snapshot_store.py
     -> migrate_provenance_schema.apply_migrations
  -> download_agfv2.py --today --tables 1 2 --force-refresh
  -> komiser.py --today
  -> process_komiser.py --today
  -> update_track_conditions.py
  -> update_workouts.py
  -> update_results.py
     -> normalize_data.normalize_entity
     -> snapshot_store.append_normalized_result
     -> results_coverage.write_results_coverage
  -> build_asof_features.py
     -> feature_contract.validate_model_feature_contract
     -> run_leakage_ci.py                         [subprocess, check=True]
     -> validate_feature_provenance.py            [subprocess, check=True]
  -> validate_feature_provenance.py
     -> all provenance and leakage checks
  -> shadow_monitor.py                            [FAILED HERE]
     -> match_prediction_results
     -> export_prediction_history
     -> prediction/feature drift
     -> calibration and ROI
     -> validate_feature_provenance.validate
     -> feature_contract validation
     -> snapshot_coverage_pass                    [false]
     -> exit 1 when any critical gate is false
```

## Exit and exception behavior

- `run_daily_pipeline.py`: exits `0` only when every step exits `0`; otherwise
  exits `1`.
- `build_asof_features.py`: nested validation uses `check=True`; a nested
  non-zero raises `CalledProcessError` and terminates the step.
- `validate_feature_provenance.py`: returns `1` when any provenance check fails.
- `shadow_monitor.py`: returns `1` when provenance, feature contract, snapshot
  coverage, prediction drift, calibration, or feature drift is critical.
- `update_race_programs.py`: explicitly exits `1` from its top-level exception
  handler.
- Other daily scripts terminate non-zero on uncaught exceptions.

## Swallowed exceptions on the daily path

These handlers intentionally continue, return a sentinel, or log without
re-raising. They do not explain the observed July 2 failure, but they are part
of the execution audit:

- `update_race_programs.py`: lines 46, 67; request failures become `None`, and
  state-read errors become warnings. Its top-level handler exits `1`.
- `snapshot_store.py`: numeric parse failures return `None`; date format probes
  continue before raising on total failure.
- `download_agfv2.py`: lines 62, 197, 286, 350, 388, 424, 442, 472, 559, 618,
  653; request/page/preview failures are recorded or skipped.
- `komiser.py`: lines 43, 63, 124, 159; request/date/page failures return a
  sentinel or stop pagination without non-zero exit.
- `process_komiser.py`: lines 86, 118, 168, 536, 550, 565, 578; malformed PDF,
  extraction, date, and optional enrichment failures are skipped or recorded.
- `update_track_conditions.py`: lines 129, 144, 166; per-track HTTP and parse
  failures are converted into unavailable records.
- `update_workouts.py`: line 57; a malformed stored JSON row is skipped.
- `update_results.py`: lines 65 and 221; resolver failures return a structured
  failure, while technical normalization failures are accumulated and raised
  after processing.
- `shadow_monitor.py`: metric `ValueError` handlers convert undefined metrics
  to `NaN`; safety-gate exceptions are not swallowed.

## Subprocess and hard-exit inventory

- `pipeline_runner.py`: `subprocess.run(..., capture_output=True)`.
- `build_asof_features.py`: two `subprocess.run(..., check=True)` validations.
- `healthcheck.py`: `systemctl is-active` subprocess.
- `web_app.py`: git `check_output` and service-log subprocess.
- `run_smoke.py`: `Popen` with stdout/stderr sent to `DEVNULL`; explicit
  `sys.exit(1)` on smoke failure.
- `update_race_programs.py`: explicit `sys.exit(1)` in its exception handler.
- No repository use of `check_call`.
- No explicit `CalledProcessError` handler; `check=True` failures propagate.

## Fix

Migration `015_snapshot_coverage_epoch.sql` records the beginning of the new
fail-closed coverage contract. Coverage remains a hard failure, but candidates
must now satisfy all of the following:

1. Race starts at or after the coverage epoch.
2. Track is supported and domestic.
3. The `start - 5 minutes` final prediction window has closed.
4. An immutable pre-race prediction exists.

Global archive integrity, feature hashes, provenance, post-start guards, drift,
calibration, and feature-contract checks are unchanged.

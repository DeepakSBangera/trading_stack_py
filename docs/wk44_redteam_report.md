# W44 — Red-Team & Recovery Drill

**As-of (IST):** 2025-11-09T14:22:26.406325+05:30

## Checks Run

- **existence_checks**: ✅
- **manifest_replay**: ✅ — rows=25; last_has=ts_utc,git_sha,artifact,config_hash; last_missing= — rows=25
- **backup_zip**: ✅ — files=3 — zip=F:\Projects\trading_stack_py\reports\W44_redteam_snapshot.zip
- **canary_log_schema**: ✅ — rows=402
- **kill_switch_config**: ✅
- **missing_data_dir**: ✅ — Temporarily moved data/csv → data/_csv_tmp_redteam.; Restored data/csv.

## Result

- **All-green:** ✅
- **Critical files present:** ✅

## Next Steps

- Keep the snapshot zip safe (off-box copy).
- If any check failed, file a hotfix task with rollback steps attached.
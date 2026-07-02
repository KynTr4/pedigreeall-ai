CREATE TABLE IF NOT EXISTS monitoring_epochs (
    monitor_name TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    reason TEXT NOT NULL
);

INSERT OR IGNORE INTO monitoring_epochs(monitor_name, started_at, reason)
VALUES(
    'race_freeze_v2_fail_closed',
    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
    'Coverage scope aligned with fail-closed race freeze and supported tracks'
);

-- Migration 017: freeze state machine v2 — failure reasons, extended state, retry tracking
-- All ADD COLUMN statements are nullable or have defaults for backward compatibility.

ALTER TABLE race_prediction_lifecycle ADD COLUMN freeze_state TEXT;
ALTER TABLE race_prediction_lifecycle ADD COLUMN failure_reason TEXT;
ALTER TABLE race_prediction_lifecycle ADD COLUMN post_start_retries INTEGER NOT NULL DEFAULT 0;
ALTER TABLE race_prediction_lifecycle ADD COLUMN ntp_offset_ms REAL;
ALTER TABLE race_prediction_lifecycle ADD COLUMN first_seen_at TEXT;
ALTER TABLE race_prediction_lifecycle ADD COLUMN capture_attempted_at TEXT;
ALTER TABLE race_prediction_lifecycle ADD COLUMN windows_json TEXT;

-- Back-fill freeze_state from existing status for historical rows
UPDATE race_prediction_lifecycle
SET freeze_state = CASE status
    WHEN 'SOURCE_UNSUPPORTED'    THEN 'SOURCE_UNSUPPORTED'
    WHEN 'RESULT_CAPTURED'       THEN 'RESULT_CAPTURED'
    WHEN 'FINAL_PREDICTION_DONE' THEN 'FINAL_CAPTURED'
    WHEN 'RESULT_PENDING'        THEN 'RESULT_PENDING'
    WHEN 'MISSED_FINAL_WINDOW'   THEN 'FAILED'
    WHEN 'WAITING'               THEN 'WAITING'
    WHEN 'FINAL_REFRESH_DUE'     THEN 'CAPTURING'
    WHEN 'RACE_STARTED'          THEN 'RESULT_PENDING'
    ELSE status
END
WHERE freeze_state IS NULL;

-- Back-fill first_seen_at from created_at
UPDATE race_prediction_lifecycle
SET first_seen_at = created_at
WHERE first_seen_at IS NULL;

-- Back-fill failure_reason for FAILED rows
UPDATE race_prediction_lifecycle
SET failure_reason = 'UNKNOWN'
WHERE freeze_state = 'FAILED' AND failure_reason IS NULL;

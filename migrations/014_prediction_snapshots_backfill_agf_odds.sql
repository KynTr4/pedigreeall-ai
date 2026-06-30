-- Migration to backfill agf_percent, agf_rank, and odds columns in prediction_snapshots
-- using values from agf_snapshots and odds_snapshots at prediction time.

-- Drop the no_update trigger to allow backfill
DROP TRIGGER IF EXISTS prediction_snapshots_no_update;

-- Backfill AGF percent and AGF rank
UPDATE prediction_snapshots
SET
  agf_percent = (
      SELECT agf_percent FROM agf_snapshots
      WHERE agf_snapshots.race_id = prediction_snapshots.race_id
        AND agf_snapshots.horse_id = prediction_snapshots.horse_id
        AND julianday(agf_snapshots.captured_at) <= julianday(prediction_snapshots.prediction_time)
      ORDER BY captured_at DESC LIMIT 1
  ),
  agf_rank = (
      SELECT agf_rank FROM agf_snapshots
      WHERE agf_snapshots.race_id = prediction_snapshots.race_id
        AND agf_snapshots.horse_id = prediction_snapshots.horse_id
        AND julianday(agf_snapshots.captured_at) <= julianday(prediction_snapshots.prediction_time)
      ORDER BY captured_at DESC LIMIT 1
  )
WHERE agf_percent IS NULL OR agf_rank IS NULL;

-- Backfill Odds
UPDATE prediction_snapshots
SET
  odds = (
      SELECT odds FROM odds_snapshots
      WHERE odds_snapshots.race_id = prediction_snapshots.race_id
        AND odds_snapshots.horse_id = prediction_snapshots.horse_id
        AND julianday(odds_snapshots.captured_at) <= julianday(prediction_snapshots.prediction_time)
      ORDER BY captured_at DESC LIMIT 1
  )
WHERE odds IS NULL;

-- Recreate the no_update trigger to preserve append-only invariants
CREATE TRIGGER IF NOT EXISTS prediction_snapshots_no_update
BEFORE UPDATE ON prediction_snapshots BEGIN
    SELECT RAISE(ABORT, 'prediction_snapshots is append-only');
END;

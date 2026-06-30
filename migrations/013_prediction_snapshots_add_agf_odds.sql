-- Migration to add immutable AGF and odds snapshot values to prediction_snapshots
-- This allows post-race diagnostics, SHAP, and bet simulator to use the exact values present at prediction time.

ALTER TABLE prediction_snapshots ADD COLUMN agf_percent REAL;
ALTER TABLE prediction_snapshots ADD COLUMN agf_rank INTEGER;
ALTER TABLE prediction_snapshots ADD COLUMN odds REAL;

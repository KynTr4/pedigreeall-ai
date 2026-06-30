\---



name: horse-racing-ai

description: Use this skill when working on the at\_yaris\_tahmini horse racing prediction project, SQLite database, VPS deployment, snapshot pipeline, AGF updates, shadow mode, predictions, dashboard, leakage gates, feature engineering, model monitoring, or production debugging.

\--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



\# Horse Racing AI Project Skill



\## Purpose



This skill provides project-specific knowledge for the `at\_yaris\_tahmini` repository.



Always use this skill when working on:



\* Horse racing prediction

\* SQLite database

\* VPS deployment

\* Shadow mode

\* Dashboard

\* Snapshot pipeline

\* AGF collection

\* Feature engineering

\* Leakage validation

\* Model monitoring

\* Production debugging

\* Prediction analysis



\---



\# Project Root



```

/opt/at\_yaris\_tahmini

```



Main database:



```

/opt/at\_yaris\_tahmini/pedigreeall\_progress.db

```



Dashboard:



```

http://5.175.136.118:8000

```



Logs:



```

/var/log/at\_yaris\_tahmini

```



\---



\# Core Rules



Never guess.



Always verify with:



\* SQL queries

\* logs

\* reports

\* source code

\* filesystem

\* systemctl

\* journalctl



If evidence is missing, explicitly state what cannot be verified.



Never fabricate successful executions.



\---



\# Database Rules



Never overwrite the production SQLite database.



Always create a timestamped backup before destructive operations.



Prefer read-only access whenever possible.



Never delete tables without confirmation.



Use transactions for write operations.



\---



\# Important Tables



\## Snapshots



\* program\_snapshots

\* agf\_snapshots

\* odds\_snapshots



\## Results



\* race\_results



\## Predictions



\* prediction\_snapshots

\* prediction\_results



\## Monitoring



\* shadow\_monitoring\_runs



\---



\# Snapshot Principles



Snapshots are immutable.



Never modify historical snapshot rows.



All model features must originate from snapshots collected before race start.



\---



\# Prediction Rules



Predictions are valid only when:



```

prediction\_time < race\_start\_at

```



Displayed predictions must come from:



```

prediction\_snapshots

```



Horse names should always be retrieved by joining:



```

prediction\_snapshots

```



with



```

program\_snapshots

```



using:



```

race\_id

horse\_id

```



\---



\# Leakage Rules



Never use post-race information as model inputs.



Forbidden features include:



\* finish\_position

\* finish\_time

\* final\_odds

\* GNY

\* prize

\* margin

\* race\_result\_status



Only use information satisfying:



```

captured\_at < race\_start\_at

```



If coverage fails:



\* report race\_id

\* explain reason

\* never silently ignore



Never disable leakage checks merely to generate predictions.



\---



\# AGF Collection Rules



Collection frequency:



More than 60 minutes



\* every 15 minutes



60–30 minutes



\* every 5 minutes



30–10 minutes



\* every 2 minutes



Last 10 minutes



\* every 60 seconds

\* nearest race only



After race start



\* no capture



Late snapshots must be retained but excluded from feature generation.



\---



\# Dashboard Rules



Dashboard is read-only.



It must never modify the database.



SQLite must operate using read-only mode.



Enable:



```

PRAGMA query\_only=ON;

```



Basic Auth must remain enabled.



Never expose:



\* .env

\* secrets

\* credentials



\---



\# Shadow Mode Rules



Shadow mode is not production betting.



Shadow mode exists only for evaluation.



Production readiness requires:



\* 90 healthy shadow days

\* leakage PASS

\* feature contract PASS

\* snapshot coverage PASS

\* acceptable calibration

\* acceptable drift

\* prediction/result matching



ROI is not considered valid unless odds snapshots were collected before prediction.



\---



\# Feature Engineering Rules



Every feature must be reproducible.



Every feature must have provenance.



Feature generation must remain deterministic.



Never use unavailable future information.



\---



\# Monitoring Rules



Investigate:



\* failed predictions

\* missing races

\* unmatched results

\* missing snapshots

\* unhealthy monitoring runs



Always identify the root cause.



\---



\# Daily Validation Order



1\. Verify database counts.

2\. Verify snapshots.

3\. Verify AGF.

4\. Verify predictions.

5\. Verify matching.

6\. Verify monitoring.

7\. Verify dashboard.

8\. Verify logs.



\---



\# Common SQL



\## Today's snapshots



```sql

SELECT date(race\_start\_at), COUNT(\*)

FROM program\_snapshots

GROUP BY date(race\_start\_at)

ORDER BY date(race\_start\_at) DESC;

```



\---



\## Today's predictions



```sql

SELECT date(race\_start\_at), COUNT(\*)

FROM prediction\_snapshots

GROUP BY date(race\_start\_at)

ORDER BY date(race\_start\_at) DESC;

```



\---



\## Today's top predictions



```sql

SELECT

&#x20;   p.race\_start\_at,

&#x20;   s.track,

&#x20;   s.race\_no,

&#x20;   s.horse\_name,

&#x20;   ROUND(p.ensemble\_probability,4),

&#x20;   p.predicted\_rank,

&#x20;   p.prediction\_time

FROM prediction\_snapshots p

JOIN program\_snapshots s

ON p.race\_id=s.race\_id

AND p.horse\_id=s.horse\_id

WHERE date(p.race\_start\_at)=date('now','localtime')

AND p.predicted\_rank=1

ORDER BY p.race\_start\_at;

```



\---



\# Manual Prediction Pipeline



```bash

cd /opt/at\_yaris\_tahmini



source .venv/bin/activate



python build\_asof\_features.py



python validate\_feature\_provenance.py



python shadow\_mode.py



python shadow\_monitor.py

```



\---



\# Services



```bash

systemctl status at-yaris-web.service --no-pager

```



```bash

systemctl list-timers 'at-yaris-\*'

```



```bash

journalctl -u at-yaris-daily.service -n 200 --no-pager

```



\---



\# Important Scripts



\* build\_asof\_features.py

\* validate\_feature\_provenance.py

\* shadow\_mode.py

\* shadow\_monitor.py

\* update\_race\_programs.py

\* update\_results.py

\* run\_daily\_pipeline.py

\* run\_agf\_update.py

\* snapshot\_store.py

\* healthcheck.py



\---



\# VPS Rules



Use systemd timers.



Never replace manual executions without verification.



Always inspect logs after failures.



Important logs:



\* daily.log

\* daily.err.log

\* agf.log

\* results.log

\* run\_latest.json

\* vps\_healthcheck.md



\---



\# Debugging Workflow



Always follow this order.



1\. Identify the failure.

2\. Collect evidence.

3\. Check SQL.

4\. Check logs.

5\. Reproduce manually.

6\. Fix the smallest possible issue.

7\. Run tests.

8\. Verify the fix.

9\. Report remaining risks.



\---



\# Output Format



When answering project questions always include:



\## Problem



What failed.



\## Evidence



SQL queries, logs, reports, or code references.



\## Root Cause



Explain the actual cause.



\## Fix



Exact modification performed.



\## Verification



Commands executed.



Observed results.



\## Remaining Risks



Anything still requiring attention.



Never claim success without evidence.




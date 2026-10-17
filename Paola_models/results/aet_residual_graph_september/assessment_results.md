# Frozen AET error corrections: September 19–30 assessment

**AET + graph error correction: 5.06125 EUR/MWh.**
**AET + correction without graph messages: 5.05213 EUR/MWh.**

Both are slightly worse than AET's **5.04427**: graph correction increases MAE **0.336%**,
and no-message correction increases it **0.156%**. Neither reaches MAE below 5. The previous
fixed 25% graph / 75% AET blend remains best at **4.99794**. Both corrections improve the
ZONE_2 average but worsen the other three zone averages. The small June–August improvements
did not carry over to this September block.

| method | MAE_EUR_MWh | change_vs_AET_pct |
| --- | --- | --- |
| AET | 5.0443 | 0.0000 |
| Previous 25% graph / 75% AET blend | 4.9979 | -0.9186 |
| AET + graph error correction | 5.0612 | 0.3365 |
| AET + correction without graph messages | 5.0521 | 0.1558 |

Lower MAE is better. All methods use the same **12 dates, four equally weighted zones and
4,608 quarter-price predictions**. A negative change versus AET means lower error.
The previous blend combines two full reconstructions; the new models add an estimated AET error.

Both new formulas are **AET + 0.25 × learned correction**, with identical frozen August-cutoff
AET teachers. Both correction networks trained only on chronological out-of-fold AET errors
through August 31. Their architectures and correction strengths were chosen on January–May
and fixed before June–August confirmation. The no-message candidate was the development-selected
variant; this September replay does not change that selection. No fitting, recalibration,
strength search or candidate retuning occurred.

**By zone**

| market_id | AET | Previous 25% graph / 75% AET blend | AET + graph error correction | AET + correction without graph messages |
| --- | --- | --- | --- | --- |
| ZONE_0 | 3.8593 | 3.7776 | 3.9027 | 3.8755 |
| ZONE_1 | 6.2577 | 6.1435 | 6.2784 | 6.2632 |
| ZONE_2 | 5.5064 | 5.5153 | 5.4702 | 5.4839 |
| ZONE_3 | 4.5537 | 4.5554 | 4.5937 | 4.5859 |

**Dates with lower error than AET**

| method | dates_better_than_AET | dates | below_5 |
| --- | --- | --- | --- |
| AET + graph error correction | 5.0000 | 12.0000 | False |
| AET + correction without graph messages | 4.0000 | 12.0000 | False |

Date/zone/quarter summaries and every prediction-versus-actual signed and absolute error are saved
alongside this report. The components' weights and original assessment input hashes are unchanged.
All four point outputs passed complete-key, finite-value and hourly-mean checks after CSV reload;
maximum hourly-mean error was 2.51e-11 EUR/MWh,
below the required 0.0001. AET used no fallback hours. No uncertainty intervals are claimed for
these deterministic corrections.

For every date, only supplied target-day hourly curves and strictly earlier quarter-price history
enter prediction features. Earlier assessment actuals may enter later dates' history, matching the
existing daily replay. September was already explored and evaluated for earlier recipes, so this
explicitly requested historical assessment is not an independent final test. No October data was read.
The result must not be used to choose a September-optimal correction strength.

Saved locally under Paola_models on branch Paola. Existing artifacts and the AET dashboard were
not changed by this assessment. No commit or push was made.

The model test suite completed with **82 passing tests** after adding this assessment runner.

Reproduce from ApplePi with a fresh output directory:

```bash
.venv-mae/bin/python -m Paola_models.assess_residual_september --out Paola_models/results/NEW_frozen_residual_assessment
```

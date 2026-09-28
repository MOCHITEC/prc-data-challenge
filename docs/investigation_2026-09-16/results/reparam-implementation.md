# Result — implementation of item 9 `mms-mixture-reparam` variant A (2026-09-17)

Implements the variant `mms-mixture-reparam.md` §3.3 measured as **A** — model B's target is
`TAXITIME_SEC_mvt − mvt_minus_sched` on LIRF rows and the plain label elsewhere, with the offset added back to
the prediction — as a `TrainConfig` flag that is **off by default**, and measures it on the forward fold with
the repository's own decision rule (`docs/README.md` §6).

Inputs read, not re-measured: `mms-mixture-reparam.md` (the variant and its jan/jul/forward OOF numbers),
`sched-filled-block.md` (the identity `y − ms = SCHED − BLOCK` and the LIRF fill), `findings.md` §3.1/§3.2,
`docs/README.md` §6/§12, `docs/guard_audit.md`.

## 1. What was changed

| file | change |
|---|---|
| `src/prc/train.py` | new constant `SCHED_OFFSET = "mvt_minus_sched"`; two `TrainConfig` fields; two module-level functions; the model-B branch of `predict_fold` |
| `src/prc/pipeline.py` | one line: `unmatched_sched_reparam_airports` added to the keys whose YAML list is turned into a tuple |
| `configs/reparam_lirf.yaml` | **new**; `configs/main.yaml` and every other existing config untouched |
| `tests/test_train.py` | one fixture helper and four tests |

**Config surface** (`TrainConfig`, documented inline as the neighbouring fields are):

* `unmatched_sched_reparam: bool = False` — the flag. Off means today's model.
* `unmatched_sched_reparam_airports: tuple[str, ...] = ("LIRF",)` — the airport set, config rather than a
  constant inside a function. The default is the set the measurement covers; `mms-mixture-reparam.md` §3.3
  measured the same shift applied to *all* unmatched rows at 2,696.7 against a 2,326.5 baseline, so the
  restriction is the substance of the variant, not a detail.

**The shift itself** is one function, used in both directions:

```python
def sched_reparam_shift(frame, cfg) -> np.ndarray:   # mvt_minus_sched at the named airports, else 0.0
def unmatched_target(frame, cfg) -> np.ndarray:      # frame[LABEL] - sched_reparam_shift(frame, cfg)
```

In `predict_fold`'s model-B branch the fit is `_fit(u_tr, lambda f: unmatched_target(f, cfg_b), ...)` and the
prediction is `g + sched_reparam_shift(u_tg, cfg_b)`. `predict_fold` is the only place model B is fitted:
`run_cv` calls it per fold and `predict_ranking` calls it for the full-data refit, so **fit-side and
prediction-side use the same two functions on both paths by construction** — there is no second
implementation to keep in step. Nothing downstream (`_assemble`'s clip and id-join, the metrics, the
submission gate) sees model space; it is handed back seconds of taxi time.

Rows that are never shifted: any row at an airport the config does not name, and any row whose
`mvt_minus_sched` is null (the shift is 0 there, so the label passes through unchanged — shifting it would put
a NaN in the label vector, which LightGBM accepts silently). Null rate of `mvt_minus_sched` is 0 on the
2,085,047 training rows, on the 344,841 ranking rows and on the 5,290 unmatched ranking rows (383 of them
LIRF), so no row is actually affected by that guard in this data; it is there because the identity is only
available when the feature is.

With the flag off, `sched_reparam_shift` returns an array of zeros and the arithmetic is `y - 0.0` and
`pred + 0.0`, which is exactly the previous model. Two new fields do change `TrainConfig.hash()`, so the
`exp_id` of an unchanged config changes (`inv_baseline_forward-67e76a5dfc` → `-9271caca22`); the model does
not (§4.1).

`predict_fold`'s `info` dict gains `sched_reparam_train_rows` / `sched_reparam_target_rows` when the flag is
on (1,348 and 140 in the forward-fold run), so the experiment record says how many rows were shifted.

## 2. Tests added (`tests/test_train.py`)

| test | what it pins |
|---|---|
| `test_sched_reparam_shifts_exactly_the_configured_airport_rows` | the shift equals `mvt_minus_sched` on LIRF rows and is exactly `0.0` on every other row; `unmatched_target` equals `y − ms` there and `y` elsewhere; naming a second airport moves exactly that airport's rows too; naming none is a no-op |
| `test_sched_reparam_never_shifts_a_row_with_a_null_schedule_offset` | LIRF rows whose `mvt_minus_sched` is null get shift `0.0`, keep the plain label, and the target stays finite |
| `test_sched_reparam_is_off_by_default_and_leaves_the_model_unchanged` | `TrainConfig().unmatched_sched_reparam is False`; the target array is bit-equal to the label; predictions are bit-equal (`==`, not `allclose`) with the flag off whatever the airport set says, and bit-equal again with the flag on and an empty airport set; `info` carries no reparam keys when off |
| `test_sched_reparam_predictions_come_back_on_the_label_scale` | end-to-end `predict_fold` against an **unlabelled** target frame (the ranking shape, i.e. the refit path): every LIRF unmatched prediction exceeds `0.8 × mvt_minus_sched` and the median \|pred − ms\| is under 20 % of the median ms, on a fixture where every LIRF departure is sched-filled so the model-space target is ~0 and the prediction has to come from the add-back; the two `info` counts equal the intended row counts |

The fixture helper `_lirf_filled_month` turns every LIRF departure of a synthetic month into an unmatched
sched-filled row (`BLOCK = SCHED`, `MVT = SCHED + 3 h`), which is the regime `sched-filled-block.md` describes;
the generator already produces a few such rows, this makes a small fixture carry enough of them.

Mutation check in the spirit of `docs/guard_audit.md`: replacing `g = g + shift_tg` with
`g = g + 0.0 * shift_tg` fails `test_sched_reparam_predictions_come_back_on_the_label_scale` (predictions come
back at 5.7–24.6 s against `ms` 10,800) and leaves the other three passing, which is the split those tests are
for. The mutation was reverted.

## 3. `make test` and `make lint`

```
$ make test
.venv/bin/python -m pytest -q
....................................................................     [100%]
68 passed in 35.96s

$ make lint
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
All checks passed!
27 files already formatted
```

(68 tests: the 63 that existed when this work started, the four above, and one a concurrent session added to
`tests/test_compare.py` while it was in flight — an earlier full run of the same suite, before that test
existed, reported 67 passed in 165 s; see §6.)

## 4. Measurement

Protocol: `.venv/bin/python -m prc.pipeline --config <yaml> --skip-predict`, `POLARS_MAX_THREADS=2`,
LightGBM `num_threads: 2`, one run at a time, no ranking prediction and no submission. Feature cache reused
(`data/work/feat_train.parquet`, 2,085,047 rows).

| side | config | exp_id | forward RMSE full | clean | rounds matched / unmatched | wall clock |
|---|---|---|---|---|---|---|
| A (baseline) | `configs/inv_baseline_forward.yaml` (unchanged) | `inv_baseline_forward-9271caca22` | **284.70384510819986** | 224.57938866638986 | 939 / 45 | 1,108.6 s |
| B (variant) | `configs/reparam_lirf.yaml` (new) | `reparam_lirf-41526561b0` | **223.56973316351826** | 209.42469379730127 | 939 / 64 | 1,186.6 s |

`configs/reparam_lirf.yaml` differs from `configs/inv_baseline_forward.yaml` in exactly three keys: `name`,
`unmatched_sched_reparam: true`, `unmatched_sched_reparam_airports: [LIRF]`.

### 4.1 The baseline rerun reproduces the logged baseline

`inv_baseline_forward-9271caca22` (this code, flag off) against `inv_baseline_forward-67e76a5dfc` (the
investigation's record, code before this change): **every logged metric is equal to all 17 printed digits** —
`rmse_full` 284.70384510819986, `rmse_clean` 224.57938866638986, `bias` 4.077331453401098,
`top20_share_of_sse` 0.4464064940663706, `n_gt_7200` 61, `rounds_matched` 939, `rounds_unmatched` 45 — and
327,983 of the 328,009 predictions are bit-identical. The other 26 are unmatched rows differing by at most
**4.547e-13** (≈ 1 ulp at these magnitudes); no matched row differs at all. So the flag-off path is the old
model to within last-bit floating point on real data, which is what the in-repo no-op tests cannot show.

### 4.2 `make compare A=inv_baseline_forward-9271caca22 B=reparam_lirf-41526561b0`, verbatim

```
$ make compare A=inv_baseline_forward-9271caca22 B=reparam_lirf-41526561b0
.venv/bin/python -m prc.compare inv_baseline_forward-9271caca22 reparam_lirf-41526561b0
A = inv_baseline_forward-9271caca22
B = reparam_lirf-41526561b0
rows = 328,009
top-20 share of B's squared error: 0.134

 dropped         A         B     diff                 CI95   P(B)
       0    284.70    223.57   -61.13    [-106.01, -22.87]  1.000
      20    212.00    210.82    -1.19    [  -2.00,  -0.33]  0.998
     200    198.46    198.09    -0.38    [  -0.63,  -0.14]  0.994
    2000    177.35    177.26    -0.09    [  -0.16,  -0.02]  0.998

month-level bootstrap (2 months): -61.13 s [-62.14, -60.14]  P(B better) = 1.000

   matched: n=325,004  A=   201.7  B=   201.7
 unmatched: n=  3,005  A=  2109.3  B=  1028.4

tail structure (reporting only - the VERDICT below is computed by verdict() from the curve alone,
and none of these lines change adopt/reject):

  unmatched rows            n=  3,005  A=   2109.3  B=   1028.4   -1080.91 [ -1590.83,   -548.98]
  unmatched @ LIRF          n=    140  A=   8578.4  B=   4124.6   -4453.76 [ -7092.61,  -1152.32]   72.8% of the unmatched disagreement

  unmatched difference by airport (ordered by share of sum (p_a - p_b)^2):
     airport       n         A         B      diff                   CI95  gap share
        LIRF     140    8578.4    4124.6  -4453.76    [-7092.61,-1152.32]     72.8%
        EHAM     545    1184.0     343.7   -840.35    [-1291.96, -237.29]      8.6%
        LFPG     685    1217.0     708.9   -508.05    [ -969.40,  -15.07]      8.0%
        EDDF     188    1683.3     507.6  -1175.72    [-2025.94,   +3.24]      5.8%
        EDDM     165    1349.8     855.3   -494.52    [-1179.04,   -7.05]      2.1%
        LEBL     156    1150.4     383.2   -767.21    [-1498.55,  +42.32]      1.9%
        EGLL     181     727.0     584.2   -142.76    [ -379.26,  +22.91]      0.5%
        LTFM     640     420.2     364.6    -55.55    [  -98.47,  -23.24]      0.3%
        LSZH     198     366.8     418.2    +51.36    [  -12.67, +117.72]      0.1%
        LEMD     107     360.4     364.7     +4.31    [   -7.95,  +18.54]      0.0%

  the K=200 drop removes 200 rows (99 unmatched, 101 matched) - these are the rows the K=200 CI is blind to:
    rank     MVT_ID_mvt  airport       grp          y        p_a        p_b
       0      200297323     LIRF unmatched    87480.0    32351.7    57514.2
       1      201198951     LIRF unmatched    87605.0    34093.3    88147.3
       2      200300302     LIRF unmatched    87598.0    34539.7    61276.7
       3      201196455     LIRF unmatched    39120.0    20890.4    38275.0
       4      199363411     EDDF unmatched      661.0    16880.0     2206.1
       5      201197366     LIRF unmatched    47767.0    32399.5    45945.6
       6      201102986     LFPG unmatched      652.0    15811.9     2155.2
       7      199512591     LFPG unmatched     1265.0    16423.1     2221.3
       8      201228485     EDDF unmatched     1080.0    16015.5     1953.7
       9      200741002     EHAM unmatched      549.0    15335.1     1830.9
      10      201098155     LFPG unmatched      657.0    14800.4     1744.0
      11      200467590     LEBL unmatched      911.0    14715.4     3248.4
      12      200738773     EHAM unmatched      852.0    14211.7     1702.5
      13      201042062     EDDM unmatched      790.0    14011.2     1948.7
      14      200728615     EHAM unmatched     1431.0    14436.0     2413.7
      15      199228497     EHAM unmatched      703.0    12949.8     1716.2
      16      200304988     LIRF unmatched    37736.0    25511.5    39305.4
      17      201193674     LIRF unmatched      782.0     9001.7    12079.9
      18      201196534     LIRF unmatched    38633.0    29713.7    42126.7
      19      200297679     LIRF unmatched      710.0     6670.4     8588.3
      20      201099330     LFPG unmatched     9182.0     1533.3     1435.9
      21      199350134     EDDF unmatched     6907.0     1190.6     1089.8
      22      201049509     EDDM   matched     1810.0     7574.9     7580.3
      23      200307345     LIRF unmatched    12371.0     6755.8     7236.8
      24      200298723     LIRF unmatched     6786.0     2200.1     1211.8
      25      201195389     LIRF unmatched     9848.0     4465.0     4857.4
      26      201194517     LIRF unmatched    10264.0     4894.8     5268.5
      27      201113278     LFPG unmatched     7392.0     2116.3     2252.7
      28      201188030     LIRF unmatched     6727.0     1550.8     3324.1
      29      200303912     LIRF unmatched     9240.0     4525.7     4097.0
      30      199519773     LFPG unmatched     7863.0     2815.3     3252.2
      31      200772204     EGLL unmatched      971.0     5980.3     1512.4
      32      200299756     LIRF unmatched    29282.0    24317.9    28990.5
      33      201103619     LFPG   matched     6600.0     1929.1     1920.4
      34      201193985     LIRF unmatched     1203.0     5765.0     5577.4
      35      199515061     LFPG unmatched     6715.0     2217.4     2704.0
      36      201193968     LIRF unmatched    14471.0    10065.3    10758.3
      37      201193287     LIRF   matched     5763.0     1455.1     1439.4
      38      201196888     LIRF   matched     5579.0     1326.4     1328.2
      39      201198643     LIRF unmatched     7091.0     2864.4     5805.6
      40      201197122     LIRF unmatched      777.0     4147.7     4949.3
      41      201194712     LIRF   matched     5351.0     1183.9     1179.0
      42      201590336     LTFM unmatched     5170.0     1022.3     1118.2
      43      201104714     LFPG   matched     7805.0     3676.6     3663.3
      44      199522384     LFPG   matched     6355.0     2235.3     2254.7
      45      201196690     LIRF unmatched     5406.0     1319.9     1397.6
      46      201189416     LIRF unmatched      534.0     3326.1     4585.8
      47      201104572     LFPG unmatched     5514.0     1540.9     1747.7
      48      201422400     LSZH unmatched      801.0     3611.2     4762.6
      49      201187802     LIRF unmatched    12728.0     8821.9    12498.3
      50      200297206     LIRF unmatched     7917.0     4024.3     4956.6
      51      201195209     LIRF   matched     4972.0     1080.7     1088.9
      52      201190135     LIRF   matched     5273.0     1404.4     1395.8
      53      200765592     EGLL   matched    11452.0     7598.8     7577.9
      54      201195009     LIRF unmatched    10609.0     7098.6     6763.6
      55      199513736     LFPG   matched     6512.0     2690.0     2692.2
      56      201192942     LIRF unmatched    11291.0     7481.5     7598.3
      57      201193863     LIRF unmatched      597.0     2748.7     4373.7
      58      201194868     LIRF   matched     5350.0     1606.8     1623.5
      59      199828026     LTFM   matched     7318.0     3624.9     3631.1
      60      200296422     LIRF   matched     5091.0     1398.1     1407.5
      61      200298915     LIRF   matched     5107.0     1426.2     1425.8
      62      200304433     LIRF   matched     4799.0     1125.2     1128.6
      63      201101018     LFPG unmatched     1201.0     3998.1     4804.2
      64      200300041     LIRF unmatched      592.0     3823.3     4172.6
      65      200299334     LIRF   matched     4620.0     1042.7     1040.2
      66      201049508     EDDM unmatched     4918.0     1344.3     1431.9
      67      200306193     LIRF unmatched    11392.0     7842.9     8436.2
      68      199837369     LTFM   matched     4375.0      851.6      845.9
      69      201049525     EDDM   matched     1609.0     5095.9     5105.6
      70      201194154     LIRF unmatched     9611.0     6120.7     7608.6
      71      200306524     LIRF   matched     4451.0      970.2      970.3
      72      201192087     LIRF   matched     5692.0     2243.1     2227.1
      73      201197309     LIRF   matched     4556.0     1093.9     1096.2
      74      201113667     LFPG   matched     4449.0     1068.6     1070.5
      75      200306565     LIRF   matched     4375.0     1001.1     1000.7
      76      201193143     LIRF   matched     4810.0     1454.8     1456.7
      77      200302090     LIRF unmatched    10738.0     7395.1     8396.8
      78      200300604     LIRF   matched     4205.0      895.9      893.6
      79      201049441     EDDM unmatched     4734.0     1431.8     1465.0
      80      199363893     EDDF   matched     5699.0     2401.5     2407.4
      81      200296808     LIRF   matched     4744.0     1459.0     1471.2
      82      200303132     LIRF   matched     4381.0     1105.4     1098.1
      83      201581606     LTFM   matched     5216.0     1936.3     1934.4
      84      201190360     LIRF   matched     5104.0     1880.9     1877.1
      85      199519932     LFPG unmatched     6066.0     2852.1     3626.6
      86      201189421     LIRF unmatched     1079.0     3654.5     4281.7
      87      201188459     LIRF   matched     4372.0     1179.6     1183.2
      88      201198068     LIRF   matched     4990.0     1799.7     1810.2
      89      201581779     LTFM unmatched     5161.0     1973.3     2241.3
      90      201193984     LIRF unmatched     1011.0     4094.4     4176.3
      91      200302945     LIRF unmatched     1203.0     3195.0     4338.3
      92      201104273     LFPG   matched     3864.0      736.4      739.3
      93      200298668     LIRF unmatched     1077.0     3763.2     4196.2
      94      200770074     EGLL unmatched     4444.0     1477.7     1371.4
      95      199837241     LTFM unmatched     4203.0     1322.3     1153.1
      96      201198682     LIRF unmatched     1073.0     4113.4     4109.4
      97      200768057     EGLL unmatched      770.0     3800.5     1895.4
      98      201194106     LIRF unmatched    10382.0     7352.6     8910.7
      99      200306771     LIRF unmatched      899.0     3459.8     3926.5
     100      201113330     LFPG   matched     4863.0     1844.9     1851.1
     101      199750336     EDDM   matched     3664.0      648.3      647.3
     102      200302128     LIRF   matched     4195.0     1205.4     1215.8
     103      200301768     LIRF unmatched      776.0     3523.8     3760.7
     104      201189027     LIRF   matched     5710.0     2754.0     2738.9
     105      201197191     LIRF   matched     4745.0     1830.6     1806.9
     106      201196892     LIRF unmatched     4510.0     1572.9     2691.0
     107      200777440     EGLL   matched     1191.0     4095.2     4104.7
     108      201192008     LIRF   matched     4430.0     1518.7     1518.1
     109      200301220     LIRF unmatched      667.0     3570.9     3350.8
     110      199509513     LFPG unmatched     4561.0     1671.9     1696.9
     111      201582853     LTFM unmatched      116.0     3001.7     1359.0
     112      200777040     EGLL   matched      666.0     3548.3     3550.4
     113      200766628     EGLL   matched     4311.0     1449.9     1448.6
     114      200767135     EGLL   matched     5268.0     2422.8     2426.5
     115      200301223     LIRF unmatched      949.0     3480.1     3785.0
     116      200297234     LIRF unmatched     5755.0     3559.8     2944.6
     117      199520650     LFPG   matched     1854.0     4634.6     4656.1
     118      199523823     LFPG unmatched     5050.0     2255.1     2484.1
     119      201195339     LIRF unmatched     1254.0     3045.5     4040.3
     120      201196882     LIRF   matched     3836.0     1068.2     1080.2
     121      201049517     EDDM unmatched     4212.0     1458.5     1529.6
     122      201107402     LFPG   matched     4253.0     1552.4     1541.4
     123      201111586     LFPG   matched     5035.0     7732.8     7739.7
     124      200298775     LIRF unmatched     5111.0     2415.2     2504.6
     125      199243246     EHAM unmatched     3129.0      534.2      434.7
     126      201199086     LIRF   matched     4430.0     1752.4     1759.6
     127      201197987     LIRF unmatched     1075.0     3451.2     3728.8
     128      199526384     LFPG unmatched     1131.0     3617.9     3783.0
     129      201596260     LTFM   matched     4025.0     1414.5     1427.1
     130      201584652     LTFM   matched     3782.0     1182.0     1182.4
     131      199522750     LFPG   matched     3619.0     1025.4     1022.5
     132      201193042     LIRF   matched     3180.0      583.7      586.8
     133      201192072     LIRF   matched     4861.0     2309.4     2306.6
     134      201198535     LIRF   matched     4671.0     2122.6     2124.8
     135      199522588     LFPG   matched     4034.0     1490.1     1492.6
     136      200302490     LIRF   matched     4031.0     1492.5     1491.8
     137      200305497     LIRF   matched     3844.0     1309.3     1307.0
     138      201196910     LIRF   matched     3830.0     1302.7     1316.9
     139      200302129     LIRF unmatched     1326.0     3123.2     3844.8
     140      201049443     EDDM   matched     4016.0     1502.0     1499.6
     141      200296316     LIRF   matched     3648.0     1141.0     1139.1
     142      200297581     LIRF   matched     3476.0      970.3      969.7
     143      200772600     EGLL   matched     2815.0     5298.2     5309.0
     144      200307420     LIRF   matched     6831.0     4338.7     4348.2
     145      201191040     LIRF   matched     4490.0     2029.1     2053.1
     146      201049519     EDDM unmatched     3661.0     1203.0     1262.5
     147      200297170     LIRF   matched      834.0     3286.4     3281.0
     148      200302675     LIRF unmatched      658.0     2526.2     3097.8
     149      199508947     LFPG unmatched     3708.0     1459.7     1271.3
     150      201049515     EDDM unmatched     3542.0     1108.0     1172.0
     151      201193739     LIRF   matched     4561.0     2127.7     2134.2
     152      200306397     LIRF   matched     3724.0     1299.3     1306.8
     153      200297037     LIRF unmatched     5041.0     2716.8     2619.4
     154      200736829     EHAM   matched     3664.0     1251.3     1250.6
     155      201192086     LIRF   matched     4390.0     1978.6     1983.9
     156      201198071     LIRF   matched     4674.0     2263.1     2263.3
     157      199832815     LTFM   matched     3649.0     1240.1     1241.6
     158      199508710     LFPG   matched     1848.0     4236.5     4245.2
     159      200764221     EGLL unmatched     2688.0     5081.3     2361.5
     160      201192178     LIRF unmatched     1131.0     3522.6     2785.6
     161      200301330     LIRF unmatched     8396.0     6005.7     6125.1
     162      201049507     EDDM unmatched     3852.0     1509.9     1470.3
     163      201199500     LIRF   matched     3469.0     1093.1     1101.9
     164      199511447     LFPG   matched     4069.0     1694.9     1707.2
     165      201043057     EDDM   matched     3415.0     1042.3     1041.9
     166      199514987     LFPG   matched     2942.0      570.1      580.4
     167      200299534     LIRF unmatched     5933.0     3567.0     3710.5
     168      200304543     LIRF   matched     1083.0     3447.6     3438.4
     169      201110150     LFPG   matched     3351.0      994.9     1000.2
     170      201196521     LIRF unmatched     7629.0     5291.2     5862.3
     171      201582535     LTFM   matched     3354.0     1021.0     1017.0
     172      200304516     LIRF unmatched     7494.0     5159.6     6107.3
     173      201581798     LTFM unmatched     4736.0     2412.1     2532.1
     174      200461566     LEBL   matched     3459.0     1142.9     1146.1
     175      201107623     LFPG   matched     5282.0     2988.2     2970.3
     176      201191934     LIRF   matched     3548.0     1253.3     1249.5
     177      199775216     LSZH   matched     3149.0      853.5      854.1
     178      199512912     LFPG   matched     3427.0     1148.0     1161.4
     179      201049445     EDDM   matched     3537.0     1271.6     1268.5
     180      199522557     LFPG unmatched     4328.0     2065.6     2421.9
     181      200306037     LIRF   matched     3911.0     1655.9     1664.3
     182      200777514     EGLL   matched     3657.0     1403.3     1407.9
     183      201106569     LFPG   matched     3586.0     1332.6     1341.5
     184      201581799     LTFM   matched     4323.0     2088.1     2076.0
     185      201188137     LIRF   matched     1259.0     3482.6     3502.4
     186      201198728     LIRF   matched     3009.0      769.6      767.5
     187      199517056     LFPG   matched     3606.0     1374.6     1378.2
     188      201113281     LFPG unmatched     3479.0     1250.8     1285.4
     189      201595211     LTFM unmatched      845.0     3067.6     1223.4
     190      201192725     LIRF   matched     3902.0     1682.7     1704.3
     191      199034447     EGLL   matched     3961.0     1746.1     1742.5
     192      201194664     LIRF   matched     3423.0     1205.7     1206.0
     193      201193210     LIRF unmatched     5583.0     3369.4     3619.6
     194      201109105     LFPG   matched     3716.0     1509.1     1527.5
     195      199525412     LFPG   matched     3371.0     1166.3     1176.5
     196      201193474     LIRF   matched     4572.0     2369.7     2381.7
     197      199509478     LFPG unmatched     2101.0     3006.5     4302.8
     198      199513944     LFPG   matched     2951.0      749.7      749.6
     199      201049477     EDDM   matched     3609.0     1407.8     1419.0

VERDICT (from the curve only, rule unchanged): ADOPT - adopt (-61.13 s, still -0.38 s at 200 rows dropped)
```

The `tail structure` block and the dropped-row listing come from the reporting block added to `prc.compare` by
a concurrent session (commit `0db8e9d`, §6); `verdict()` and the curve are unchanged.

### 4.3 Unmatched RMSE per airport (forward fold, 3,005 unmatched rows)

| airport | n | A | B | B − A |
|---|---|---|---|---|
| EDDF | 188 | 1,683.3 | 507.6 | **−1,175.7** |
| EDDM | 165 | 1,349.8 | 855.3 | −494.5 |
| EGLL | 181 | 727.0 | 584.2 | −142.8 |
| EHAM | 545 | 1,184.0 | 343.7 | −840.3 |
| LEBL | 156 | 1,150.4 | 383.2 | −767.2 |
| LEMD | 107 | 360.4 | 364.7 | **+4.3** |
| LFPG | 685 | 1,217.0 | 708.9 | −508.1 |
| **LIRF** | 140 | **8,578.4** | **4,124.6** | **−4,453.8** |
| LSZH | 198 | 366.8 | 418.2 | **+51.4** |
| LTFM | 640 | 420.2 | 364.6 | −55.6 |
| **all unmatched** | **3,005** | **2,109.32** | **1,028.41** | **−1,080.91** |
| matched | 325,004 | 201.6633 | 201.6601 | −0.0033 |

Eight of ten airports improve, two get worse (LSZH +51.4 on 198 rows, LEMD +4.3 on 107).

Subsets of the same 3,005 rows:

| subset | n | A | B |
|---|---|---|---|
| LIRF unmatched, `ms` ≥ 2 h | 58 | 13,119.3 | 5,973.5 |
| LIRF unmatched, `ms` < 2 h | 82 | 1,974.3 | 1,951.0 |
| LIRF unmatched, sched-filled (\|y − ms\| ≤ 60) | 99 | 3,597.4 | 1,858.1 |
| LIRF unmatched, not filled | 41 | 14,833.4 | 7,053.7 |
| non-LIRF unmatched | 2,865 | 1,034.7 | 527.2 |
| non-LIRF unmatched, `ms` ≥ 2 h | 596 | 2,132.3 | 857.4 |

### 4.4 Tail counts of the predictions

Labels on the fold: 10 rows ≥ 4 h, 4 rows ≥ 12 h, all of them unmatched and at LIRF.

| | pred ≥ 4 h | pred ≥ 12 h | non-LIRF pred ≥ 4 h | max unmatched prediction |
|---|---|---|---|---|
| A (baseline) | 17 | 0 | **8** | 34,539.68 |
| B (reparam) | **9** | **4** | **0** | **88,147.34** |

Every prediction ≥ 4 h is an unmatched row on both sides. B removes all eight non-LIRF predictions ≥ 4 h — the
`false-alarm-non-lirf.md` rows — and emits four predictions ≥ 12 h against four such labels; the largest is
88,147.3 on a row whose label is 87,605 (baseline predicted 34,093.3 there).

### 4.5 The whole-fold gain with model A held fixed

Model A's predictions are not bit-identical between the two runs (§6, item 8), although the change cannot
reach them. Substituting the baseline run's matched column into B isolates the model-B change:

| dropped | A | B (model A held fixed) | diff | CI95 | P(B) |
|---|---|---|---|---|---|
| 0 | 284.70 | 223.57 | −61.13 | [−106.00, −22.87] | 1.000 |
| 20 | 212.00 | 210.82 | −1.18 | [−1.99, −0.33] | 0.998 |
| 200 | 198.46 | 198.09 | −0.37 | [−0.62, −0.13] | 0.994 |
| 2000 | 177.35 | 177.24 | −0.11 | [−0.17, −0.04] | 0.998 |

i.e. the same curve as §4.2 to 0.02 s. On matched rows alone the two runs differ by −0.0033 s
[−0.0247, +0.0186], a CI centred on zero.

Unmatched rows alone (the 3,005 rows, `robustness_curve` over that subset):

| dropped | A | B | diff | CI95 | P(B) |
|---|---|---|---|---|---|
| 0 | 2,109.32 | 1,028.41 | −1,080.91 | [−1,590.83, −548.98] | 1.000 |
| 20 | 714.57 | 674.99 | −39.58 | [−68.70, −15.17] | 1.000 |
| 200 | 339.37 | 334.99 | −4.38 | [−13.43, +4.25] | 0.836 |
| 2000 | 84.40 | 74.81 | −9.59 | [−12.69, −5.71] | 1.000 |

## 5. Against the investigation's own numbers

| quantity | `mms-mixture-reparam.md` (jan+jul+forward, 8,378 rows) | this run (forward only, 3,005 rows) |
|---|---|---|
| baseline unmatched RMSE | 2,326.5 (its own repro; the OOF file 2,296.1; its forward fold 2,177.4) | 2,109.32 (the logged forward record, reproduced) |
| variant A unmatched RMSE | 1,632.1 pooled; **1,024.6 on the forward fold** | **1,028.41** |
| non-LIRF predictions ≥ 4 h | 10 → 0 | 8 → 0 |
| predictions ≥ 12 h | 4 → 15 (17 such labels) | 0 → 4 (4 such labels) |
| airports improved / worsened | 9 / 0 | 8 / 2 (LSZH +51.4, LEMD +4.3) |

The forward-fold figure the script measured for variant A (1,024.6) and the one the pipeline produced
(1,028.41) differ by **3.8 s (0.37 %)**.

## 6. What differs from the investigation script's measurement, and why

1. **Fold set.** The script pooled jan + jul + forward (8,378 unmatched OOF rows); this run is `folds:
   [forward]` only (3,005 unmatched rows), because `docs/README.md` §6 makes forward the deciding fold and the
   task asked for that config. Nothing here re-measures the pooled 1,632.1.
2. **Whole-fold numbers exist here and not there.** The script fitted model B only and never refitted model A,
   so it reported unmatched RMSE alone. The pipeline refits both, which is where 284.70 → 223.57 and the
   `make compare` VERDICT come from. Conversely the script's per-fold jan/jul figures have no counterpart here.
3. **Baseline identity.** The script compared against its own re-fit (2,326.5 pooled, 2,177.4 forward) because
   it could not reproduce the logged run bit-exactly at `num_threads` 2 against the OOF file's 4. This run's
   baseline is the logged forward record itself, reproduced to all printed digits (§4.1), so the comparison is
   against the same numbers `findings.md` §3.2 uses (2,109.32 / 201.6633 / 284.7038 / rounds 45).
4. **Null `mvt_minus_sched`.** The script asserted the column is null-free and shifted every LIRF row
   unconditionally. The shipped code skips a null row explicitly. On this data the two are the same code path
   (0 nulls among the 22,470 unmatched training rows, the 3,005 validation rows and the 5,290 unmatched
   ranking rows), so this cannot account for any difference in the numbers.
5. **Clipping and assembly.** The script clipped at 0 by hand and evaluated arrays; the pipeline goes through
   `_assemble` (id-join, `fill_null`/`fill_nan` with the training median, `clip(lower_bound=0)`). No unmatched
   prediction was negative on either side, so this too is not a source of difference.
6. **Round counts.** Model B early-stops at 45 rounds in the baseline and **64** under the reparameterisation —
   the target it stops on is a different one, so the round count is not comparable between the two sides. The
   script reported the same behaviour per fold but its numbers are for its own baseline.
7. **`prc.compare` is not the version the investigation ran.** A concurrent session committed a reporting
   block to `src/prc/compare.py` (`0db8e9d`, 2026-09-17 05:13) while this work was in flight; `make compare`
   therefore prints a `tail structure` section and the K=200 drop listing that `findings.md` §3.2's runs did
   not print. `verdict()`, `robustness_curve` and the drop order are unchanged by that commit, so the curve
   and the VERDICT line are comparable with the earlier items.
8. **Model A moved between the two pipeline runs, and it should not have.** `rounds_matched` is 939 on both
   sides and matched RMSE moves only 201.6633 → 201.6601, but 324,995 of 325,004 matched predictions differ
   (mean \|diff\| 2.19 s, max 48.15 s). The reparameterisation cannot reach model A: it is fitted and predicted
   before the model-B branch, and the shift is applied inside that branch. Two facts bound it: the flag-off
   rerun (§4.1) reproduced the September-16 baseline's matched rows **bit-identically**, so the code change is
   not what moves them; and holding model A fixed reproduces the whole curve to 0.02 s (§4.5), so it does not
   move the verdict. What is *not* established is the mechanism. One procedural difference from the
   investigation protocol is known: read-only analysis jobs (parquet reads) were run on this machine **during**
   the reparam run and not during the baseline run, where the investigation scripts ran one fit at a time with
   nothing else running, and `src/prc/train.py`'s own docstring records that machine load has changed
   LightGBM's behaviour in this repository before. That is a candidate, not a measurement.
9. **`exp_id`s of unchanged configs change.** `TrainConfig.hash()` hashes `__dict__`, so adding two fields
   renames every experiment: `main-48c19bc830` would be logged under a new id by the next run of
   `configs/main.yaml`. The model is unchanged (§4.1); the id is not.
10. **Nothing was submitted, no ranking prediction was run, no existing config was modified, and no commit was
    made by this work.** The files were nevertheless committed in `0db8e9d` / `78a10d2` by the concurrent
    session that shares them.

## 7. Facts and inference, separated

**Facts (measured here).** Everything in §3, §4 and §5: the two experiment records and their metrics; the
327,983 / 26 / 4.547e-13 reproduction of the logged baseline; the `make compare` output as printed; the
per-airport and per-subset unmatched RMSEs; the tail counts and the maximum unmatched prediction; the
model-A-held-fixed curve and the matched-only CI; the unmatched-only curve; the round counts 45 → 64; the
1,348 / 140 shifted-row counts; the null rates of `mvt_minus_sched`; the test and lint output; the mutation
check.

**Inference (not measured here).** That the K=200 and K=2000 whole-fold gains are *carried by* the unmatched
rows — the matched-only CI centred on zero and the model-A-held-fixed curve are consistent with it, but no
ablation isolates each path separately. That the model-A difference of item 8 is caused by machine load; only
its size and its irrelevance to the curve were measured. That the LIRF gain comes from the model no longer
having to represent `ms`-sized targets (the co-movement with the eight vanished non-LIRF false alarms is
consistent, no tree was inspected). That any of this transfers to the 2026 ranking rows: no 2026 label exists,
and `mms-mixture-reparam.md` §5 already lists the fill rate of the 383 LIRF unmatched ranking rows as unknown.

**Out of scope by instruction.** No submission, no ranking prediction, no change to `configs/main.yaml`
(the flag is off there by default), no adoption decision — §4.2 prints the rule's verdict, it does not act on it.

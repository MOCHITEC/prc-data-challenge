# Runbook: the final upload on 2026-10-14

> **Status: done.** `resilient-hedgehog_final.parquet` was uploaded on 2026-10-14; the bucket object's
> `last_modified` is **10:30:29Z**, md5 **`c0d4a1439862e68852b65f6854322a29`** (2,151,179 bytes), read-back verified.
> The file is `handoff/2026-10-13/final_v5_plus_febjun_appear.parquet` from `ea0e0a8` (`docs/FINAL_SUBMISSION.md` §1).
> The ledger lines were committed in `8189b58`. The scorer wrote its result file at 10:30:38Z: `"status":
> "Succeeded"`, `"used_pairs": 670790`, no score (`final_result_2026-10-14.json` next to this runbook).
> The organisers' pages state the deadline as "2026-10-14T12:00 CET (2026-10-14T12:00+02:00)" and "2026-10-14T10:00Z";
> the upload was at 10:30:29Z, after it. The steps below are kept as the procedure, with the values of that upload.
> Until 10:30Z on 10-14 this runbook targeted the base file `be8d89dc…` (`25f6246`), which was not uploaded.

Who and when (`NO_FINAL_UPLOAD.md`):
- **Only the RPC thread** uploads, on branch `claude/model-b-and-arrival-features`.
- Upload **after 13:00 JST (04:00Z)** on 10-14. The deadline is **19:00 JST (10:00Z)**.

What is uploaded:

| item | value |
|---|---|
| object | `prc-2026-resilient-hedgehog/resilient-hedgehog_final.parquet` |
| file | `final_v5_plus_febjun_appear.parquet`: the v48-based file `be8d89dc…` (Jan/Jul = v48's file, Feb/Jun = the laughing-cray chain, June on our own ADS-B layer on 29 of 30 days) plus the 12-month appear increment on 103,311 Feb and 123,624 Jun rows. The file's identity is stated in `docs/FINAL_SUBMISSION.md` §1; this table and §0 copy it |
| source | `ea0e0a85bec37c0a98b2a14f663c1742303f6813:handoff/2026-10-13/final_v5_plus_febjun_appear.parquet`, branch `origin/claude/beautiful-davinci-ehugk4` (contained in `main`), commit "Final candidate on the integrator's v5: appear increment on Feb and Jun rows (06-11 excluded), so Feb/Jun follow the v48 recipe". Restored at the same path on `main` in `0d43558` |
| md5 | `c0d4a1439862e68852b65f6854322a29` (2,151,179 bytes) |
| not these | The base `handoff/2026-10-13/final_from_v48.parquet` @ `25f6246`, md5 `be8d89dc…` (2,145,836 bytes): this runbook's target until 10:30Z on 2026-10-14, not uploaded. The same path in `558c736`, md5 `c7ffa31e…` (2,145,277 bytes), June on the A12 layer: the target until earlier on 2026-10-14. |
| template | `final_submitting.parquet` from `prc-2026-datasets`, 670,790 rows, md5 `2c1f407296e217fd7d242c9d51ba27ad` |
| months | `final_ranking.parquet`, md5 `1ce0c6ee9ea15b8c3d0cd282591d4e0a` |

Before the upload the object **already existed** (it has now been replaced):
- 4,079,671 bytes, last modified 2026-10-09 16:02:32Z, etag `7f95f323928691c9e167e1a1b330f40b`.
- Nobody knows who uploaded it.
- Its scorer sidecar `resilient-hedgehog_final.parquet_result.json` (16:32:57Z) says `"status": "Failed"`,
  `"error_type": "FileNotFoundError"`, `"error": "s3://prc-2026-datasets/dev_sub_template.parquet"`. That is a missing
  file on the organisers' side, not a fault in the uploaded file.
- **The upload had to replace it deliberately**, with `--confirm-final-overwrite`. It did (the ledger line has
  `"overwrite_confirmed": true` and the old object's size, mtime and etag).

All timings below were measured in the dry run on 2026-10-13 around 01:10–01:25Z, in this container.

---

## 0. Shell variables (every step uses them)

```bash
export WT=/tmp/claude-0/-home-user-prc-data-challenge/ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad/wt_modelb2
export OSN_TEAM_BUCKET=prc-2026-resilient-hedgehog
export PRC_TEAM_NAME=resilient-hedgehog
export OSN_COMPETITION_BUCKET=prc-2026-datasets
export SRC_COMMIT=ea0e0a85bec37c0a98b2a14f663c1742303f6813
export SRC_PATH=handoff/2026-10-13/final_v5_plus_febjun_appear.parquet
export FINAL_MD5=c0d4a1439862e68852b65f6854322a29
# The credentials come from the container environment. Check they are set; never print them:
env | grep -E '^OSN_(ID|ACCESS_KEY|SECRET_KEY)=' | sed 's/=.*/=<set>/'
```

- The three names and buckets are not secrets. They are not set in the container, so export them yourself.
- Their source is `docs/decisions.md` on `claude/prc-data-challenge-planning-l7dk7k`: "公式データ `prc-2026-datasets` …
  提出先 `prc-2026-resilient-hedgehog`。チーム名は `resilient-hedgehog`".
- The ledger confirms them: `object_name: resilient-hedgehog_v46.parquet`.
- A listing on 10-13 showed exactly the two buckets `prc-2026-datasets` and `prc-2026-resilient-hedgehog`.
- If `OSN_*` is missing, the container's environment secrets were not loaded. Set them in the environment settings;
  never paste them into chat.

## 1. Worktree (only if `$WT` is missing; about 5 s)

```bash
cd /home/user/prc-data-challenge
git fetch origin claude/model-b-and-arrival-features claude/laughing-cray-vvwh0k
git worktree prune
git worktree add -B claude/model-b-and-arrival-features "$WT" origin/claude/model-b-and-arrival-features
cd "$WT" && git pull --no-rebase && git log --oneline -3
```

- If the worktree exists, run `cd "$WT" && git pull --no-rebase` only.
- If even `/home/user/prc-data-challenge` is gone, clone the repo first. It is `MOCHITEC/prc-data-challenge`.

## 2. Environment (.venv; 1 min 34 s measured)

```bash
cd "$WT"
uv venv -q .venv
uv pip install -q --python .venv/bin/python -e ".[dev]"
uv pip install -q --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python -c "import prc, lightgbm, polars; print('env ok')"
```

- These are the `make bootstrap` steps without its full-bucket download.
- `uv` 0.11.32 and Python 3.11.17 were used.

**Tests (optional):**
- The final tests alone take about 5 s:

  ```bash
  .venv/bin/python -m pytest -q tests/test_submit_final.py tests/test_validate_final.py tests/test_submit_guard.py
  ```

- The full suite (`.venv/bin/python -m pytest -q`) takes about 5 min. Skip it if time is short.

## 3. Competition files, read-only (16.8 s measured for 132 MB)

```bash
cd "$WT"
.venv/bin/python -c "
import logging; logging.basicConfig(level=logging.INFO)
from pathlib import Path
from prc.io_minio import download_files
for p in download_files(bucket='prc-2026-datasets', dest=Path('data/raw/final')): print(p, p.stat().st_size)
"
md5sum data/raw/final/*
```

- `download_files` only GETs objects. It fetches `final_submitting.parquet`, `final_ranking.parquet` and
  `ranking.parquet`.
- The md5 values on 10-13 were:
  - `final_submitting` = `2c1f407296e217fd7d242c9d51ba27ad`;
  - `final_ranking` = `1ce0c6ee9ea15b8c3d0cd282591d4e0a`;
  - `ranking` = `80a5d97cdbcb25b768558feb71287517`.
- If `final_submitting` or `final_ranking` now has a different md5, the organisers changed the files. Stop and re-check
  that the ids still match (step 5 fails if they do not).

## 4. The final file and its md5 (about 1.3 s fetch, 0.03 s extract)

```bash
cd "$WT"
git fetch origin claude/beautiful-davinci-ehugk4
mkdir -p data/final_candidate
git show "$SRC_COMMIT:$SRC_PATH" > data/final_candidate/final.parquet
md5sum data/final_candidate/final.parquet   # must print $FINAL_MD5 (c0d4a1439862e68852b65f6854322a29)
```

## 5. Validation (2.6 s measured)

```bash
cd "$WT"
.venv/bin/python -m prc.validate_final data/final_candidate/final.parquet \
  --template data/raw/final/final_submitting.parquet --ranking data/raw/final/final_ranking.parquet \
  --expect-md5 "$FINAL_MD5" --json outputs/final/validate_final.json; echo "exit=$?"
```

It printed **PASS, exit 0** on 10-14 for `c0d4a143…` and for its base `be8d89dc…`, and on 10-13 for the earlier
`c7ffa31e…`. The values below are for `c0d4a143…`. The base `be8d89dc…` differs only in the Feb p50 / p99 (922 /
2,198) and the Jun p99 (2,066); `c7ffa31e…` has Jun p50 925 and p99 2,073.

**Shape and values**
- 670,790 rows, equal to the template.
- Schema `MVT_ID_mvt: double, TAXITIME_SEC_mvt: int32`, the same as the template.
- `MVT_ID_mvt` is identical to the template in set and order.
- 0 nulls and 0 negatives.
- min 0, max 112,725 s.
- 66 rows are above 10,800 s. This is a warning, not a failure.

**Per month**

| month | rows | p50 (s) | p99 (s) | rows > 10,800 s |
|---|---|---|---|---|
| Jan | 152,719 | 970 | 3,095 | 15 |
| Feb | 144,158 | 921 | 2,216 | 11 |
| Jun | 181,791 | 926 | 2,075 | 9 |
| Jul | 192,122 | 936 | 2,138 | 31 |

**Per airport.** All 10 airports have rows in all 4 months.

**Exit code.** Exit 1 on any hard failure. Do not upload a file that fails.

## 6. Dry run (3.1 s measured)

```bash
cd "$WT"
.venv/bin/python -m prc.submit --final --dry-run --confirm-final-overwrite \
  --file data/final_candidate/final.parquet \
  --template data/raw/final/final_submitting.parquet --ranking data/raw/final/final_ranking.parquet \
  --source-commit "$SRC_COMMIT" --source-path "$SRC_PATH" --expect-md5 "$FINAL_MD5" \
  --note "dry run 2026-10-14 before the final upload"; echo "exit=$?"
```

**What it does:**
- runs the step 5 gate;
- checks that the local file equals `git show $SRC_COMMIT:$SRC_PATH`;
- HEADs the target and prints its size, mtime and etag;
- writes one `"status": "dry_run"` line to `experiments/submissions.jsonl`;
- uploads nothing.

**Expected output:** `target: … EXISTS: 4,079,671 bytes, last modified 2026-10-09 16:02:32+00:00, etag 7f95f323…`,
then `DRY RUN: would upload …`, `status: dry_run`, and exit 0.

**Without `--confirm-final-overwrite`** it prints the same target line, then `REFUSED: … already exists`, and exits 1.
Both outcomes were seen on 10-13.

**If the size, mtime or etag differ from the values above**, someone else has uploaded since 10-13. Stop and find out
before going on.

## 7. The real upload (done 2026-10-14 10:30:29Z)

```bash
cd "$WT"
.venv/bin/python -m prc.submit --final --confirm-final-overwrite \
  --file data/final_candidate/final.parquet \
  --template data/raw/final/final_submitting.parquet --ranking data/raw/final/final_ranking.parquet \
  --source-commit "$SRC_COMMIT" --source-path "$SRC_PATH" --expect-md5 "$FINAL_MD5" \
  --note "final: v48-based be8d89dc + Feb/Jun appear increment (final_v5_plus_febjun_appear, ea0e0a8)"; echo "exit=$?"
```

**Order of operations:**
1. the gate;
2. the source-blob check;
3. HEAD and print;
4. the confirm check;
5. a write-ahead `"status": "attempted"` ledger line;
6. a second HEAD, refusing if the object is not the one printed in 3;
7. the PUT;
8. GET it back into a temp dir and compare md5;
9. a `"status": "uploaded", "verified": true` ledger line with `remote_md5` and the new etag.

**Duration.** About 2 s on 10-14: the `attempted` line has `uploaded_at` 10:30:27Z and the object's
`last_modified` is 10:30:29Z.

**Exit codes**

| code | output | meaning | what to do |
|---|---|---|---|
| 0 | `UPLOADED and verified: … md5 c0d4a143…` | done | step 8 |
| 1 | `REFUSED: …` | nothing was uploaded | Read the reason. "changed since it was shown" means another PUT landed between the two HEADs: find out who, then re-run step 6 and step 7. A gate or md5 failure means the wrong file: redo step 4. |
| 2 | `REFUSED: cannot build the OSN config` | credentials are missing | step 0 |
| 3 | `VERIFY FAILED` | the PUT happened, but the read-back md5 differs or the read-back failed | Re-run step 7 once. `--confirm-final-overwrite` is needed again, because the object is now ours. If it fails again, run step 8 by hand and report. |
| traceback after "attempted" | — | the PUT itself failed (network); the ledger has `attempted` and then `failed` | Re-run step 7. |

## 8. Post-upload verification (a few seconds, then wait for the scorer)

```bash
cd "$WT"
.venv/bin/python -c "
import tempfile
from pathlib import Path
from prc.config import OsnConfig
from prc.io_minio import client, download_team_object, final_object_name, stat_submission
from prc.validate_final import file_md5
cfg = OsnConfig.from_env(); name = final_object_name(cfg); c = client(cfg)
st = stat_submission(name, cfg, c); print('HEAD', cfg.team_bucket, name, st.size, st.last_modified, st.etag)
with tempfile.TemporaryDirectory() as t:
    print('md5 of the bucket object:', file_md5(download_team_object(name, Path(t) / name, cfg, c)))
for o in c.list_objects(cfg.team_bucket, prefix=name, recursive=True): print(o.object_name, o.size, o.last_modified)
r = c.get_object(cfg.team_bucket, name + '_result.json'); print(r.read().decode()); r.close(); r.release_conn()
"
```

**What to check**
- The md5 of the bucket object must be `$FINAL_MD5` (`c0d4a1439862e68852b65f6854322a29`).
- The size must be 2,151,179 bytes.
- The scorer sidecar `resilient-hedgehog_final.parquet_result.json` must have a `last_modified` **after** the upload.
  - On 10-09 the scorer wrote it 30 minutes after the upload (16:02 → 16:32Z). On 10-14 it wrote it 9 s after
    (10:30:29 → 10:30:38Z), with `"status": "Succeeded"` and `"used_pairs": 670790`.
  - Re-run the block until it shows a new sidecar.
  - **Until then, the printed sidecar is the old 10-09 "Failed" one.**
- If the new sidecar says `Failed` again with the same `dev_sub_template.parquet` error, the fault is on the
  organisers' side. Report it to them; do not re-upload a different file.

**Record it.** The ledger lines must be committed:

```bash
cd "$WT"
git add experiments/submissions.jsonl
git commit -m "Final upload: resilient-hedgehog_final.parquet = final_v5_plus_febjun_appear @ ea0e0a8 (md5 c0d4a143…), verified"
git push -u origin claude/model-b-and-arrival-features || { git pull --no-rebase && git push -u origin claude/model-b-and-arrival-features; }
```

Add the session's attribution lines to the commit message.

## 9. If the container was reset

Everything outside git is lost: the `.venv`, `data/` and the exported variables. The worktree under `/tmp/…/scratchpad`
may be gone too. Redo the steps in order:

| step | time |
|---|---|
| 0, variables | instant |
| 1, worktree | about 5 s |
| 2, .venv | about 1.5 min |
| 3, competition files | about 17 s |
| 4, final file | about 2 s |
| 5, validation | about 3 s |
| 6, dry run | about 3 s |

That is about 2 minutes, then step 7 and step 8.

The tools are committed on this branch:
- `src/prc/validate_final.py`;
- `src/prc/submit.py` (`--final`);
- `src/prc/io_minio.py` (`download_files`, `upload_final`, `download_team_object`).

`experiments/submissions.jsonl` is in git. A dry-run or upload line that was not committed before the reset is lost.
Commit after step 6 and step 7.

## Facts measured on 10-13 (for the record)

- **The final file's ids match `final_submitting.parquet` exactly**: 670,790 ids, equal in order, and no
  duplicates (checked on `final_from_v48.parquet` on 10-13; `c0d4a143…` has the same ids in the same order).
- **Jan/Jul departures, `final_ranking` vs `ranking.parquet`**, joined on `MVT_ID_mvt`. Each of the 344,841 rows is
  present in both files.
  - `MVT_TIME_UTC_mvt` differs on **316,651 rows (91.83 %)**; 28,190 are unchanged.
  - Every difference is a whole number of seconds within ±12 s. Mean +0.01 s, sd and RMS 4.92 s.
  - Among the changed rows, the median |d| is 4 s, p95 is 9 s, and the max is 12 s.
  - The histogram is triangular: about 26,000 rows at ±1 s, falling to about 450 rows at ±12 s.
  - The share is the same everywhere: 91.6–92.0 % changed at each of the 10 airports, and in each month (Jan
    140,360 of 152,719; Jul 176,291 of 192,122). The 5 s-grid airports (EDDM, EGLL, LFPG, LIRF, LTFM) also move in 1 s
    steps.
  - Arrivals show the same pattern: 317,088 of 344,693 rows changed.
  - `SCHED_TIME_UTC_mvt`, `AOBT_3_flt` and `LOBT_flt` are unchanged on all Jan/Jul departures.
- **Test suite** (10-13, after 7673584): 314 passed, 2 skipped in 4 min 56 s. Before the final tooling: 287 passed, 2 skipped. 27 new tests: 10 in `test_validate_final.py`, 17 in `test_submit_final.py`; every bucket call there is mocked.

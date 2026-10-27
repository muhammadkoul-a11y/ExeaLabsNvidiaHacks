# OPS-3a: time and cost report

`python -m arrival.eval.cost [--out out] [--usage usage.json] [--pricing configs/pricing.example.yaml] [--baseline teleop_baseline.json] [--json]`

Reads `out/reports/*.json`, `out/eval/*.json` and `out/jobs/*/status.json` (finished jobs only) and prints a markdown table, or the report as JSON with `--json`. Exit code 2 and an `error:` line for an unusable pricing, usage or baseline file.

## Rows

One row per source and room, and per family where the file has families. Columns: attempts, successes, drop rate, seconds, demos per hour, seconds per successful demo, attempts per successful demo. Local CPU wall seconds is the same seconds figure, kept as its own key with no price on it. A value the file does not hold is null and prints `-`: sim reports written before per-family `seconds` existed, and merged shard reports that include one, print `-` for per-family throughput. The three sources describe overlapping runs (a job wraps a sim report); never add across them.

## Dollars

Same keys as the report schema in `docs/04-contracts.md`. `local_cpu` is `$0, measured`. A product row has a number only when `--usage` supplies its units. The unit rates come from the pricing file, and the placeholders in `configs/pricing.example.yaml` are marked `assumed, not invoiced`. Such a row has `measured: false` and `estimate: true`; `measured: true` needs units from the usage file and `basis: invoiced` on the pricing line. Everything else stays `usd: null, measured: false`.

| usage key | report key |
|---|---|
| `gpu_hour` | `serverless_jobs` |
| `cpu_hour` | `endpoint` |
| `object_storage_gb_month` | `object_storage` |
| `token_factory_per_mtoken` | `token_factory` |

`cosmos` has no pricing entry and stays unmeasured.

For a generated-versus-teleop **USD per demo** comparison, the usage file must also identify the measured room and
task. Product totals without a scope still appear in the dollars table, but are not divided by a task's successes:

```json
{"scope": {"room": "hero_kitchen", "task": "counter_to_sink"}, "gpu_hour": 2}
```

Use `"task": "all_tasks"` only when the usage covers the entire room-level report. Record units for this scope only;
the report divides their priced total by the matching generated success count.

## Teleop baseline

`out/teleop_baseline.json` (or `--baseline`):

```json
{"task": "counter_to_sink", "room": "hero_kitchen", "n_success": 10, "operator_seconds_total": 1500,
 "setup_assumptions": ["one operator", "VR rig owned"], "cost_basis_usd": 50}
```

`n_success` is an integer >= 1, `operator_seconds_total` > 0, `cost_basis_usd` >= 0 is the total operator and setup cost, `setup_assumptions` a non-empty string or list. Anything else raises `CostError` naming the field. With no baseline the output says `comparison not measured` and carries no ratio. With one, the comparison uses the baseline room's report rows (the baseline task's family when the sim report has one, else the room total) and prints both sample sizes. Generated seconds are local wall time of the generate call, teleop seconds are operator time. Generated USD per demo is null without a matching usage scope, and marked estimate unless every priced product is measured.

The report fields `teleop_baseline` and `dollars` follow the existing schema; `arrival/eval/report.py` is unchanged.

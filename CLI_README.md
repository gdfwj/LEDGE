# CLI and result reference

Install the dependencies as described in [README.md](README.md#install-and-start), then run the commands below from this directory. The [paper examples](README.md#paper-examples) provide ready-to-run configurations.

## CLI guide

```sh
python -m simulator --help
python -m simulator serve --help
python -m simulator run --help
```

| Command / option | Meaning |
|---|---|
| `serve` | Start the local GUI server. |
| `serve --port PORT` | Listening port; default `8765`. |
| `serve --run-root DIRECTORY` | GUI run storage; default `data/runs/` in this release. |
| `run CONFIG` | Run an experiment JSON file. |
| `run --mode fast` | Request metrics and token timestamps; the default mode. |
| `run --mode log` | The same metrics plus event and interval logs. |
| `run --output FILE` | Write the complete result as JSON. Without this option, print only `summary` to standard output. |
| `run --requests-csv FILE` | Also write per-request timing as CSV. |

The CLI's `--mode` controls logging independently of any GUI `logging_level` saved in the configuration. Config and output paths are relative to the current working directory; parameter-file paths are relative to `simulator/`. CLI runs create the requested files and do not add entries to GUI history.

Start the GUI with a custom port and storage directory:

```sh
python -m simulator serve --port 8766 --run-root saved-runs
```

Use the same run directory to reopen saved history. Press Ctrl+C to stop the server.

Print one example's summary:

```sh
python -m simulator run simulator/examples/case_H_H.json
```

Save a complete result and request CSV:

```sh
python -m simulator run simulator/examples/contention_original.json \
  --mode fast --output results/contention_original.json \
  --requests-csv results/contention_original.csv
```

Save a replay trace, then open it through **Import results**:

```sh
python -m simulator run simulator/examples/contention_optimal.json \
  --mode log --output results/contention_optimal_trace.json
```

Run all fixed paper examples once with Summary logging:

```sh
for config in simulator/examples/*.json; do
  name=$(basename "$config" .json)
  python -m simulator run "$config" --mode fast --output "results/$name.json"
done
```

Successful commands exit with status `0`. Invalid arguments or simulation failures produce a nonzero exit status and an error on standard error.

## Returned values and file formats

Simulation timestamps and durations are in seconds. Timestamps share the experiment's time origin; durations are differences between timestamps. They exclude model loading and warmup. A result row represents a request cohort, not a separate row for every sequence in its batch.

### Complete simulation result

This object is written by `run --output`, returned by `run_simulation`, and stored in a GUI run's `result` field.

| Field | Type | Meaning |
|---|---|---|
| `schema` | integer | Output schema version, currently `2`. |
| `engine` | string | Engine identifier, currently `general_dag`. |
| `mode` | string | `fast` or `log`. |
| `input` | object | Experiment used for the run. CLI results include validated defaults; GUI results retain the saved input. |
| `requests` | array | Per-request records in input order. |
| `summary` | object | Aggregates for all requests, instances, and models. |
| `warnings` | array of strings | Validation messages; empty when there are none. |
| `resolved` | object | Effective configuration, parameter snapshots, routes, and assignments. |
| `duration_s` | number | Latest request completion timestamp; `0` for no requests. |
| `engine_event_count` | integer | Internal engine events processed, including events not exposed in the trace. |
| `wall_time_s` | number | Actual simulator execution time, including configuration and workload preparation. |
| `log` | object | Present only in `log` mode; contains replay intervals and milestone events. |

### Per-request records

| Field | Meaning |
|---|---|
| `request_id`, `instance`, `model` | Request, deployment, and model identifiers. |
| `arrival_s` | Request arrival timestamp. |
| `start_s` | Timestamp when the instance begins serving the request. |
| `first_token_s` | First output-token timestamp. |
| `completion_s` | Last output-token timestamp. |
| `queue_s` | `start_s - arrival_s`. |
| `service_s` | `completion_s - start_s`. |
| `response_s` | `completion_s - arrival_s`; end-to-end latency including queueing. |
| `ttft_s` | `first_token_s - arrival_s`. |
| `tpot_s` | `(completion_s - first_token_s) / (output_tokens - 1)`; `null` for one output token. |
| `token_times_s` | Ordered array of all output-token timestamps. |
| `output_tokens` | Number of output-token timestamps. |

Request CSV exports use this header; a `null` TPOT becomes an empty CSV cell:

```csv
request_id,instance,model,arrival_s,start_s,first_token_s,completion_s,queue_s,service_s,response_s,ttft_s,tpot_s
```

### Summary

`summary.all` aggregates the whole run. `summary.instances[instance_id]` and `summary.models[model_id]` contain the same fields for their respective subsets.

| Field | Meaning |
|---|---|
| `requests` | Number of request records in the subset. |
| `tpot_requests` | Number of records with a non-null TPOT. |
| `mean_service_s`, `mean_response_s`, `mean_queue_s`, `mean_ttft_s` | Arithmetic means of the corresponding per-request fields. |
| `mean_tpot_s` | Mean of non-null per-request TPOT values. |

Means are `null` for an empty contributing set. Each request has equal weight; means are not weighted by batch size or token count.

### Resolved configuration

| Field in `resolved` | Contents |
|---|---|
| `parameter_mapping` | Effective file references, device/link profiles, overrides, and shared-link selection. |
| `artifacts` | Loaded parameter snapshots indexed by precision. |
| `interference_artifact` | Loaded shared-network parameter snapshot. |
| `provenance` | Source entries with a `path` field. GUI runs refer to their saved snapshots. |
| `shared_resources` | Map keyed by link ID. Each entry has `bandwidth` in bytes/s, `endpoints`, `parameter_ref`, and `duplex`. |
| `requests` | Resolved assignments indexed by request ID, described below. |
| `graph` | Resolved network: `nodes`, `links`, `routes`, and metadata (`schema_version`, `id`, `version`, `cost_rule`). |

Each `resolved.requests[request_id]` contains:

- `row`: effective workload fields: `id`, `model`, `quantization`, `mode`, ordered `nodes`, `split`, `controller`, `topology`, `collective_policy`, `prompt_tokens`, `output_tokens`, `batch`, `requests`, and `microbatch_tokens`. Here `requests` is `1` for the individual cohort.
- `layout`: TP uses `q` and `ff` arrays indexed by layer and rank, and an `output` array indexed by rank, giving assigned attention widths, feed-forward widths, and output vocabulary sizes. PP and Single use `stages`, an array of layer-index lists including the output head.
- `paths`: entries with `source`, `target`, ordered node IDs in `path`, and ordered link IDs in `links`.
- `parameters`: the effective numeric parameter dictionary for the request.
- `operation_count`: number of prepared workload operations.
- `execution_adapter`, `mathematical_model`: identifiers for the saved assignment format and model version.

Graph nodes carry `kind` and, for devices, `device_class`. Links carry `id`, `endpoints`, `duplex`, `alpha_s`, and `B_bytes_per_s`; the latter two may be `null` because effective values come from the parameter mapping. Routes map `source->target` to an ordered node path. Parameter snapshots allow inspection and reproduction without requiring interpretation of individual coefficients.

### Detailed log

`log` contains `events`, `intervals`, and `engine_event_count` (the same count as the top-level field).

**Events** are sorted by `time`. Each has `time`, `kind`, `request`, and `instance`. Kinds are `arrival`, `start`, `token`, and `complete`; a `token` event also has a zero-based `token` index.

**Intervals** have `id`, `start`, `end`, `kind`, `request`, `instance`, and `lane`. Their duration is `end - start`. Intervals can overlap across lanes and are not necessarily sorted by start time.

| Interval kind | Meaning |
|---|---|
| `compute` | Computation on a device. |
| `communication` | Transfer/reservation on a shared network resource. |
| `request_queue` | Waiting behind a previous request on the same instance. |
| `queue_wait` | Waiting for shared network resources. |
| `synchronization` | Waiting for computation or collective communication to synchronize. |
| `private_path` | Path startup and transfer outside shared resources. |
| `receiver_wait` | Waiting for receiver readiness. |
| `progress_wait` | Additional delay associated with cross-instance contention. |
| `floor_wait` | Waiting for collective completion. |

Additional fields depend on interval kind:

| Fields | Meaning |
|---|---|
| `device`, `rank`, `stage` | Device and zero-based TP rank or PP stage; a non-applicable rank/stage is `null`. |
| `token`, `layer`, `microbatch` | Zero-based work indices; `layer` may be `null` for work outside a layer. |
| `work`, `reason` | Operation label and explanation. |
| `source`, `target`, `path`, `links` | Communication endpoints and route. |
| `resource`, `link`, `message` | Shared-resource ID, physical-link ID, and message ID. |
| `bytes` | Bytes transferred in this communication slice. |
| `route_reservation` | `true` when the interval records reservation of a route resource. |

A message may appear on several resource lanes. Summing `bytes` across lanes counts route reservations, not unique application payload.

### GUI run records and validation

Full GUI exports wrap the simulation result in a run record:

| Field | Meaning |
|---|---|
| `id` | Run identifier. |
| `state` | `starting`, `running`, `complete`, `failed`, or `cancelled`. A cancellation acknowledgement may temporarily report `cancelling`. |
| `created_unix`, `ended_unix` | Wall-clock Unix timestamps in seconds; `ended_unix` is present after termination. |
| `config_name` | Configuration name, or the run ID when unnamed. |
| `elapsed_s` | Wall-clock time from creation to termination, or to the current status check. |
| `config` | Saved input configuration. |
| `result` | Complete simulation result when available. |
| `error` | Failure description when available. |

History listings omit `config` and `result`; selecting a run loads them. A successful validation returns `{"valid": true, "warnings": []}`. Server request failures return `{"error": "..."}` with an HTTP error status. Failed or cancelled runs may have no result; their state and error fields describe the outcome.

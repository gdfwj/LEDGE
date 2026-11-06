# LEDGE

LEDGE simulates distributed LLM inference from a network topology, model deployments, and a request stream.

## Install and start

Requires Python 3.10+ and NumPy. Run these commands from this directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m simulator serve --port 8765
```

Open <http://127.0.0.1:8765>. The server listens on the local machine. The included parameters are ready to use; model weights and device connections are not required.

GUI runs are stored in `data/runs/`. To choose another port and run directory:

```sh
python -m simulator serve --port 8766 --run-root saved-runs
```

Use the same run directory to reopen saved history. Press Ctrl+C in the terminal to stop the server.

## GUI guide

Use **Load example** or **Import configuration** to supply an experiment JSON file containing the topology, deployments, and requests. **Export configuration** saves your edits.

| Category | What you can change |
|---|---|
| Network topology | Add, connect, remove, and move devices, routers, and switches. The Inspector edits node/link IDs, device types, profiles, and overrides. |
| Model deployment | Select models, TP/PP/Single parallelism, ordered workers, controller, split ratios or PP layer counts, and collective policy. |
| Requests | Edit arrival times, input/output tokens, batch size, and microbatch size; import CSV or JSON request lists. |
| Parameters | Edit the parameter mapping and device/link overrides, or keep the included defaults. |

Click **Validate configuration**, choose **Summary** or **Detailed** logging, and click **Run simulation**. **Cancel run** stops a run; **Details** shows its status.

Results show request latency, TTFT, TPOT, token timestamps, and request milestones. Detailed logging adds a filterable event timeline and topology replay; **History** reopens saved runs and compares their metrics.

Import saved result JSON, or export a full run, metrics JSON, request CSV, or resolved configuration. Full run JSON and CLI result JSON can be reimported for replay.

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

## Paper examples

The examples reproduce configurations from *LEDGE: An Efficient Simulator for Distributed LLM Serving on Heterogeneous Edge Devices*. They use the included defaults and the paper's network and device settings. Running an example evaluates that fixed configuration. Large workloads default to Summary logging in the GUI.

### Shared-link contention (Section 4.5)

Both examples launch three disjoint TP2 groups simultaneously. In model order Qwen3-4B Q8_0, Qwen3-4B FP16, and Qwen3-8B Q8_0, the batches are 1, 32, and 8; every sequence has 32 input and 32 output tokens. The respective TP splits are 50/50, 50/50, and 75/25. O1, O2, and P3 attach to the router; X, P1, and P2 attach through a shared switch uplink. Device labels follow the paper.

| Example | Workers in model order | Reference mean E2E |
|---|---|---:|
| [contention_original.json](simulator/examples/contention_original.json) | (P1, P3), (O2, X), (O1, P2) | 24.004 s |
| [contention_optimal.json](simulator/examples/contention_optimal.json) | (P2, P3), (O1, O2), (X, P1) | 20.622 s |

`contention_original` uses the placement selected from independent group costs, evaluated under concurrent arrivals. `contention_optimal` uses the contention-aware placement. Only the concurrent workloads are included.

### Network topology and scaling (Section 4.6, Figure 5)

These examples use Qwen3-14B FP16, equally partitioned over RPi5 workers, with `device1` as controller. One request has 128 input and 64 output tokens. Access links and switch uplinks are 1 Gbps; each switch serves up to eight workers.

| Example | Topology | Workers | Reference E2E |
|---|---|---:|---:|
| [scaling_direct_8.json](simulator/examples/scaling_direct_8.json) | Router-connected workers | 8 | 30.86 s |
| [scaling_direct_64.json](simulator/examples/scaling_direct_64.json) | Router-connected workers | 64 | 19.36 s |
| [scaling_switch_8.json](simulator/examples/scaling_switch_8.json) | One switch | 8 | 30.86 s |
| [scaling_switch_64.json](simulator/examples/scaling_switch_64.json) | Eight switches | 64 | 61.09 s |

### Cross-scenario case study (Section 5, Figure 6 and Table 1)

All 16 examples use Qwen3-8B Q8_0 and eight workers at four two-device sites in a star topology. Worker service-rate multipliers for d0 through d7 are `1, 1, 1.5, 1.5, 2, 2, 1.5, 1.5`.

| Code | Environment | Device profile | Links: Mbps / startup ms | Input/output tokens | Arrivals | Objective |
|---|---|---|---|---|---|---|
| H | Smart home | RPi5 | 1000 / 0.05 | 64 / 32 | 1 request | Minimum E2E |
| O | Office | Orin | 1000 / 0.05 | 256 / 512 | 16 requests, 62.5 ms apart | Maximum throughput |
| F | Field operation | Orin | 10 / 2 | 128 / 32 | 1 request | Minimum E2E |
| M | Infrastructure monitoring | RPi5 | 10 / 2 | 1024 / 16 | 48 requests, 60 s apart | Maximum deadline goodput |

The four fixed deployments all use d4 as controller:

| Deployment code | Workers and parallelism | Partition |
|---|---|---|
| H | TP4 on d0, d2, d4, d6 | 1 : 1.5 : 2 : 1.5 |
| O | TP2 on d4, d5 | 1 : 1 |
| F | PP2, d5 then d4 | 24 / 13 layers, including the output head |
| M | PP2, d4 then d5 | 18 / 19 layers, including the output head |

Files are named `case_<environment>_<deployment>.json`: the first code selects the environment and request stream, and the second identifies the environment whose optimized deployment is reused. For example, `case_O_F` applies the Field-selected deployment to Office.

| Environment | H deployment | O deployment | F deployment | M deployment |
|---|---|---|---|---|
| Smart home | [case_H_H](simulator/examples/case_H_H.json) | [case_H_O](simulator/examples/case_H_O.json) | [case_H_F](simulator/examples/case_H_F.json) | [case_H_M](simulator/examples/case_H_M.json) |
| Office | [case_O_H](simulator/examples/case_O_H.json) | [case_O_O](simulator/examples/case_O_O.json) | [case_O_F](simulator/examples/case_O_F.json) | [case_O_M](simulator/examples/case_O_M.json) |
| Field operation | [case_F_H](simulator/examples/case_F_H.json) | [case_F_O](simulator/examples/case_F_O.json) | [case_F_F](simulator/examples/case_F_F.json) | [case_F_M](simulator/examples/case_F_M.json) |
| Monitoring | [case_M_H](simulator/examples/case_M_H.json) | [case_M_O](simulator/examples/case_M_O.json) | [case_M_F](simulator/examples/case_M_F.json) | [case_M_M](simulator/examples/case_M_M.json) |

Reference results for the diagonal examples are **8.27 s** E2E (H), **0.0346 requests/s** throughput (O), **6.40 s** E2E (F), and **47/48** requests within the deadline, or **0.610 requests/min** goodput (M). Office throughput is request count divided by the span from first arrival to last completion. Monitoring counts requests with response time at most 1800 s and divides by the fixed 4620 s evaluation window; multiply by 60 for requests/min. These rates can be computed from the exported request results.

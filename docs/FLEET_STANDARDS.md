# Fleet CI standards — runner tier labels

## Problem (measured)

The self-hosted fleet sizes an ephemeral worker from the **job it intends to
serve**. Example (observed repeatedly for `cargo check/test`):

| Job name (heuristic) | Fleet tier | Resources   |
|----------------------|------------|-------------|
| cargo check/test     | large      | 4 cpu / 8 GiB |
| gitleaks / light lint| micro      | 0.25 cpu / 512 MiB |

GitHub Actions routes jobs by **exact label match** on `runs-on`. If a job only
requests:

```yaml
runs-on: [self-hosted, linux, x64, podman]
```

then **any** idle worker carrying those four base labels may take it — including
a `micro` worker spawned for gitleaks. That defeats fleet sizing:

- Fleet sizes a **large** worker for `cargo check/test`.
- Job labels omit `large`, so a **micro** worker can still match.
- `rustc` is OOM-killed at 512 MiB; the same branch passes on adequate memory.

Measured consequence (mycelium-l1 PR #8): job `cargo check/test` landed on
`wsl-cpu-w7` (tier=micro 0.25c/512m) → `error: could not compile` / rustc exit
failure that looks like a code defect but is infrastructure.

`size_for_job()` checks **labels before the name heuristic**. An explicit tier
label both fixes routing **and** pins the container size. That is the correct
fix — not further heuristic tuning alone.

## Tier table (measured)

| Tier   | CPU  | Memory |
|--------|------|--------|
| micro  | 0.25 | 512 MiB |
| small  | 0.5  | 1 GiB  |
| medium | 2    | 4 GiB  |
| large  | 4    | 8 GiB  |
| xlarge | 8    | 16 GiB |
| gpu    | 4    | 8 GiB  |

## Choices in this repository

| Job (name unchanged) | Tier   | Why |
|----------------------|--------|-----|
| `detect stack` | **micro** | File probes only (`Cargo.toml` / `pyproject`). No compile. |
| `cargo check/test` | **large** | Compiles. Fleet measured intent for this job is large (4c/8g). `cargo check --workspace --all-targets` and `cargo test --workspace` need compile RAM; clippy `--all-targets` measured **1225 MiB** (2.4× micro cap) — compile work, not a cheap lint. Prefer **xlarge** only for heavier multi-crate / `--all-features` workspace stress if large OOMs again. |
| `python lint/test` | **medium** | `uv sync` + pytest installs a dep tree. Measured guidance: pytest with deps → medium (2c/4g). |
| `no stack detected` | **micro** | Echo-only advisory path. |
| `gitleaks` | **micro** | Secret scan binary; does not compile Rust. |
| `trivy filesystem (...)` | **micro** | FS vuln/secret/license scan; no compile. |

GitHub-hosted jobs (`ci.yml` on `ubuntu-latest`, temporary `release.yml` host)
do not use fleet tier labels.

## Rules for future jobs

1. **Every job that compiles** (`cargo check`, `cargo test`, `cargo build`,
   `cargo clippy --all-targets`, `cargo doc`) must list an explicit tier:
   `large` by default; `xlarge` for workspace-wide `--all-features` /
   `--all-targets` stress across many crates.
2. **Non-compile** gates (`cargo fmt`, gitleaks, trivy, yaml/shell lint) use
   `micro` or `small`.
3. **Do not rename jobs** — required status contexts key off exact names.
4. **Do not** weaken tests with `continue-on-error` or `|| true` on compile/test
   gates to mask OOM; fix the tier label instead.

## Verification note

YAML and label presence are validated locally. **End-to-end proof** that
GitHub routes `cargo check/test` only to `large` workers requires a real
Actions run on the self-hosted fleet after merge — label routing cannot be
proven offline.

# mycelium-interp

> Reference interpreter — the trusted executable small-step semantics for the Core IR (RFC-0004; ADR-009; M-110).

**Tier:** compiler  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-interp` is the *meaning* of a Mycelium program: a call-by-value, small-step substitution evaluator over closed Core IR `Node`s. The AOT path (M-150/M-151) is differential-tested against it — not the other way round. Errors are always explicit (`EvalError`) and the interpreter is never silent (SC-3/G2). Zero `unsafe` — compiler-enforced.

The evaluator covers the full v0 calculus: let-bindings, primitive ops, swaps, algebraic data (`Construct`/`Match`), first-class functions (`Lam`/`App`), `Fix` (structural recursion with a fuel clock), and mutual recursion (`FixGroup`). Approximate composition is refused where no ε-propagation rule is defined (ADR-010/M-204).

## `wild:` host-op seam (A1 / RFC-0028 §4.3)

L1 elaboration lowers `wild { name(args) }` to `Node::Op { prim: "wild:name", … }` (no new Core-IR node — KC-3). The runtime half lives here:

| Piece | Role |
|---|---|
| `HostOpRegistry` | Bare-name → typed host handler map |
| `HostCapabilities::ffi` | Runtime half of `@std-sys` + `!{ffi}` |
| `HostFloor` / `StdHostFloor` | OS-contact trait boundary (A1b wires `mycelium-std-sys`) |
| `Interpreter::with_host_floor()` | Opt-in: min ops **and** `ffi` grant |

**Default is fail-closed:** `Interpreter::default()` has an empty host registry and `ffi = false`. Unknown `wild:<name>` is a typed `EvalError::UnknownPrim` (`host-op-not-registered` in Display). Registered but ungranted → `EvalError::HostCapabilityDenied`. Pure fragments cannot invoke host ops silently (`is_pure` excludes any `wild:` prim).

**Min built-in set** (prove-the-seam, not a full OS surface), all tagged **`Declared`**:

- `wild:entropy_fill` — fill `n` bytes from host RNG (`Binary{N} → Bytes`)
- `wild:mono_nanos` — monotonic nanos (`() → Binary{64}`)
- `wild:read_capped` — read a UTF-8 path up to `min(max, 1 MiB)`; **refuses** when the source has more bytes than the effective cap (fail-closed; never a silent prefix)

`HostOpRegistry::with_floor(Arc<dyn HostFloor>)` installs handlers that route through the provided floor (dynamic dispatch). `with_min_floor()` is `with_floor(Arc::new(StdHostFloor))`.

**Not “Residual at elab”:** elaboration succeeds; the historical gap was an **empty runtime registry** (host miss). A1 closes that miss for the min set when the host floor is opted in.

**A1b residual:** `StdHostFloor` mirrors `mycelium-std-sys` with pure `std` (no cross-repo dep). Wire a real `mycelium-std-sys` adapter through `HostFloor` / `with_floor` next; L0 does not re-check the L1 `@std-sys` nodule marker (source-level gate).

## Key items

- `Interpreter` — the reference interpreter: a `PrimRegistry` + optional `HostOpRegistry` + `SwapEngine`, iterating `step` to a normal form.
- `Interpreter::step` — one small-step reduction (the `⟶` relation from RFC-0004 §2).
- `Interpreter::eval` / `eval_core` — multi-step evaluation to a `Value` or `CoreValue` (repr + data fragment).
- `Interpreter::with_host_floor` — opt into the A1 min `wild:` host set + `ffi` grant.
- `EvalError` — exhaustive explicit refusal type covering free variables, type errors, overflow, fuel exhaustion, depth limits, effect budgets, host-op misses/capability denials, and swap failures.
- `PrimRegistry` — dispatch table for pure named primitive operations.
- `HostOpRegistry` / `HostCapabilities` / `HostFloor` — `wild:` host-op registry + capability gate (RFC-0028 §4.3).
- `Supervisor` / `CancelToken` — structured concurrency primitives for the runtime layer.
- `Budgets` / `EffectBudget` — named effect-budget ledger (RFC-0014 §4.5/§4.8).

## Guarantee posture

Metadata is threaded honestly: an `Op`/`Swap` result's guarantee is the `meet` of its inputs and the operation's own intrinsic strength (RFC-0001 §4.7). Provenance is `Derived{op, inputs}` over content hashes. Host-op results are **`Declared`** (VR-5 — OS contact has no proven bound). A free variable, unknown primitive, host-op miss, capability denial, or unsupported swap is always an explicit `EvalError`, never a silent default.

## Design references

- RFC-0004, RFC-0007, RFC-0011, RFC-0014, RFC-0028 §4.3, ADR-009, ADR-010, ADR-014, M-110, M-120, M-204, NFR-7, A1

## Role in the workspace

Depends on `mycelium-core` and `mycelium-numerics`. Used by `mycelium-l1`, `mycelium-mlir`, `mycelium-cert`, and the differential test harness. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-interp).

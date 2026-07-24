//! `mycelium-interp` — the **reference interpreter**: the trusted, executable small-step semantics
//! for the Core IR (M-110; RFC-0004 §2; ADR-009; NFR-7). It is the *meaning* of a program — the AOT
//! path (M-150/M-151) is differential-tested against it, never the other way round.
//!
//! # Small-step operational semantics (closes SPEC §10.3)
//!
//! Programs are **closed** Core IR [`Node`]s (RFC-0001 §4.5). The values (normal forms) are the
//! constants, `Const(v)`. Evaluation is **call-by-value** and proceeds by substitution; we write
//! `e ⟶ e'` for one step and `e[x ↦ v]` for capture-avoiding substitution (trivial here: the only
//! substituends are closed `Const` values, so there are no free variables to capture).
//!
//! ```text
//!  (E-Let-Step)   bound ⟶ bound'
//!                 ───────────────────────────────────────────────
//!                 Let{x, bound, body} ⟶ Let{x, bound', body}
//!
//!  (E-Let-Bind)   ───────────────────────────────────────────────         (bound is a value)
//!                 Let{x, Const(v), body} ⟶ body[x ↦ Const(v)]
//!
//!  (E-Op-Arg)     argᵢ ⟶ argᵢ'         (args 0..i are values, i leftmost non-value)
//!                 ───────────────────────────────────────────────
//!                 Op{p, [..,argᵢ,..]} ⟶ Op{p, [..,argᵢ',..]}
//!
//!  (E-Op-Apply)   all args are Const(vⱼ)        δ(p, [vⱼ]) = Const(r)
//!                 ───────────────────────────────────────────────
//!                 Op{p, [Const(vⱼ)]} ⟶ Const(r)
//!
//!  (E-Swap-Arg)   src ⟶ src'
//!                 ───────────────────────────────────────────────
//!                 Swap{src, t, π} ⟶ Swap{src', t, π}
//!
//!  (E-Swap-Apply) ───────────────────────────────────────────────       σ(v, t, π) = Const(r)
//!                 Swap{Const(v), t, π} ⟶ Const(r)
//! ```
//!
//! `δ` is the primitive-operator semantics ([`prims`]); `σ` is the swap semantics ([`swap`]). Both
//! thread metadata **honestly**: an `Op`/`Swap` result's guarantee is the `meet` of its inputs and
//! the operation's own intrinsic strength (RFC-0001 §4.7, via `GuaranteeStrength::propagate`), and
//! its provenance is `Derived{ op, inputs }` over content hashes (RFC-0001 §4.6). A `Var` that is
//! free (an open term) is **stuck** — an explicit [`EvalError::FreeVariable`], not a silent default.
//!
//! # Algebraic data (r3 — RFC-0001 §4.5 / RFC-0011)
//! Two more node families evaluate here, mirroring the L1 evaluator's `try_match` so L1-eval and
//! L0-interp agree (NFR-7):
//!
//! ```text
//!  (E-Con-Arg)    argᵢ ⟶ argᵢ'      (args 0..i are values, i leftmost non-value)
//!                 ─────────────────────────────────────────────────────
//!                 Construct{c, [..,argᵢ,..]} ⟶ Construct{c, [..,argᵢ',..]}
//!
//!  (E-Con-Value)  every arg is a value  ⇒  Construct{c, [v…]} is a NORMAL FORM (a data value)
//!
//!  (E-Match-Scrut) s ⟶ s'
//!                  ─────────────────────────────────────────────
//!                  Match{s, alts, d} ⟶ Match{s', alts, d}
//!
//!  (E-Match-Sel)  s is a value, first-matching alt/default selects body, binders ↦ fields
//!                 ─────────────────────────────────────────────  (scrutinee guarantee Exact)
//!                 Match{s, alts, d} ⟶ body[binders ↦ fields]
//! ```
//!
//! # Functions & recursion (r4 — RFC-0001 r4 / RFC-0007 §4.1)
//! Three more nodes complete L1-in-Core-IR, retiring the elaboration `Residual` entirely. The v0
//! surface is first-order, so an elaborated `Lam` is **closed** (no captured environment) and
//! application is capture-free substitution — the existing `subst` carries it:
//!
//! ```text
//!  (E-Lam)        Lam{x, e} is a NORMAL FORM (a function value)
//!
//!  (E-App-Fun)    f ⟶ f'                  (E-App-Arg)  f value, a ⟶ a'
//!                 ─────────────────────────             ────────────────────────────
//!                 App{f, a} ⟶ App{f', a}               App{f, a} ⟶ App{f, a'}
//!
//!  (E-App-Beta)   ─────────────────────────────────────────────  (a is a value)
//!                 App{Lam{x, e}, a} ⟶ e[x ↦ a]
//!
//!  (E-Fix)        ─────────────────────────────────────────────  (under the fuel clock)
//!                 Fix{f, e} ⟶ e[f ↦ Fix{f, e}]
//! ```
//!
//! `Fix` unfolds by substitution every step, so a non-productive recursion is an explicit
//! [`EvalError::FuelExhausted`], never a hang (RFC-0007 §4.5, CakeML clock); the totality checker
//! gates `matured` (packaging), never meaning. Applying a non-function is an explicit
//! [`EvalError::ApplyNonFunction`]; a program that evaluates to a bare function is
//! [`EvalError::FunctionResult`] (a v0 entry returns a repr/data value, not a function).
//!
//! A `Construct` whose arguments are all values is itself a value (a data value, GHC-Core style); at
//! the `eval` boundary it reads off as a [`mycelium_core::Datum`]. `Match` selects the first
//! matching alternative (constructor arm on `CtorRef` identity; literal arm on `repr+payload`
//! equality), binds its fields left-to-right, and defaults on no match (the checker proves coverage,
//! WF7; a genuine no-match is an explicit [`EvalError::NonExhaustiveMatch`]). **Guarantee meet
//! (RFC-0011 §4.6):** a `Match` result is met with the scrutinee's guarantee — for the *reachable r3
//! fragment* the scrutinee is `Exact`, so the meet is the identity; a **non-`Exact` data scrutinee**
//! is the explicit r3 boundary [`EvalError::GuaranteeMeetUnsupported`] (degrading a precise
//! per-value bound by a composite *summary* would force fabricating a bound — refused, never
//! silent). `Construct` itself takes the meet of its fields' guarantees (in the [`mycelium_core::Datum`]
//! summary).
//!
//! # Host ops (`wild:`, A1 / RFC-0028 §4.3)
//! L1 elaborates `wild { name(args) }` to `Node::Op { prim: "wild:name", … }` (KC-3 — no new node).
//! The runtime half is [`wild`]: a [`HostOpRegistry`] + [`HostCapabilities::ffi`] gate. Default
//! interpreters register **no** host ops and grant **no** `ffi` (fail closed / pure). Opt in with
//! [`Interpreter::with_host_floor`]. Unknown `wild:<name>` is a typed
//! [`EvalError::UnknownPrim`] (`host-op-not-registered`); registered-but-ungranted is
//! [`EvalError::HostCapabilityDenied`]. This is **not** an elaboration Residual — elab succeeds;
//! the historical gap was an empty runtime registry.
//!
//! # What is *not* here (by scope)
//! The certified binary↔ternary **swap** is **M-120** (this crate ships the trivial identity swap,
//! [`swap::IdentitySwapEngine`], plus `mycelium-cert` engines for opt-in use). The A1 min host floor
//! uses pure `std` ([`StdHostFloor`]); wiring through `mycelium-std-sys` is **A1b**. L0 does not
//! re-check the L1 `@std-sys` nodule-header marker (source-level gate).
//!
//! **Trusted-base discipline (ADR-014 / DN-21 §5 F-1):** zero `unsafe` — compiler-enforced.
#![forbid(unsafe_code)]

pub mod budget;
pub mod parallel;
pub mod prims;
pub mod supervise;
pub mod swap;
pub mod wild;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use mycelium_core::{Alt, CoreValue, Datum, GuaranteeStrength, Node, Repr, Value, WfError};
use mycelium_workstack::{ensure_sufficient_stack, BudgetError, DepthGuard, RecursionBudget};

pub use budget::{Budgets, EffectBudget, EffectBudgetExhausted, EffectKind};
pub use parallel::{is_pure, plan_parallel, BatchHead, ParallelPlan};
pub use prims::PrimRegistry;
pub use supervise::{
    CancelToken, Cancelled, Escalation, RestartIntensity, Supervisor, TaskOutcome,
};
pub use swap::{IdentitySwapEngine, SwapEngine};
pub use wild::{
    HostCapabilities, HostFloor, HostOpFn, HostOpRegistry, StdHostFloor, FFI_EFFECT,
    HOST_BYTE_HARD_CAP,
};

/// The result of one small-step attempt on a node.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// The node is already a value (`Const`) — no redex.
    Value,
    /// The node reduced by one step to this successor. Boxed because a [`Node`] embeds a whole
    /// [`Value`] and would otherwise dwarf the `Value` variant.
    Next(Box<Node>),
}

/// Why evaluation could not proceed (always explicit — the interpreter is never silent; SC-3/G2).
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    /// A free variable was encountered (the program is not closed).
    FreeVariable(String),
    /// No primitive is registered under this name.
    ///
    /// For a `wild:<name>` key this is the **typed host-op miss** (RFC-0028 §4.3 / A1): the host
    /// registry has no handler for `name`. Never a silent no-op or panic (G2). See the
    /// [`Display`](EvalError) impl for the host-capability wording.
    UnknownPrim(String),
    /// A registered `wild:` host op was invoked without the runtime `ffi` capability grant (the
    /// interpreter half of `@std-sys` + `!{ffi}`). Fail-closed: pure/deterministic interpreters
    /// keep `ffi = false` and refuse every host invoke (A1; G2).
    HostCapabilityDenied {
        /// The bare host-op name (without the `wild:` prefix).
        op: String,
        /// Why the grant was refused.
        why: String,
    },
    /// A primitive was applied to the wrong arity/paradigm/width.
    PrimType {
        /// The primitive name.
        prim: String,
        /// A human-readable explanation.
        why: String,
    },
    /// A primitive would have to compose an approximate input for which it has **no defined
    /// ε-propagation rule** (the logical `bit.*` ops; `trit.mul` pending the Dense magnitudes; or an
    /// input carrying a non-`Error` bound). The additive arithmetic *does* compose now via the
    /// verified-numerics kernel (M-204; ADR-010); this is refused rather than fabricating a bound.
    ApproxCompositionUnsupported {
        /// The primitive name.
        prim: String,
    },
    /// The swap engine does not support this `(from → to)` conversion (the certified cross-paradigm
    /// swap is M-120).
    UnsupportedSwap {
        /// Source representation.
        from: Repr,
        /// Target representation.
        to: Repr,
    },
    /// A fixed-width arithmetic result fell outside the representable range — explicit, never a
    /// silent wrap (SC-3; balanced-ternary range, `binary-ternary.md` §1).
    Overflow {
        /// The primitive name.
        prim: String,
    },
    /// Evaluation exceeded its step budget (a non-termination guard).
    FuelExhausted,
    /// Evaluation exceeded its **control-stack depth** budget — the space analogue of `FuelExhausted`
    /// (M-347): the AOT env-machine (a trampoline over an explicit heap control stack) refuses past a
    /// depth ceiling with this **explicit, graceful** error rather than growing memory unboundedly /
    /// aborting. Never silent. The reference interpreter is O(1)-stack and does not raise this.
    DepthLimit {
        /// The control-stack depth ceiling that was hit.
        limit: usize,
    },
    /// A declared **effect budget** was exceeded — the effect analogue of `FuelExhausted` (time) and
    /// `DepthLimit` (space) (RFC-0014 §4.5 I4). A bounded effect (retry/cascade/alloc/time) overruns
    /// its *named* budget **gracefully**: the runtime refuses with this explicit error rather than
    /// hanging or OOM-ing, exactly as a runaway recursion does. **One enforcement mechanism over
    /// separate named budgets** (RFC-0014 §8): the recovery [`Budgets`] ledger and the env-machine
    /// share this channel (RFC-0014 §4.8). Lives at the runtime/checker layer; introduces no L0 node
    /// (KC-3). Never silent (G2).
    EffectBudget(EffectBudgetExhausted),
    /// A swap engine reported a failure (e.g. an illegal pair or an out-of-range conversion). The
    /// message comes from the engine; it is always explicit, never a silent coercion.
    Swap(String),
    /// A constructed result violated a Core IR well-formedness invariant (RFC-0001 §4.3/§4.5).
    Wf(WfError),
    /// A `Match` reduced with no alternative matching and no `default` (RFC-0011 §4.3 WF7). The
    /// checker proves coverage above the kernel, so this is unreachable for checked programs — kept
    /// as the explicit never-silent fallback (G2), never a panic or a silent default.
    NonExhaustiveMatch,
    /// A `Construct`/`Match` node was malformed against the data fragment (an arity mismatch the
    /// checker should have caught, a non-saturated constructor — WF6/WF7). Explicit, never a guess.
    DataMalformed {
        /// What was malformed, and why.
        why: String,
    },
    /// The r3 boundary (RFC-0011 §4.6): a `Match` on a **non-`Exact` data scrutinee` would have to
    /// fold the scrutinee's composite *summary* guarantee into the result. Realising that without
    /// fabricating a bound is deferred (the reachable r3 fragment is `Exact`). Refused explicitly,
    /// never silently dropped.
    GuaranteeMeetUnsupported {
        /// The scrutinee's (non-`Exact`) summary guarantee.
        scrutinee: GuaranteeStrength,
    },
    /// [`Interpreter::eval`] was asked for a representation [`Value`] but the program evaluated to a
    /// **data value** ([`mycelium_core::Datum`]). Use [`Interpreter::eval_core`] for the data
    /// fragment. Explicit, so a repr-only caller never silently mishandles a datum.
    DataResult,
    /// An `App` whose function position reduced to a **non-function** value (a `Const`/`Construct`,
    /// not a `Lam`) — RFC-0001 r4. The checker proves applications are well-typed, so this is
    /// unreachable for checked programs; kept as the explicit never-silent fallback (G2).
    ApplyNonFunction,
    /// The program evaluated to a **function value** (a bare `Lam`) — RFC-0001 r4. A v0 entry returns
    /// a representation or data value, never a function (the first-order surface has no function-typed
    /// results), so this is an explicit refusal rather than a silent or partial observable.
    FunctionResult,
}

impl From<EffectBudgetExhausted> for EvalError {
    /// Route a bounded-effect overrun onto the unified runtime refusal channel (RFC-0014 §4.8): the
    /// env-machine and the recovery driver share this conversion so an effect overrun surfaces exactly
    /// as `FuelExhausted`/`DepthLimit` do — explicit and graceful, never a hang.
    fn from(e: EffectBudgetExhausted) -> Self {
        EvalError::EffectBudget(e)
    }
}

impl From<BudgetError> for EvalError {
    /// Reconcile the shared `mycelium-workstack` over-budget surface to the interpreter's existing
    /// never-silent `DepthLimit` (RFC-0041 W4/§5.1). The canonical [`BudgetError::DepthExceeded`]
    /// (a `u32` ceiling on the §4.0 depth metric — W1) becomes [`EvalError::DepthLimit`] (`usize`
    /// ceiling) at the **same threshold**, so a deep-but-fuel-cheap value refuses cleanly here exactly
    /// as the AOT env-machine's `depth_limit_error` does (W3½). The interpreter charges **only** the
    /// depth budget (`try_enter`), so `DepthExceeded` is the only variant it can produce; the
    /// `OutOfBudget` arm is unreachable in this crate and is mapped defensively onto the same
    /// `DepthLimit` channel (its ceiling reported) rather than panicking — never-silent (G2).
    fn from(e: BudgetError) -> Self {
        let limit = match e {
            BudgetError::DepthExceeded { limit } => limit,
            BudgetError::OutOfBudget { limit, .. } => u32::try_from(limit).unwrap_or(u32::MAX),
        };
        EvalError::DepthLimit {
            limit: limit as usize,
        }
    }
}

/// Charge one structural-recursion frame against the shared [`RecursionBudget`] (RFC-0041 W4),
/// mapping a never-silent over-budget to [`EvalError::DepthLimit`]. The returned [`DepthGuard`]
/// releases the frame on `Drop`, so the budget's *live* depth tracks the current structural nesting
/// — sibling descents (e.g. an `Op`/`Construct`'s argument list) do **not** accumulate, only nesting
/// does. This is the guard the substitution machine wraps each recursive descent in so a crafted
/// deep value refuses with `DepthLimit` at the ceiling instead of a host-stack `SIGABRT` (RR-29 §0.1).
fn charge_depth(budget: &RecursionBudget) -> Result<DepthGuard<'_>, EvalError> {
    budget.try_enter().map_err(EvalError::from)
}

impl core::fmt::Display for EvalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EvalError::FreeVariable(x) => write!(f, "free variable: {x}"),
            // A `wild:`-namespaced key is a host/FFI operation (RFC-0028 §4.3): the registry is the
            // capability handle, and the default registry grants none — so an unresolved `wild:` key
            // is an *ungranted host capability*, not a typo. Report it as such (never silent — G2).
            EvalError::UnknownPrim(p) => match p.strip_prefix("wild:") {
                // Continued lines begin with an explicit `\u{20}` space, not a trailing space before
                // the `\`: the repo's `trailing-whitespace` hook would strip the latter and silently
                // fuse the words (`§4.3)dispatches`) — never-silent (G2). (Copilot #508.)
                Some(op) => write!(
                    f,
                    "host-op-not-registered: `wild:{op}` — the host-op registry (RFC-0028 §4.3 / A1)\
\u{20}has no handler for `{op}`; register it (or use Interpreter::with_host_floor for the min\
\u{20}built-in set) and grant `ffi` (never silent — G2)"
                ),
                None => write!(f, "unknown primitive: {p}"),
            },
            EvalError::HostCapabilityDenied { op, why } => {
                write!(f, "host capability `{op}` denied: {why}")
            }
            EvalError::PrimType { prim, why } => write!(f, "type error in {prim}: {why}"),
            EvalError::ApproxCompositionUnsupported { prim } => write!(
                f,
                "{prim}: no defined ε-propagation rule for an approximate input (ADR-010/M-204)"
            ),
            EvalError::UnsupportedSwap { from, to } => {
                write!(
                    f,
                    "unsupported swap: {from:?} → {to:?} (certified swap is M-120)"
                )
            }
            EvalError::Overflow { prim } => {
                write!(
                    f,
                    "{prim}: fixed-width arithmetic overflow (result out of range)"
                )
            }
            EvalError::FuelExhausted => write!(f, "evaluation exceeded its step budget"),
            EvalError::DepthLimit { limit } => {
                write!(
                    f,
                    "evaluation exceeded its control-stack depth budget ({limit})"
                )
            }
            EvalError::EffectBudget(e) => write!(f, "{e}"),
            EvalError::Swap(msg) => write!(f, "swap failed: {msg}"),
            EvalError::Wf(e) => write!(f, "well-formedness violation: {e}"),
            EvalError::NonExhaustiveMatch => write!(
                f,
                "match had no matching alternative and no default (WF7 — the checker requires \
                 coverage)"
            ),
            EvalError::DataMalformed { why } => write!(f, "malformed data node: {why}"),
            EvalError::GuaranteeMeetUnsupported { scrutinee } => write!(
                f,
                "match on a non-Exact data scrutinee ({scrutinee:?}): the guarantee-meet through \
                 Match is deferred in r3 (RFC-0011 §4.6) — the reachable fragment is Exact"
            ),
            EvalError::DataResult => write!(
                f,
                "the program evaluated to a data value; use eval_core for the data fragment"
            ),
            EvalError::ApplyNonFunction => write!(
                f,
                "applied a non-function value (the checker should have refused this application)"
            ),
            EvalError::FunctionResult => write!(
                f,
                "the program evaluated to a function value (a v0 entry returns a repr/data value)"
            ),
        }
    }
}

impl std::error::Error for EvalError {}

/// Default step budget — generous for the non-recursive core language (it always terminates), a
/// guard against pathological inputs.
const DEFAULT_FUEL: u64 = 1_000_000;

/// The reference interpreter: a primitive registry + a swap engine. [`Interpreter::default`] wires
/// the exact built-in prims and the identity swap engine.
///
/// **`Clone` (M-864):** the parallel-pure-fragment evaluator (`crate::parallel`) needs an *owned*
/// handle to the interpreter's config it can move into a `'static` job closure submitted to the
/// M-861/M-864 persistent [`Scheduler`](mycelium_sched::scheduler::Scheduler) pool — a borrow no
/// longer suffices once the pool's worker threads outlive any single `run_indexed` call (see
/// `crate::parallel::eval_top_batch`). Cloning is cheap: `swap` is stored as `Arc<dyn SwapEngine>`
/// (an `Arc::clone` bump, not a deep copy — `SwapEngine` itself has no `Clone` bound, since a boxed
/// trait object can't derive one; `Arc` sidesteps that without changing the trait), `prims` /
/// `host_ops` are small `BTreeMap` clones (bounded by the built-in prim/host-op count, not by
/// program size), and `fuel` / `host_caps` / `depth_limit` are `Copy`.
#[derive(Clone)]
pub struct Interpreter {
    prims: PrimRegistry,
    /// Host-op registry for the reserved `wild:` prim namespace (RFC-0028 §4.3; A1). Empty by
    /// default — pure/deterministic fragments see a typed miss on every `wild:<name>`.
    host_ops: HostOpRegistry,
    /// Runtime half of `@std-sys` + `!{ffi}`: must grant `ffi` before a registered host op runs.
    host_caps: HostCapabilities,
    swap: Arc<dyn SwapEngine>,
    fuel: u64,
    /// The shared [`RecursionBudget`] depth ceiling on the §4.0 metric (RFC-0041 W4/W7). Defaults to
    /// the global floor [`RecursionBudget::DEFAULT_DEPTH_LIMIT`] (4096) — [`Default`]/[`new`](Self::new)
    /// preserve the established behavior exactly. Tunable per-invocation via
    /// [`with_depth`](Self::with_depth): the additive constructor that lets a caller verify the
    /// budget→[`EvalError::DepthLimit`] mapping at an *arbitrary* ceiling (not only the floor), and that
    /// backs the CLI `--unbounded` escape hatch by setting it to [`u32::MAX`] (RFC-0041 §5 / DN-84 §9.3).
    depth_limit: u32,
}

impl Default for Interpreter {
    fn default() -> Self {
        Interpreter {
            prims: PrimRegistry::with_builtins(),
            // A1: default grants **no** host ops and **no** `ffi` — pure fragments stay pure.
            host_ops: HostOpRegistry::empty(),
            host_caps: HostCapabilities::default(),
            swap: Arc::new(IdentitySwapEngine),
            fuel: DEFAULT_FUEL,
            depth_limit: RecursionBudget::DEFAULT_DEPTH_LIMIT,
        }
    }
}

impl Interpreter {
    /// Build an interpreter with a custom prim registry and swap engine (e.g. M-120's certified
    /// swap, or M-111's arithmetic prims). Takes an owned `Box` (the existing, stable public
    /// signature); internally stored as `Arc<dyn SwapEngine>` (M-864 — see the struct docs), an
    /// unconditional, allocation-free `Box` → `Arc` conversion (`Arc::from`).
    ///
    /// Host ops stay empty and `ffi` ungranted — use [`with_host_floor`](Self::with_host_floor) or
    /// [`with_host_ops`](Self::with_host_ops) to opt into the `wild:` seam.
    #[must_use]
    pub fn new(prims: PrimRegistry, swap: Box<dyn SwapEngine>) -> Self {
        Interpreter {
            prims,
            host_ops: HostOpRegistry::empty(),
            host_caps: HostCapabilities::default(),
            swap: Arc::from(swap),
            fuel: DEFAULT_FUEL,
            depth_limit: RecursionBudget::DEFAULT_DEPTH_LIMIT,
        }
    }

    /// Install the A1 min host-op floor (`wild:entropy_fill`, `wild:mono_nanos`,
    /// `wild:read_capped`) **and** grant the runtime `ffi` capability. The only opt-in that makes
    /// host-shaped programs evaluate; the default interpreter remains fail-closed (G2).
    #[must_use]
    pub fn with_host_floor(mut self) -> Self {
        self.host_ops = HostOpRegistry::with_min_floor();
        self.host_caps = HostCapabilities::default().with_ffi();
        self
    }

    /// Install a custom host-op registry and capability grants (A1 extensibility / A1b wiring).
    #[must_use]
    pub fn with_host_ops(mut self, host_ops: HostOpRegistry, host_caps: HostCapabilities) -> Self {
        self.host_ops = host_ops;
        self.host_caps = host_caps;
        self
    }

    /// Override the step budget.
    #[must_use]
    pub fn with_fuel(mut self, fuel: u64) -> Self {
        self.fuel = fuel;
        self
    }

    /// Override the shared [`RecursionBudget`] **depth ceiling** on the §4.0 metric (RFC-0041 W7 —
    /// additive; the error enum and every observable are unchanged). The default is the global floor
    /// [`RecursionBudget::DEFAULT_DEPTH_LIMIT`] (4096); this lets a caller run the substitution machine
    /// under an *arbitrary* ceiling.
    ///
    /// Two uses (both never-silent, G2): (1) a **uniform-mapping check** — a small `depth_limit` (e.g.
    /// 8) proves a controlled-depth value refuses with [`EvalError::DepthLimit`] at *exactly* that
    /// ceiling, confirming the budget→error mapping is uniform across the range, not merely
    /// floor-verified; (2) the CLI **`--unbounded`** escape hatch (RFC-0041 §5 / DN-84 §9.3) passes
    /// [`u32::MAX`] to lift the deterministic ceiling for opt-in, non-deterministic REPL/exploration —
    /// a machine-dependent mode excluded from the conformance corpus. Even at [`u32::MAX`] the refusal
    /// stays never-silent: the growable deep stack ([`eval`](Self::eval)) makes memory, not a host-stack
    /// `SIGABRT`, the binding limit.
    #[must_use]
    pub fn with_depth(mut self, depth_limit: u32) -> Self {
        self.depth_limit = depth_limit;
        self
    }

    /// A fresh shared [`RecursionBudget`] sized to this interpreter's [`depth_limit`](Self::with_depth),
    /// with the memory/work-step ceilings left effectively unbounded (as [`RecursionBudget::default`]
    /// does — the real memory ceiling is a startup/process-arena concern, §4.2, not this per-eval knob).
    /// A default interpreter yields a budget identical to [`RecursionBudget::default`], so the
    /// established behavior is preserved exactly.
    fn budget(&self) -> RecursionBudget {
        RecursionBudget::new(self.depth_limit, u64::MAX, u64::MAX)
    }

    /// The registered primitive names (for tooling/EXPLAIN).
    #[must_use]
    pub fn prim_names(&self) -> Vec<&str> {
        self.prims.names()
    }

    /// Perform exactly one small-step reduction on `node` (the `⟶` relation above).
    ///
    /// Returns [`Step::Value`] if `node` is already a `Const`, or [`Step::Next`] with the reduced
    /// term. Errors are explicit (free variable, unknown/ill-typed prim, unsupported swap).
    ///
    /// **RFC-0041 W4:** the single-step entry drives the substitution recursion on a fresh default
    /// [`RecursionBudget`], so even a direct `step` on a deep node refuses with [`EvalError::DepthLimit`]
    /// at the ceiling rather than a host-stack abort. Callers evaluating adversarial input should use
    /// [`eval`](Self::eval)/[`eval_core`](Self::eval_core), which additionally run on the growable deep
    /// stack (§4.3); a bare `step` charges the budget but is not itself stack-wrapped.
    pub fn step(&self, node: &Node) -> Result<Step, EvalError> {
        self.step_budgeted(node, &self.budget())
    }

    /// The budgeted small-step relation (RFC-0041 §4.1): identical to [`step`](Self::step) but charging
    /// the shared [`RecursionBudget`] at each structural descent. L0 keeps its substitution shape — only
    /// the budget + guard are threaded in. One frame is charged on entry (via [`charge_depth`]) and
    /// released when this call returns, so the budget's *live* depth equals the current structural
    /// nesting; when a descent would exceed the ceiling the enter refuses with `DepthLimit`.
    fn step_budgeted(&self, node: &Node, budget: &RecursionBudget) -> Result<Step, EvalError> {
        let _frame = charge_depth(budget)?;
        match node {
            Node::Const(_) => Ok(Step::Value),

            Node::Var(x) => Err(EvalError::FreeVariable(x.clone())),

            Node::Let { id, bound, body } => match self.step_budgeted(bound, budget)? {
                // (E-Let-Bind): bound is a value → substitute it into the body.
                Step::Value => Ok(Step::Next(Box::new(subst(body, id, bound, budget)?))),
                // (E-Let-Step): reduce the bound expression first (call-by-value).
                Step::Next(bound2) => Ok(Step::Next(Box::new(Node::Let {
                    id: id.clone(),
                    bound: bound2,
                    body: body.clone(),
                }))),
            },

            Node::Op { prim, args } => {
                // (E-Op-Arg): reduce the leftmost non-value argument, if any.
                for (i, arg) in args.iter().enumerate() {
                    if let Step::Next(arg2) = self.step_budgeted(arg, budget)? {
                        let mut next = args.clone();
                        next[i] = *arg2;
                        return Ok(Step::Next(Box::new(Node::Op {
                            prim: prim.clone(),
                            args: next,
                        })));
                    }
                }
                // (E-Op-Apply): all arguments are values → apply δ.
                // A1 / RFC-0028 §4.3: `wild:<name>` is the reserved host-capability namespace.
                // It never routes through the pure `PrimRegistry` (even if a caller registered a
                // colliding name there) — only through the host-op registry + `ffi` gate.
                let values = collect_values(args)?;
                let result = if prim.starts_with("wild:") {
                    crate::wild::dispatch_wild(&self.host_ops, self.host_caps, prim, &values)?
                } else {
                    let f = self
                        .prims
                        .get(prim)
                        .ok_or_else(|| EvalError::UnknownPrim(prim.clone()))?;
                    f(prim, &values)?
                };
                Ok(Step::Next(Box::new(Node::Const(result))))
            }

            Node::Swap {
                src,
                target,
                policy,
            } => match self.step_budgeted(src, budget)? {
                // (E-Swap-Apply): source is a value → apply σ.
                Step::Value => {
                    let v = as_const(src)?;
                    let result = self.swap.swap(v, target, policy)?;
                    Ok(Step::Next(Box::new(Node::Const(result))))
                }
                // (E-Swap-Arg): reduce the source first.
                Step::Next(src2) => Ok(Step::Next(Box::new(Node::Swap {
                    src: src2,
                    target: target.clone(),
                    policy: policy.clone(),
                }))),
            },

            Node::Construct { ctor, args } => {
                // (E-Con-Arg): reduce the leftmost non-value argument, if any.
                for (i, arg) in args.iter().enumerate() {
                    if let Step::Next(arg2) = self.step_budgeted(arg, budget)? {
                        let mut next = args.clone();
                        next[i] = *arg2;
                        return Ok(Step::Next(Box::new(Node::Construct {
                            ctor: ctor.clone(),
                            args: next,
                        })));
                    }
                }
                // (E-Con-Value): all arguments are values → this Construct is a normal form.
                Ok(Step::Value)
            }

            Node::Match {
                scrutinee,
                alts,
                default,
            } => match self.step_budgeted(scrutinee, budget)? {
                // (E-Match-Scrut): reduce the scrutinee to a value first.
                Step::Next(s2) => Ok(Step::Next(Box::new(Node::Match {
                    scrutinee: s2,
                    alts: alts.clone(),
                    default: default.clone(),
                }))),
                // (E-Match-Sel): the scrutinee is a value → select the arm and meet the guarantee.
                Step::Value => {
                    // The Match result is met with the scrutinee's guarantee (RFC-0011 §4.6). For the
                    // reachable r3 fragment the scrutinee is Exact, so the meet is the identity; a
                    // non-Exact data scrutinee is the explicit r3 boundary (never a fabricated bound).
                    let g = guarantee_of_value(scrutinee, budget)?;
                    if g != GuaranteeStrength::Exact {
                        return Err(EvalError::GuaranteeMeetUnsupported { scrutinee: g });
                    }
                    let body = select_arm(scrutinee, alts, default.as_deref(), budget)?;
                    Ok(Step::Next(Box::new(body)))
                }
            },

            // (E-Lam): a lambda abstraction is a normal form (a function value).
            Node::Lam { .. } => Ok(Step::Value),

            Node::App { func, arg } => match self.step_budgeted(func, budget)? {
                // (E-App-Fun): reduce the function position to a value first.
                Step::Next(f2) => Ok(Step::Next(Box::new(Node::App {
                    func: f2,
                    arg: arg.clone(),
                }))),
                Step::Value => match self.step_budgeted(arg, budget)? {
                    // (E-App-Arg): then reduce the argument (call-by-value).
                    Step::Next(a2) => Ok(Step::Next(Box::new(Node::App {
                        func: func.clone(),
                        arg: a2,
                    }))),
                    // (E-App-Beta): both are values → β-reduce. The function must be a Lam (the
                    // checker proves this); applying any other value is an explicit refusal.
                    Step::Value => match func.as_ref() {
                        Node::Lam { param, body } => {
                            Ok(Step::Next(Box::new(subst(body, param, arg, budget)?)))
                        }
                        _ => Err(EvalError::ApplyNonFunction),
                    },
                },
            },

            // (E-Fix): unfold by substitution under the fuel clock — Fix(f, e) ⟶ e[f ↦ Fix(f, e)].
            // Always a redex; a non-productive recursion exhausts fuel explicitly, never hangs
            // (RFC-0007 §4.5, CakeML clock).
            Node::Fix { name, body } => {
                let unfolded = subst(body, name, node, budget)?;
                Ok(Step::Next(Box::new(unfolded)))
            }

            // (E-FixGroup): mutual recursion (RFC-0001 r5; R7-Q3). A *focus* `FixGroup(defs, fᵢ)`
            // (body is a bare member name) unfolds to that member's definition; any other body is the
            // *continuation*. Either way every member name is substituted by its own focus thunk, so
            // siblings remain mutually recursive across the unfold (subst shadows the group's names,
            // so the freshly-placed thunks are not re-substituted). Always a redex like `Fix`, under
            // the same fuel clock — a non-productive group exhausts fuel explicitly, never hangs.
            Node::FixGroup { defs, body } => {
                let target: Node = match body.as_ref() {
                    Node::Var(v) => defs
                        .iter()
                        .find(|(name, _)| name == v)
                        .map_or_else(|| (**body).clone(), |(_, d)| (**d).clone()),
                    _ => (**body).clone(),
                };
                let unfolded = defs.iter().try_fold(target, |acc, (name, _)| {
                    let focus = Node::FixGroup {
                        defs: defs.clone(),
                        body: Box::new(Node::Var(name.clone())),
                    };
                    subst(&acc, name, &focus, budget)
                })?;
                Ok(Step::Next(Box::new(unfolded)))
            }
        }
    }

    /// Evaluate `node` to a **representation** value by iterating [`step`](Self::step) to a normal
    /// form. Returns the resulting [`Value`], or an [`EvalError`] (including
    /// [`EvalError::FuelExhausted`] if the budget is exceeded, or [`EvalError::DataResult`] if the
    /// program evaluates to a data value — use [`eval_core`](Self::eval_core) for the data fragment).
    pub fn eval(&self, node: &Node) -> Result<Value, EvalError> {
        match self.eval_core(node)? {
            CoreValue::Repr(v) => Ok(v),
            CoreValue::Data(_) => Err(EvalError::DataResult),
        }
    }

    /// Evaluate `node` to a [`CoreValue`] — a representation value **or** a data value (the r3 data
    /// fragment, RFC-0011). Iterates [`step`](Self::step) to a normal form and reads it off: a
    /// `Const` is a representation value; a saturated `Construct` of values is a [`Datum`] (its
    /// meet-summary guarantee computed from its fields). This is the path the M-210 differential
    /// runs for matching/data, against the L1 evaluator (NFR-7).
    pub fn eval_core(&self, node: &Node) -> Result<CoreValue, EvalError> {
        // RFC-0041 W4 (closes the RR-29 §0.1 flagship bug). Run the substitution machine on the
        // **growable deep worker stack** (§4.3) and bound its structural recursion with the shared
        // [`RecursionBudget`] (§4.1), so a crafted deep-but-fuel-cheap value refuses with
        // [`EvalError::DepthLimit`] — never a host-stack `SIGABRT`. The budget is created *inside* the
        // closure: it is `Send` but not `Sync` (interior `Cell` charge state), so it is owned on the
        // worker rather than borrowed across the thread boundary — the AOT `run_core` precedent. The
        // outer `sizing` budget is only read on the caller thread to size the guard.
        let sizing = self.budget();
        ensure_sufficient_stack(&sizing, move || {
            let budget = self.budget();
            self.eval_core_budgeted(node, &budget)
        })
    }

    /// The budgeted normal-form driver (RFC-0041 W4): the O(1) fuel loop of
    /// [`eval_core`](Self::eval_core), threading the shared [`RecursionBudget`] into every structural
    /// recursion (`step` and the value read-off) so deep input refuses with [`EvalError::DepthLimit`]
    /// at the budget's ceiling. Separated from `eval_core` so the fuel loop itself stays O(1) host
    /// stack while the (already `Send`-owned) budget is threaded down.
    fn eval_core_budgeted(
        &self,
        node: &Node,
        budget: &RecursionBudget,
    ) -> Result<CoreValue, EvalError> {
        let mut current = node.clone();
        let mut fuel = self.fuel;
        loop {
            match self.step_budgeted(&current, budget)? {
                Step::Value => return node_to_core_value(&current, budget),
                Step::Next(next) => {
                    fuel = fuel.checked_sub(1).ok_or(EvalError::FuelExhausted)?;
                    current = *next;
                }
            }
        }
    }
}

/// Read a normal-form node off as a [`CoreValue`]: a `Const` is a representation value; a saturated
/// `Construct` of values is a [`Datum`] (the meet-summary computed by [`Datum::new`]). Any other
/// node is not a normal form — an explicit error, never a silent default.
fn node_to_core_value(node: &Node, budget: &RecursionBudget) -> Result<CoreValue, EvalError> {
    // RFC-0041 W4: the read-off recurses over a (saturated `Construct`) data value's spine, so it is
    // charged on the same shared budget as `step` — a deep VALUE read off after evaluation refuses with
    // `DepthLimit` at the ceiling rather than a host-stack abort (the value-walk half of RR-29 §0.1).
    let _frame = charge_depth(budget)?;
    match node {
        Node::Const(v) => Ok(CoreValue::Repr(v.clone())),
        Node::Construct { ctor, args } => {
            let fields = args
                .iter()
                .map(|a| node_to_core_value(a, budget))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(CoreValue::Data(Datum::new(ctor.clone(), fields)))
        }
        Node::Var(x) => Err(EvalError::FreeVariable(x.clone())),
        // A bare Lam normal form is a function value — not an observable v0 result (RFC-0001 r4).
        Node::Lam { .. } => Err(EvalError::FunctionResult),
        _ => Err(EvalError::DataMalformed {
            why: "evaluation ended on a non-normal-form node".to_owned(),
        }),
    }
}

/// The guarantee of a **value** node (a `Const`, a saturated `Construct`, or a `Lam`): the
/// representation value's own `Meta.guarantee`, the `meet`-summary of a data value's fields
/// (RFC-0011 §4.6), or — for a bare function value — `Exact` (a closed lambda term carries no
/// representational approximation to summarize; it is the identity element for the field-wise
/// meet, same as an empty `Construct`).
///
/// **`Node::Lam` (ADR-033/DN-74, M-923).** Before this fix a `Construct` field holding a `Lam`
/// (the normal form of a `FieldSpec::Fn`-typed field — ADR-033 §3.1: "a `Lam` node *is* a function
/// value") fell through to the `DataMalformed` catch-all below, because no program ever produced
/// such a field until `mycelium-l1`'s `field_spec`/`elaborate_direct` (M-923) made `FieldSpec::Fn`
/// reachable. This is a pre-existing latent gap, not a new relaxation: a `Lam` was already a
/// legitimate `Step::Value` normal form (`Node::Lam { .. } => Ok(Step::Value)` above); this only
/// teaches the **guarantee**-meet computation the same fact `Step`-reduction already knew.
fn guarantee_of_value(
    node: &Node,
    budget: &RecursionBudget,
) -> Result<GuaranteeStrength, EvalError> {
    // RFC-0041 W4: recurses over a data value's field spine; charged on the shared budget so a deep
    // Match scrutinee refuses with `DepthLimit` rather than a host-stack abort.
    let _frame = charge_depth(budget)?;
    match node {
        Node::Const(v) => Ok(v.meta().guarantee()),
        Node::Construct { args, .. } => {
            let mut g = GuaranteeStrength::Exact;
            for a in args {
                g = g.meet(guarantee_of_value(a, budget)?);
            }
            Ok(g)
        }
        Node::Lam { .. } => Ok(GuaranteeStrength::Exact),
        Node::Var(x) => Err(EvalError::FreeVariable(x.clone())),
        _ => Err(EvalError::DataMalformed {
            why: "match scrutinee did not reduce to a value".to_owned(),
        }),
    }
}

/// Select the first-matching `Match` alternative (or the default) and return its binder-substituted
/// body (RFC-0011 §4.6; mirrors the L1 evaluator's `try_match`). A constructor arm matches a
/// `Construct` of the same [`CtorRef`](mycelium_core::CtorRef), binding its fields left-to-right; a
/// literal arm matches a `Const` equal on `repr+payload`. No match + no default is an explicit
/// [`EvalError::NonExhaustiveMatch`].
fn select_arm(
    scrutinee: &Node,
    alts: &[Alt],
    default: Option<&Node>,
    budget: &RecursionBudget,
) -> Result<Node, EvalError> {
    for alt in alts {
        match alt {
            Alt::Ctor {
                ctor,
                binders,
                body,
            } => {
                if let Node::Construct { ctor: c2, args } = scrutinee {
                    if c2 == ctor {
                        if binders.len() != args.len() {
                            return Err(EvalError::DataMalformed {
                                why: format!(
                                    "constructor arm binds {} of {} field(s) (WF6/WF7)",
                                    binders.len(),
                                    args.len()
                                ),
                            });
                        }
                        // Bind fields left-to-right; the args are closed values, so substitution is
                        // capture-free (the same property the Const substitution relies on).
                        let mut b = body.clone();
                        for (binder, arg) in binders.iter().zip(args) {
                            b = subst(&b, binder, arg, budget)?;
                        }
                        return Ok(b);
                    }
                }
            }
            Alt::Lit { value, body } => {
                if let Node::Const(v) = scrutinee {
                    if v.repr() == value.repr() && v.payload() == value.payload() {
                        return Ok(body.clone());
                    }
                }
            }
        }
    }
    match default {
        Some(d) => Ok(d.clone()),
        None => Err(EvalError::NonExhaustiveMatch),
    }
}

/// Extract the `Value` from a `Const` node (an internal invariant when [`Step::Value`] was reported).
fn as_const(node: &Node) -> Result<&Value, EvalError> {
    match node {
        Node::Const(v) => Ok(v),
        Node::Var(x) => Err(EvalError::FreeVariable(x.clone())),
        // Unreachable when called after a `Step::Value`; treated as "stuck" defensively.
        _ => Err(EvalError::FreeVariable(
            "<non-value normal form>".to_owned(),
        )),
    }
}

/// Collect a list of nodes known to be values into their `Value`s.
fn collect_values(args: &[Node]) -> Result<Vec<&Value>, EvalError> {
    args.iter().map(as_const).collect()
}

/// Capture-avoiding substitution `node[var ↦ value]`. Substituends are closed `Const` values, so no
/// renaming is ever needed; substitution stops under a binder that shadows `var`.
///
/// **RFC-0041 W4:** this walks the *whole* term in one recursive descent (a single `step`'s
/// `Let`/β/`Fix` unfold), so a deep term would otherwise overflow the host stack. It charges the
/// shared [`RecursionBudget`] at every descent (via [`charge_depth`]) and is therefore fallible — a
/// term deeper than the ceiling refuses with [`EvalError::DepthLimit`] rather than a `SIGABRT`. L0
/// keeps its substitution shape (§4.1); only the budget + guard are threaded in.
fn subst(
    node: &Node,
    var: &str,
    value: &Node,
    budget: &RecursionBudget,
) -> Result<Node, EvalError> {
    let _frame = charge_depth(budget)?;
    Ok(match node {
        Node::Const(_) => node.clone(),
        Node::Var(x) => {
            if x == var {
                value.clone()
            } else {
                node.clone()
            }
        }
        Node::Let { id, bound, body } => Node::Let {
            id: id.clone(),
            bound: Box::new(subst(bound, var, value, budget)?),
            // Shadowing: a re-binding of `var` blocks substitution in the body.
            body: if id == var {
                body.clone()
            } else {
                Box::new(subst(body, var, value, budget)?)
            },
        },
        Node::Op { prim, args } => Node::Op {
            prim: prim.clone(),
            args: args
                .iter()
                .map(|a| subst(a, var, value, budget))
                .collect::<Result<Vec<_>, _>>()?,
        },
        Node::Swap {
            src,
            target,
            policy,
        } => Node::Swap {
            src: Box::new(subst(src, var, value, budget)?),
            target: target.clone(),
            policy: policy.clone(),
        },
        Node::Construct { ctor, args } => Node::Construct {
            ctor: ctor.clone(),
            args: args
                .iter()
                .map(|a| subst(a, var, value, budget))
                .collect::<Result<Vec<_>, _>>()?,
        },
        Node::Match {
            scrutinee,
            alts,
            default,
        } => Node::Match {
            scrutinee: Box::new(subst(scrutinee, var, value, budget)?),
            alts: alts
                .iter()
                .map(|alt| match alt {
                    Alt::Ctor {
                        ctor,
                        binders,
                        body,
                    } => Ok(Alt::Ctor {
                        ctor: ctor.clone(),
                        binders: binders.clone(),
                        // Shadowing: an arm binder that re-binds `var` blocks substitution in its body.
                        body: if binders.iter().any(|b| b == var) {
                            body.clone()
                        } else {
                            subst(body, var, value, budget)?
                        },
                    }),
                    Alt::Lit { value: lit, body } => Ok(Alt::Lit {
                        value: lit.clone(),
                        body: subst(body, var, value, budget)?,
                    }),
                })
                .collect::<Result<Vec<_>, EvalError>>()?,
            default: match default.as_ref() {
                Some(d) => Some(Box::new(subst(d, var, value, budget)?)),
                None => None,
            },
        },
        // r4: a Lam/Fix binder shadows `var` in its body; App substitutes into both positions.
        Node::Lam { param, body } => Node::Lam {
            param: param.clone(),
            body: if param == var {
                body.clone()
            } else {
                Box::new(subst(body, var, value, budget)?)
            },
        },
        Node::App { func, arg } => Node::App {
            func: Box::new(subst(func, var, value, budget)?),
            arg: Box::new(subst(arg, var, value, budget)?),
        },
        Node::Fix { name, body } => Node::Fix {
            name: name.clone(),
            body: if name == var {
                body.clone()
            } else {
                Box::new(subst(body, var, value, budget)?)
            },
        },
        // A `FixGroup` binds *every* member name; substitution shadows when `var` is one of them
        // (the group rebinds it), otherwise it descends into each definition and the continuation.
        Node::FixGroup { defs, body } => {
            if defs.iter().any(|(name, _)| name == var) {
                node.clone()
            } else {
                Node::FixGroup {
                    defs: defs
                        .iter()
                        .map(|(name, d)| {
                            Ok((name.clone(), Box::new(subst(d, var, value, budget)?)))
                        })
                        .collect::<Result<Vec<_>, EvalError>>()?,
                    body: Box::new(subst(body, var, value, budget)?),
                }
            }
        }
    })
}

#[cfg(test)]
mod data_tests {
    //! The r3 data fragment at the L0 boundary (RFC-0011): `Construct`/`Match` evaluation, the
    //! meet-summary guarantee, and the never-silent refusals. These pin the L0 semantics
    //! independently of the L1 elaborator (the L1↔L0 agreement is the M-210 differential).
    use super::*;
    use mycelium_core::{
        Bound, BoundBasis, BoundKind, CtorRef, CtorSpec, DataRegistry, DeclSpec, FieldSpec, Meta,
        NormKind, Payload, Provenance, Repr, Value,
    };
    use std::collections::BTreeMap;

    /// `type Nat = Z | S(Nat)` plus `type Box = Mk(Binary{8})`.
    fn registry() -> DataRegistry {
        let mut m = BTreeMap::new();
        m.insert(
            "Nat".to_owned(),
            DeclSpec {
                ctors: vec![
                    CtorSpec { fields: vec![] },
                    CtorSpec {
                        fields: vec![FieldSpec::Data("Nat".to_owned())],
                    },
                ],
            },
        );
        m.insert(
            "Box".to_owned(),
            DeclSpec {
                ctors: vec![CtorSpec {
                    fields: vec![FieldSpec::Repr(Repr::Binary { width: 8 })],
                }],
            },
        );
        DataRegistry::build(&m).unwrap()
    }

    fn z(reg: &DataRegistry) -> Node {
        Node::Construct {
            ctor: reg.ctor_ref("Nat", 0).unwrap(),
            args: vec![],
        }
    }
    fn s(reg: &DataRegistry, inner: Node) -> Node {
        Node::Construct {
            ctor: reg.ctor_ref("Nat", 1).unwrap(),
            args: vec![inner],
        }
    }
    fn byte(g: GuaranteeStrength) -> Value {
        let meta = if g == GuaranteeStrength::Exact {
            Meta::exact(Provenance::Root)
        } else {
            Meta::new(
                Provenance::Root,
                g,
                Some(Bound {
                    kind: BoundKind::Error {
                        eps: 0.1,
                        norm: NormKind::Linf,
                    },
                    basis: BoundBasis::EmpiricalFit {
                        trials: 1,
                        method: "m".into(),
                    },
                }),
                None,
                None,
                None,
            )
            .unwrap()
        };
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![false; 8]),
            meta,
        )
        .unwrap()
    }

    fn datum(reg: &DataRegistry, ty: &str, i: u32, fields: Vec<CoreValue>) -> CoreValue {
        CoreValue::Data(Datum::new(reg.ctor_ref(ty, i).unwrap(), fields))
    }

    #[test]
    fn construct_evaluates_to_a_datum() {
        let reg = registry();
        let interp = Interpreter::default();
        // S(S(Z)) ⟶ the data value S(S(Z)).
        let node = s(&reg, s(&reg, z(&reg)));
        let v = interp.eval_core(&node).expect("evaluates");
        assert_eq!(
            v,
            datum(
                &reg,
                "Nat",
                1,
                vec![datum(&reg, "Nat", 1, vec![datum(&reg, "Nat", 0, vec![])])]
            )
        );
    }

    #[test]
    fn eval_on_a_data_result_is_an_explicit_refusal() {
        // The repr-only `eval` must refuse a data result explicitly (never silently mishandle it).
        let reg = registry();
        let err = Interpreter::default().eval(&z(&reg)).unwrap_err();
        assert_eq!(err, EvalError::DataResult);
    }

    #[test]
    fn match_selects_the_constructor_arm_and_binds_fields() {
        // match S(Z) { Z => Z, S(m) => m } ⟶ Z   (the S arm binds m = Z).
        let reg = registry();
        let node = Node::Match {
            scrutinee: Box::new(s(&reg, z(&reg))),
            alts: vec![
                Alt::Ctor {
                    ctor: reg.ctor_ref("Nat", 0).unwrap(),
                    binders: vec![],
                    body: z(&reg),
                },
                Alt::Ctor {
                    ctor: reg.ctor_ref("Nat", 1).unwrap(),
                    binders: vec!["m".to_owned()],
                    body: Node::Var("m".to_owned()),
                },
            ],
            default: None,
        };
        let v = Interpreter::default().eval_core(&node).expect("evaluates");
        assert_eq!(v, datum(&reg, "Nat", 0, vec![]));
    }

    #[test]
    fn match_picks_the_first_matching_arm_not_a_later_one() {
        // Mutant-witness: matching Z must take the Z arm, not the S arm or the default.
        let reg = registry();
        let node = Node::Match {
            scrutinee: Box::new(z(&reg)),
            alts: vec![Alt::Ctor {
                ctor: reg.ctor_ref("Nat", 0).unwrap(),
                binders: vec![],
                body: s(&reg, z(&reg)), // Z arm yields S(Z) so we can tell which arm fired
            }],
            default: Some(Box::new(z(&reg))),
        };
        let v = Interpreter::default().eval_core(&node).expect("evaluates");
        assert_eq!(
            v,
            datum(&reg, "Nat", 1, vec![datum(&reg, "Nat", 0, vec![])])
        );
    }

    #[test]
    fn literal_arm_matches_on_repr_and_payload() {
        // match Mk(0b1111_1111) { Mk(b) => match b { 0b1111_1111 => Z, _ => S(Z) } }
        let reg = registry();
        let all_ones = Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![true; 8]),
            Meta::exact(Provenance::Root),
        )
        .unwrap();
        let inner_match = Node::Match {
            scrutinee: Box::new(Node::Var("b".to_owned())),
            alts: vec![Alt::Lit {
                value: all_ones.clone(),
                body: z(&reg),
            }],
            default: Some(Box::new(s(&reg, z(&reg)))),
        };
        let node = Node::Match {
            scrutinee: Box::new(Node::Construct {
                ctor: reg.ctor_ref("Box", 0).unwrap(),
                args: vec![Node::Const(all_ones)],
            }),
            alts: vec![Alt::Ctor {
                ctor: reg.ctor_ref("Box", 0).unwrap(),
                binders: vec!["b".to_owned()],
                body: inner_match,
            }],
            default: None,
        };
        let v = Interpreter::default().eval_core(&node).expect("evaluates");
        assert_eq!(v, datum(&reg, "Nat", 0, vec![])); // the 0b1111_1111 literal arm fired → Z
    }

    #[test]
    fn no_match_and_no_default_is_an_explicit_non_exhaustive_error() {
        // Mutant-witness: a Match with a non-covering alt set and no default must refuse, not hang
        // or default silently (WF7 is the checker's job, but the kernel never silently assumes it).
        let reg = registry();
        let node = Node::Match {
            scrutinee: Box::new(s(&reg, z(&reg))),
            alts: vec![Alt::Ctor {
                ctor: reg.ctor_ref("Nat", 0).unwrap(), // only Z covered; scrutinee is S(Z)
                binders: vec![],
                body: z(&reg),
            }],
            default: None,
        };
        assert_eq!(
            Interpreter::default().eval_core(&node).unwrap_err(),
            EvalError::NonExhaustiveMatch
        );
    }

    #[test]
    fn construct_summary_guarantee_is_the_meet_of_fields() {
        // Mk(Empirical byte) → an Empirical data value (honesty degrades — RFC-0011 §4.6).
        let reg = registry();
        let node = Node::Construct {
            ctor: reg.ctor_ref("Box", 0).unwrap(),
            args: vec![Node::Const(byte(GuaranteeStrength::Empirical))],
        };
        let v = Interpreter::default().eval_core(&node).expect("evaluates");
        assert_eq!(v.guarantee(), GuaranteeStrength::Empirical);
    }

    #[test]
    fn matching_a_non_exact_data_scrutinee_is_the_explicit_r3_boundary() {
        // match Mk(Empirical) { Mk(b) => b } — the scrutinee's summary is Empirical, so the
        // guarantee-meet through Match is the explicit r3 deferral (never a fabricated bound).
        let reg = registry();
        let node = Node::Match {
            scrutinee: Box::new(Node::Construct {
                ctor: reg.ctor_ref("Box", 0).unwrap(),
                args: vec![Node::Const(byte(GuaranteeStrength::Empirical))],
            }),
            alts: vec![Alt::Ctor {
                ctor: reg.ctor_ref("Box", 0).unwrap(),
                binders: vec!["b".to_owned()],
                body: Node::Var("b".to_owned()),
            }],
            default: None,
        };
        assert_eq!(
            Interpreter::default().eval_core(&node).unwrap_err(),
            EvalError::GuaranteeMeetUnsupported {
                scrutinee: GuaranteeStrength::Empirical
            }
        );
    }

    #[test]
    fn aot_lowerable_now_spans_the_data_fragment() {
        // M-342 (RFC-0011 §4.4 Q5 closed): the AOT env-machine covers the data fragment, so a
        // Construct is AOT-lowerable too (it runs on the three-way differential, not just the repr
        // path). The predicate is now total over the v0 node set.
        let reg = registry();
        assert!(z(&reg).is_aot_lowerable());
        assert!(Node::Const(byte(GuaranteeStrength::Exact)).is_aot_lowerable());
    }

    fn _unused(_: CtorRef) {}
}

#[cfg(test)]
mod r4_tests {
    //! r4 functions + recursion at the L0 boundary (RFC-0001 r4): β-reduction, Fix unfolding under
    //! the fuel clock, and the never-silent refusals. Pins the L0 semantics independently of the
    //! elaborator (the L1↔L0 agreement is the M-210 differential).
    use super::*;
    use mycelium_core::{
        CtorSpec, DataRegistry, DeclSpec, FieldSpec, Meta, Payload, Provenance, Repr, Value,
    };
    use std::collections::BTreeMap;

    fn nat() -> DataRegistry {
        let mut m = BTreeMap::new();
        m.insert(
            "Nat".to_owned(),
            DeclSpec {
                ctors: vec![
                    CtorSpec { fields: vec![] },
                    CtorSpec {
                        fields: vec![FieldSpec::Data("Nat".to_owned())],
                    },
                ],
            },
        );
        DataRegistry::build(&m).unwrap()
    }
    fn z(r: &DataRegistry) -> Node {
        Node::Construct {
            ctor: r.ctor_ref("Nat", 0).unwrap(),
            args: vec![],
        }
    }
    fn s(r: &DataRegistry, n: Node) -> Node {
        Node::Construct {
            ctor: r.ctor_ref("Nat", 1).unwrap(),
            args: vec![n],
        }
    }
    fn byte(bits: [bool; 8]) -> Node {
        Node::Const(
            Value::new(
                Repr::Binary { width: 8 },
                Payload::Bits(bits.to_vec()),
                Meta::exact(Provenance::Root),
            )
            .unwrap(),
        )
    }

    #[test]
    fn beta_reduction_applies_a_closed_lambda() {
        // (λx. not(x)) 0b0000_1111  ⟶  not(0b0000_1111) = 0b1111_0000
        let lam = Node::Lam {
            param: "x".into(),
            body: Box::new(Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::Var("x".into())],
            }),
        };
        let app = Node::App {
            func: Box::new(lam),
            arg: Box::new(byte([false, false, false, false, true, true, true, true])),
        };
        let v = Interpreter::default().eval(&app).expect("runs");
        assert_eq!(
            v.payload(),
            &Payload::Bits(vec![true, true, true, true, false, false, false, false])
        );
    }

    #[test]
    fn curried_application_reduces_left_to_right() {
        // (λx. λy. xor(x, y)) a b
        let lam = Node::Lam {
            param: "x".into(),
            body: Box::new(Node::Lam {
                param: "y".into(),
                body: Box::new(Node::Op {
                    prim: "bit.xor".into(),
                    args: vec![Node::Var("x".into()), Node::Var("y".into())],
                }),
            }),
        };
        let app = Node::App {
            func: Box::new(Node::App {
                func: Box::new(lam),
                arg: Box::new(byte([true, true, true, true, false, false, false, false])),
            }),
            arg: Box::new(byte([false, false, false, false, true, true, true, true])),
        };
        let v = Interpreter::default().eval(&app).expect("runs");
        assert_eq!(v.payload(), &Payload::Bits(vec![true; 8])); // xor of disjoint halves = all ones
    }

    /// `drop_ = Fix(f, λn. match n { Z => Z, S(m) => f m })` — structural recursion to Z.
    fn drop_(r: &DataRegistry) -> Node {
        Node::Fix {
            name: "f".into(),
            body: Box::new(Node::Lam {
                param: "n".into(),
                body: Box::new(Node::Match {
                    scrutinee: Box::new(Node::Var("n".into())),
                    alts: vec![
                        Alt::Ctor {
                            ctor: r.ctor_ref("Nat", 0).unwrap(),
                            binders: vec![],
                            body: z(r),
                        },
                        Alt::Ctor {
                            ctor: r.ctor_ref("Nat", 1).unwrap(),
                            binders: vec!["m".into()],
                            body: Node::App {
                                func: Box::new(Node::Var("f".into())),
                                arg: Box::new(Node::Var("m".into())),
                            },
                        },
                    ],
                    default: None,
                }),
            }),
        }
    }

    #[test]
    fn fix_drives_structural_recursion_to_a_value() {
        // drop_(S(S(S(Z)))) ⟶ Z
        let r = nat();
        let app = Node::App {
            func: Box::new(drop_(&r)),
            arg: Box::new(s(&r, s(&r, s(&r, z(&r))))),
        };
        let v = Interpreter::default().eval_core(&app).expect("terminates");
        assert_eq!(v.as_data().expect("data").fields().len(), 0, "Z");
    }

    #[test]
    fn an_unproductive_fix_exhausts_fuel_explicitly() {
        // Fix(f, f) loops; the fuel clock makes it an explicit refusal, never a hang.
        let spin = Node::Fix {
            name: "f".into(),
            body: Box::new(Node::Var("f".into())),
        };
        let err = Interpreter::default()
            .with_fuel(100)
            .eval_core(&spin)
            .unwrap_err();
        assert_eq!(err, EvalError::FuelExhausted);
    }

    #[test]
    fn applying_a_non_function_is_an_explicit_refusal() {
        // (0b…)(0b…) — applying a representation value is a type error the checker would catch.
        let app = Node::App {
            func: Box::new(byte([false; 8])),
            arg: Box::new(byte([true; 8])),
        };
        assert_eq!(
            Interpreter::default().eval_core(&app).unwrap_err(),
            EvalError::ApplyNonFunction
        );
    }

    #[test]
    fn a_function_result_is_an_explicit_refusal() {
        // A program that evaluates to a bare lambda is not an observable v0 result.
        let lam = Node::Lam {
            param: "x".into(),
            body: Box::new(Node::Var("x".into())),
        };
        assert_eq!(
            Interpreter::default().eval_core(&lam).unwrap_err(),
            EvalError::FunctionResult
        );
    }

    #[test]
    fn lam_app_fix_are_now_aot_lowerable() {
        // M-342: the recursion fragment (Lam/App/Fix/Match/Construct) lowers to ANF and runs on the
        // AOT env-machine, so a recursive definition is AOT-lowerable (three-way differential).
        let r = nat();
        assert!(drop_(&r).is_aot_lowerable());
    }
}

#[cfg(test)]
mod mutant_witness_tests {
    //! Mutant-witness tests for lib.rs survivors (M-654 Gate A3).
    //! Each test is annotated with the mutant location it targets.
    use super::*;
    use mycelium_core::{
        Alt, CtorSpec, DataRegistry, DeclSpec, FieldSpec, Meta, Payload, Provenance, Repr, Value,
    };
    use std::collections::BTreeMap;

    fn nat() -> DataRegistry {
        let mut m = BTreeMap::new();
        m.insert(
            "Nat".to_owned(),
            DeclSpec {
                ctors: vec![
                    CtorSpec { fields: vec![] },
                    CtorSpec {
                        fields: vec![FieldSpec::Data("Nat".to_owned())],
                    },
                ],
            },
        );
        DataRegistry::build(&m).unwrap()
    }
    fn z(r: &DataRegistry) -> Node {
        Node::Construct {
            ctor: r.ctor_ref("Nat", 0).unwrap(),
            args: vec![],
        }
    }
    fn s(r: &DataRegistry, n: Node) -> Node {
        Node::Construct {
            ctor: r.ctor_ref("Nat", 1).unwrap(),
            args: vec![n],
        }
    }
    fn byte_val() -> Value {
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![true, false, true, false, true, false, true, false]),
            Meta::exact(Provenance::Root),
        )
        .unwrap()
    }

    // ---- lib.rs:232 — Display for EvalError → Ok(Default::default()) ----
    // Mutant: the fmt body becomes a no-op — all variants format as empty string.
    // Kill: assert the formatted output contains the content-specific field (the var name).
    #[test]
    fn eval_error_display_is_non_empty_and_contains_payload() {
        // Mutant-witness: lib.rs:232 replace fmt → Ok(Default::default()).
        let msg = EvalError::FreeVariable("the_var".to_owned()).to_string();
        assert!(
            msg.contains("the_var"),
            "Display for FreeVariable must include the var name; got: {msg:?}"
        );
        let msg2 = EvalError::UnknownPrim("bit.nope".to_owned()).to_string();
        assert!(
            msg2.contains("bit.nope"),
            "Display for UnknownPrim must include the prim name; got: {msg2:?}"
        );
        // FuelExhausted is a unit variant — just check it's non-empty.
        assert!(
            !EvalError::FuelExhausted.to_string().is_empty(),
            "Display for FuelExhausted must not be empty"
        );
    }

    // ---- lib.rs:335 — Interpreter::prim_names → vec![] / vec![""] / vec!["xyzzy"] ----
    // Mutant: prim_names returns wrong/empty vec, losing knowledge of registered prims.
    // Kill: assert specific known built-in names appear in the returned list.
    #[test]
    fn prim_names_contains_known_builtins() {
        // Mutant-witness: lib.rs:335 prim_names mutants.
        let interp = Interpreter::default();
        let names = interp.prim_names();
        for expected in &[
            "core.id", "bit.not", "bit.and", "bit.or", "bit.xor", "trit.neg", "trit.add",
            "trit.sub", "trit.mul",
        ] {
            assert!(
                names.contains(expected),
                "prim_names must contain '{expected}'; got {names:?}"
            );
        }
        assert!(
            !names.is_empty(),
            "prim_names must not be empty for the default interpreter"
        );
    }

    // ---- lib.rs:483 (delete Node::Var arm) and lib.rs:485 (== → !=) in Interpreter::step ----
    // Mutant A: deleting the Var arm causes FixGroup to always use the continuation body clone,
    //   never looking up the named member — Var("g") is not substituted → stays a Var → FreeVariable.
    // Mutant B: == → != inverts the lookup, finding the WRONG member (or none) by name.
    // Kill: FixGroup with Var("g") body where "g" names a member must unfold to that member's def.
    #[test]
    fn fix_group_var_body_selects_the_correct_named_member() {
        // Mutant-witness: lib.rs:483 (delete Var arm) and lib.rs:485 (== → !=).
        // FixGroup([("f", Z), ("g", S(Z))], Var("g")).
        // Correct: Var("g") arm fires, find "g" → its def S(Z) is the target, then subst each name.
        // Mutant A (no Var arm): target = *body = Var("g") → stays as Var → after substitution,
        //   FixGroup replaces "g" with its thunk, giving FixGroup(..., Var("g")=the focus thunk)
        //   which loops. Or the substitution produces a nested FixGroup, which may not terminate.
        //   Actually: the Var("g") body clone has "g" substituted by a FixGroup thunk, so
        //   the result is a FixGroup whose body is Node::FixGroup (a non-Var body), unfolding
        //   differently — the g def S(Z) never appears correctly.
        // Mutant B (== → !=): find() finds the first member whose name != "g", so "f" is found
        //   (since "f" != "g") → target = f's def = Z (0 fields), not g's (1 field).
        // Kill by running to completion and checking the final datum has 1 field (S, not Z).
        let r = nat();
        let node = Node::FixGroup {
            defs: vec![
                ("f".to_owned(), Box::new(z(&r))),
                ("g".to_owned(), Box::new(s(&r, z(&r)))),
            ],
            body: Box::new(Node::Var("g".to_owned())),
        };
        // Run with ample fuel; since g's def S(Z) has no recursive calls to f or g inside,
        // the group unfolds to S(Z) (S refs are in the normal form already).
        let interp = Interpreter::default().with_fuel(10_000);
        let result = interp
            .eval_core(&node)
            .expect("FixGroup with non-recursive defs must terminate");
        let datum = result.as_data().expect("must be a data value");
        assert_eq!(
            datum.fields().len(),
            1,
            "FixGroup Var('g') body must unfold to g's def S(Z) (1 field), not f's Z (0 fields)"
        );
    }

    // ---- lib.rs:545 — delete Node::Var arm in node_to_core_value ----
    // Mutant: the Var arm is removed → a free Var inside a Construct arg falls to the wildcard
    //   `_ => DataMalformed` branch instead of FreeVariable.
    // Kill: eval_core of a Construct with a free-Var arg must yield FreeVariable, not DataMalformed.
    #[test]
    fn construct_with_free_var_arg_yields_free_variable() {
        // Mutant-witness: lib.rs:545 delete Node::Var arm in node_to_core_value.
        // Construct{S, [Var("x")]} has a free Var arg. eval_core calls step → step tries step(Var("x"))
        // → FreeVariable("x") from the Var arm in step(). So this test also demonstrates that the
        // overall eval_core path correctly surfaces FreeVariable rather than DataMalformed.
        // (The node_to_core_value path for a Var in args is only reachable if a Var somehow survived
        // as a "value" — an internal invariant. Both paths must give FreeVariable.)
        let r = nat();
        let node = Node::Construct {
            ctor: r.ctor_ref("Nat", 1).unwrap(),
            args: vec![Node::Var("x".to_owned())],
        };
        let err = Interpreter::default().eval_core(&node).unwrap_err();
        assert_eq!(
            err,
            EvalError::FreeVariable("x".to_owned()),
            "Construct with free Var arg must yield FreeVariable, not DataMalformed (lib.rs:545)"
        );
    }

    // ---- lib.rs:566 — delete Node::Var arm in guarantee_of_value ----
    // Mutant: the Var arm in guarantee_of_value is removed → falls through to DataMalformed
    //   rather than FreeVariable when a Var appears as an arg to a Construct scrutinee.
    // Kill: Match with a Construct scrutinee containing a free Var must yield FreeVariable.
    #[test]
    fn match_on_construct_with_free_var_field_yields_free_variable() {
        // Mutant-witness: lib.rs:566 delete Node::Var arm in guarantee_of_value.
        // Build: Match{ Construct{S, [Var("free")]}, [...], None }.
        // step(Match{Construct{S,[Var("free")]},...}) → step(Construct{S,[Var("free")]})
        // → step(Var("free")) → Err(FreeVariable("free")).
        // This surfaces before guarantee_of_value is called (which requires a fully-valued scrutinee).
        // The defensive arm at lib.rs:566 only fires if guarantee_of_value gets a Var directly
        // (internal invariant violation); the public-API test confirms FreeVariable propagates correctly.
        let r = nat();
        let node = Node::Match {
            scrutinee: Box::new(Node::Construct {
                ctor: r.ctor_ref("Nat", 1).unwrap(),
                args: vec![Node::Var("free".to_owned())],
            }),
            alts: vec![Alt::Ctor {
                ctor: r.ctor_ref("Nat", 1).unwrap(),
                binders: vec!["m".to_owned()],
                body: z(&r),
            }],
            default: None,
        };
        let err = Interpreter::default().eval_core(&node).unwrap_err();
        assert_eq!(
            err,
            EvalError::FreeVariable("free".to_owned()),
            "Match with free Var in Construct scrutinee must yield FreeVariable (lib.rs:566)"
        );
    }

    // ---- lib.rs:609 — select_arm: && → || (literal arm repr AND payload check) ----
    // Mutant: the literal arm matches on repr || payload — so same-repr values with different
    //   payloads would wrongly match.
    // Kill: a literal arm with all-zeros must NOT match an all-ones scrutinee (same repr, diff payload).
    #[test]
    fn literal_arm_requires_both_repr_and_payload_match() {
        // Mutant-witness: lib.rs:609 replace && with || in select_arm literal branch.
        // Lit arm value: Binary{8}, all-zeros payload.
        // Scrutinee: Binary{8}, all-ones payload. Same repr, different payload → should NOT match.
        // With mutant (||): repr matches → wrongly fires the literal arm → result is wrong.
        let zero_byte = Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![false; 8]),
            Meta::exact(Provenance::Root),
        )
        .unwrap();
        let ones_byte = Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![true; 8]),
            Meta::exact(Provenance::Root),
        )
        .unwrap();
        // match ones_byte { [zeros] => zeros, _ => ones } → default fires → ones_byte result.
        let node = Node::Match {
            scrutinee: Box::new(Node::Const(ones_byte.clone())),
            alts: vec![Alt::Lit {
                value: zero_byte.clone(),
                // Arm body = zero_byte (sentinel: if this fires we got the wrong result).
                body: Node::Const(zero_byte),
            }],
            default: Some(Box::new(Node::Const(ones_byte))),
        };
        let v = Interpreter::default().eval_core(&node).expect("evaluates");
        match v {
            CoreValue::Repr(val) => assert_eq!(
                val.payload(),
                &Payload::Bits(vec![true; 8]),
                "payload mismatch → default must fire; mutant (||) would fire the literal arm"
            ),
            _ => panic!("expected Repr result"),
        }
    }

    // ---- lib.rs:626 — delete Node::Var arm in as_const ----
    // Mutant: the Var(x) arm is removed; as_const falls through to the wildcard, returning
    //   FreeVariable("<non-value normal form>") instead of FreeVariable(the actual name).
    // Kill: the Swap path with a Var source must propagate FreeVariable with the correct name.
    #[test]
    fn swap_with_free_var_source_propagates_correct_var_name() {
        // Mutant-witness: lib.rs:626 delete Node::Var arm in as_const.
        // Swap{Var("q"), Binary{8}, policy}: step(Swap{Var("q"), ...}) → step(Var("q"))
        //   → Err(FreeVariable("q")) from the Var arm in step(), BEFORE as_const is called.
        // The as_const Var arm (lib.rs:626) is defensive (internal invariant). The test confirms
        // the public observable: a Swap with a Var source → FreeVariable with the correct name.
        use mycelium_core::ContentHash;
        let policy = ContentHash::parse("blake3:round_trip_safe").unwrap();
        let node = Node::Swap {
            src: Box::new(Node::Var("q".to_owned())),
            target: Repr::Binary { width: 8 },
            policy,
        };
        let err = Interpreter::default().eval(&node).unwrap_err();
        assert_eq!(
            err,
            EvalError::FreeVariable("q".to_owned()),
            "Swap with Var src must give FreeVariable with the var name (lib.rs:626)"
        );
    }

    // ---- lib.rs:724 — subst Fix binder shadowing (== → !=) ----
    // Mutant: Fix binder shadow check inverted: substitution DESCENDS INTO the body even when
    //   the Fix's own name shadows the substituted var — replacing the recursive reference.
    // Kill: `let x = v in Fix("x", Var("x"))` must loop (FuelExhausted), not evaluate to v.
    #[test]
    fn fix_binder_shadows_outer_substitution() {
        // Mutant-witness: lib.rs:724 replace == with != in subst Fix case.
        // let x = <byte> in Fix("x", Var("x")):
        //   - Let binds x → substitutes x in Fix{"x", Var("x")}.
        //   - Fix re-binds "x", so the shadow condition fires: body is NOT substituted.
        //   - Fix{"x", Var("x")} unproductively loops (Fix(x,x) → Fix(x,x) → ...) → FuelExhausted.
        // With mutant (== → !=): the shadow condition fires when name != var, i.e. when "x" != "x"
        //   is false — wait, != means the body IS substituted when name != var (same as correct).
        //   But actually: correct is `if name == var { body.clone() } else { subst(body) }`.
        //   Mutant (== → !=): `if name != var { body.clone() } else { subst(body) }` — INVERTED.
        //   So when name == var ("x" == "x" is true, but check is !=, so false): goes to ELSE →
        //   subst(body, "x", v). The Var("x") in body becomes Const(v).
        //   Fix{"x", Const(v)} → unfold: subst(Const(v), "x", Fix{"x", Const(v)}) = Const(v)
        //   → evaluates to v successfully. But correct code → FuelExhausted.
        let v = byte_val();
        let node = Node::Let {
            id: "x".to_owned(),
            bound: Box::new(Node::Const(v.clone())),
            body: Box::new(Node::Fix {
                name: "x".to_owned(),
                body: Box::new(Node::Var("x".to_owned())),
            }),
        };
        // Correct: Fix binder shadows → Fix(x, Var(x)) is preserved → loops → FuelExhausted.
        // Mutant: Fix binder shadowing inverted → Fix(x, Const(v)) → evaluates to v.
        let err = Interpreter::default()
            .with_fuel(100)
            .eval_core(&node)
            .unwrap_err();
        assert_eq!(
            err,
            EvalError::FuelExhausted,
            "Fix binder must shadow outer substitution; broken shadow causes it to terminate (lib.rs:724)"
        );
    }

    // ---- lib.rs:733 — subst FixGroup binder shadowing (== → !=) ----
    // Mutant: FixGroup binder check inverted — substitution does NOT descend when it should.
    // Kill: a FixGroup whose non-member Var "z" is substituted must correctly replace it.
    #[test]
    fn fix_group_non_member_var_is_substituted() {
        // Mutant-witness: lib.rs:733 replace == with != in subst FixGroup case.
        // Correct: `if defs.iter().any(|(name,_)| name == var)` → stop at shadow.
        //   When var is NOT a member: condition false → ELSE branch → descend and substitute.
        // Mutant (== → !=): `any(|(name,_)| name != var)` — this is true whenever ANY member name
        //   differs from var, which is almost always true → wrongly prevents descent (stops when
        //   should descend), leaving "z" un-substituted as a free variable.
        // Test: let z = S(Z) in FixGroup([("x", Var("z"))], Var("x")).
        //   "z" is NOT a FixGroup member → substitution must replace Var("z") with S(Z) in defs.
        //   After subst: FixGroup([("x", S(Z))], Var("x")) → unfolds: target=S(Z), subst x←thunk.
        //   S(Z) has no Var("x") inside → result is S(Z). eval_core → data value with 1 field.
        //   With mutant: "z" is not substituted → Var("z") remains → free variable error.
        let r = nat();
        let sz = s(&r, z(&r));
        let node = Node::Let {
            id: "z".to_owned(),
            bound: Box::new(sz.clone()),
            body: Box::new(Node::FixGroup {
                defs: vec![("x".to_owned(), Box::new(Node::Var("z".to_owned())))],
                body: Box::new(Node::Var("x".to_owned())),
            }),
        };
        // Correct: substitution descends → FixGroup{[("x", S(Z))], Var("x")} → eval → S(Z).
        // Mutant: substitution stopped → FixGroup{[("x", Var("z"))], Var("x")} → Var("z") is free.
        let v = Interpreter::default()
            .eval_core(&node)
            .expect("evaluates — non-member var must be substituted (lib.rs:733)");
        let datum = v.as_data().expect("data value");
        assert_eq!(
            datum.fields().len(),
            1,
            "FixGroup: non-member var 'z' must be substituted; mutant leaves it free"
        );
    }
}

//! Parallel evaluation of **provably-pure** (effect-free) Core IR fragments (M-862; RFC-0008 §4.2;
//! ADR-034; DN-25). This is a **perf path**, validated against the sequential reference by a
//! differential — never a second semantics: [`Interpreter::eval_core`]/[`Interpreter::step`] stay
//! the trusted, small-step meaning of a program (RFC-0008 §4.2, mirroring the AOT path's
//! relationship to the interpreter, KC-3 — concurrency adds *scheduling*, never new *meaning*).
//!
//! # What gets parallelized — the **bounded, top-level-only** plan ([`plan_parallel`])
//! Only the **outermost independent-argument batch** of a pure fragment is fanned out: the direct
//! argument list of a top-level [`Node::Op`] or [`Node::Construct`] (the "independent pure Construct
//! elements" the M-862 issue names). Each of those arguments is then evaluated **sequentially,
//! wholly inside its own worker**, by the trusted small-step interpreter ([`Interpreter::step`]) —
//! with **no nested** [`Scheduler::run_indexed`]. Everything else (a `Let`/`Match`/`App`/`Fix` head,
//! or an `Op`/`Construct` with fewer than two arguments) is evaluated **wholly sequentially** through
//! the trusted [`Interpreter::eval_core`]. The selection is reified in [`ParallelPlan`] — explicit
//! and EXPLAIN-able, never a silent reorder (house rule #2 / G2).
//!
//! **Why top-level only (interim bound, VR-5/G2).** The relocated M-709/M-861 [`Scheduler`]
//! ([`mycelium_sched`]) spawns **fresh OS threads per [`Scheduler::run_indexed`] call** (the
//! persistent, bounded work-stealing pool is a deferred follow-up). Fanning parallelism *recursively*
//! at every `Op`/`Construct`/`App` node would therefore spawn `O(depth × fan-out)` OS threads — a
//! resource-exhaustion hazard, not a correctness one, but real. Bounding the fan-out to the single
//! outermost batch caps total spawned threads at that batch's width, which is safe and predictable.
//! **This is an explicit interim measure**; once the pooled scheduler lands, the bound can widen
//! (the differential/EXPLAIN contract here is unaffected by *how much* is parallelized).
//!
//! # The purity gate ([`is_pure`])
//! [`is_pure`] is a **whole-fragment, structural** (syntactic) predicate, deliberately conservative
//! and all-or-nothing: if *any* subterm is not provably effect-free, the *whole* fragment is
//! evaluated by the ordinary sequential [`Interpreter::eval_core`] — never a partial/mixed order.
//! Grounding for "provably pure":
//! - Every built-in [`crate::prims::PrimFn`] is documented pure ([`crate::prims`] module docs) —
//!   **except** the reserved `wild:`-namespaced host-capability escape hatch (RFC-0028 §4.3), the
//!   *only* place a `Node::Op` reaches an arbitrary, potentially-effectful host operation. `is_pure`
//!   therefore excludes any `Op` whose `prim` starts with `"wild:"`.
//! - [`Node::Swap`] delegates to a runtime-supplied `Box<dyn SwapEngine>` (crate::swap) whose
//!   concrete behaviour cannot be statically inspected from a `Node` alone. `is_pure` conservatively
//!   treats **every** `Swap` node as an opacity boundary — never assumed pure, even though the
//!   shipped [`crate::swap::IdentitySwapEngine`] happens to be.
//! - `Const`/`Var`/`Lam` are trivially pure (no reduction).
//!
//! # Tag: Empirical (differential-checked)
//! The equivalence `eval_core_parallel(e) == eval_core(e)` for `e` in the pure fragment is checked by
//! a corpus differential (`src/tests/parallel.rs`), not proven — tagged **Empirical** on the
//! transparency lattice (never upgraded to `Proven` without a checked side-condition, VR-5).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use mycelium_core::{Alt, CoreValue, Node};
use mycelium_sched::scheduler::Scheduler;
use mycelium_workstack::{ensure_sufficient_stack, RecursionBudget};

use crate::{collect_values, node_to_core_value, EvalError, Interpreter, Step};

/// Whether `node` is a **provably pure** (effect-free) Core IR fragment — the structural,
/// conservative, EXPLAIN-able gate. See the module docs for the grounding of each case.
/// All-or-nothing over the whole subtree: a single impure/opaque leaf (a `wild:` op, or any `Swap`)
/// makes the **entire** fragment ineligible, never just the leaf.
#[must_use]
pub fn is_pure(node: &Node) -> bool {
    // RFC-0041 W4: an **explicit work-stack** traversal — O(1) host stack for any depth, so a crafted
    // deep `Node` cannot `SIGABRT` this pure predicate (the RR-29 guard-hole this planner owned). The
    // semantics are byte-for-byte the prior recursive `all`-over-subtree: a single impure/opaque leaf
    // (a `wild:` op, or any `Swap`) makes the whole fragment ineligible; a `Lam` is pure *without*
    // descending into its unapplied body (unchanged — its body is not evaluated until applied).
    let mut work: Vec<&Node> = vec![node];
    while let Some(n) = work.pop() {
        match n {
            Node::Const(_) | Node::Var(_) | Node::Lam { .. } => {}
            // The reserved host-capability escape hatch (RFC-0028 §4.3) is the one channel a `Node::Op`
            // can reach an arbitrary, potentially-effectful implementation through; every other prim is
            // documented pure (`crate::prims` module docs).
            Node::Op { prim, args } => {
                if prim.starts_with("wild:") {
                    return false;
                }
                work.extend(args.iter());
            }
            // A `Box<dyn SwapEngine>` is an opaque, runtime-supplied implementation (crate::swap) — its
            // purity cannot be verified from the `Node` alone, so it is conservatively never pure.
            Node::Swap { .. } => return false,
            Node::Let { bound, body, .. } => {
                work.push(bound);
                work.push(body);
            }
            Node::Construct { args, .. } => work.extend(args.iter()),
            Node::Match {
                scrutinee,
                alts,
                default,
            } => {
                work.push(scrutinee);
                for a in alts {
                    match a {
                        Alt::Ctor { body, .. } | Alt::Lit { body, .. } => work.push(body),
                    }
                }
                if let Some(d) = default.as_deref() {
                    work.push(d);
                }
            }
            Node::App { func, arg } => {
                work.push(func);
                work.push(arg);
            }
            Node::Fix { body, .. } => work.push(body),
            Node::FixGroup { defs, body } => {
                for (_, d) in defs {
                    work.push(d);
                }
                work.push(body);
            }
        }
    }
    true
}

/// The head node of a parallelized top-level batch — the "independent pure elements" M-862 fans out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchHead {
    /// A top-level [`Node::Op`]: its argument list is the batch; the batch results are the prim's
    /// operands.
    Op,
    /// A top-level [`Node::Construct`]: its argument list is the batch; the batch results are the
    /// datum's fields.
    Construct,
}

/// The **reified, EXPLAIN-able** decision of what (if anything) [`Interpreter::eval_core_parallel`]
/// will parallelize for a given fragment — never a silent/opaque choice (house rule #2 / G2). A
/// caller/tool can ask *exactly* what the parallel evaluator intends to do before it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelPlan {
    /// The fragment is **impure** (a `wild:` op or a `Swap` appears somewhere) — evaluated wholly
    /// sequentially by [`Interpreter::eval_core`], never reordered.
    SequentialImpure,
    /// The fragment is pure but its **outermost node is not an independent-argument batch worth
    /// parallelizing** (fewer than two top-level arguments, or a `Let`/`Match`/`App`/`Fix`/… head) —
    /// evaluated wholly sequentially by [`Interpreter::eval_core`].
    SequentialNoBatch,
    /// The outermost node is a pure `Op`/`Construct` with **≥2 independent arguments**; those
    /// arguments are the batch dispatched across [`Scheduler::run_indexed`] (a single, non-nested
    /// fan-out), each evaluated sequentially within its worker.
    TopLevelBatch {
        /// Which node family heads the batch.
        head: BatchHead,
        /// The batch width (number of independent arguments fanned out).
        width: usize,
    },
}

/// The number of independent top-level arguments a fragment must have before the batch is fanned out
/// rather than run sequentially. Below this, the scheduling overhead is not worth a thread.
const MIN_BATCH_WIDTH: usize = 2;

/// Compute the [`ParallelPlan`] for `node` — the explicit, side-effect-free decision procedure
/// [`Interpreter::eval_core_parallel`] follows (and that a caller can inspect up front). Looks at the
/// **outermost node only** (the interim top-level bound; see module docs).
#[must_use]
pub fn plan_parallel(node: &Node) -> ParallelPlan {
    if !is_pure(node) {
        return ParallelPlan::SequentialImpure;
    }
    match node {
        Node::Op { args, .. } if args.len() >= MIN_BATCH_WIDTH => ParallelPlan::TopLevelBatch {
            head: BatchHead::Op,
            width: args.len(),
        },
        Node::Construct { args, .. } if args.len() >= MIN_BATCH_WIDTH => {
            ParallelPlan::TopLevelBatch {
                head: BatchHead::Construct,
                width: args.len(),
            }
        }
        _ => ParallelPlan::SequentialNoBatch,
    }
}

/// Tick the shared fuel counter (an [`AtomicU64`] so concurrent batch workers share one budget),
/// returning [`EvalError::FuelExhausted`] on underflow — never silent, and never a per-worker budget
/// that could let two threads jointly overrun the declared total (RFC-0007 §4.5 CakeML clock).
fn tick(fuel: &AtomicU64) -> Result<(), EvalError> {
    fuel.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |f| f.checked_sub(1))
        .map(|_| ())
        .map_err(|_| EvalError::FuelExhausted)
}

impl Interpreter {
    /// Evaluate `node` to a [`CoreValue`] with the **same result** as [`Interpreter::eval_core`],
    /// except that when the fragment's [`plan_parallel`] is a [`ParallelPlan::TopLevelBatch`] the
    /// outermost independent argument batch is evaluated **in parallel** on the M-861
    /// [`Scheduler`] — a single, non-nested [`Scheduler::run_indexed`] fan-out (the interim
    /// top-level bound; see module docs). Observable behaviour is identical to the sequential
    /// reference (RT2-preserving): [`Scheduler::run_indexed`] returns outputs in **spawn order**, so
    /// the result is deterministic regardless of the steal schedule, and each argument is reduced by
    /// the trusted sequential [`Interpreter::step`]. Any fragment that is impure, or whose outermost
    /// node is not a ≥2-argument `Op`/`Construct`, falls back **wholesale** to
    /// [`Interpreter::eval_core`] (never a partial/mixed order — G2).
    ///
    /// **Empirical, differential-checked** (M-862): `eval_core_parallel(e) == eval_core(e)` for `e`
    /// in the pure fragment is checked over a corpus in `src/tests/parallel.rs`, not proven.
    pub fn eval_core_parallel(&self, node: &Node) -> Result<CoreValue, EvalError> {
        match plan_parallel(node) {
            // Wholesale sequential — the trusted reference, never reordered.
            ParallelPlan::SequentialImpure | ParallelPlan::SequentialNoBatch => {
                self.eval_core(node)
            }
            ParallelPlan::TopLevelBatch { head, .. } => self.eval_top_batch(node, head),
        }
    }

    /// Evaluate `node` to a representation [`crate::Value`], mirroring [`Interpreter::eval`] — see
    /// [`Interpreter::eval_core_parallel`] for the parallel-evaluation contract.
    pub fn eval_parallel(&self, node: &Node) -> Result<crate::Value, EvalError> {
        match self.eval_core_parallel(node)? {
            CoreValue::Repr(v) => Ok(v),
            CoreValue::Data(_) => Err(EvalError::DataResult),
        }
    }

    /// Fan the outermost `Op`/`Construct` argument batch across the scheduler (a single, non-nested
    /// `run_indexed`), then recombine. Each argument is reduced to a normal-form [`Node`] by the
    /// trusted sequential small-step interpreter ([`Interpreter::eval_to_normal_node`]) inside its
    /// worker — so this introduces **no** second semantics, only scheduling. Only ever called for a
    /// [`ParallelPlan::TopLevelBatch`] node (a pure `Op`/`Construct` with ≥2 args).
    ///
    /// # Exact-equivalence discipline (fixes a fuel-starvation divergence, M-862 follow-up)
    /// The trusted sequential `eval_core`/`step` is strict left-to-right and **short-circuits**: it
    /// stops at the first erroring argument and never spends fuel on the ones after it. Dispatching
    /// every argument as a concurrent job against one shared fuel counter breaks that: a
    /// fuel-hungry/non-terminating sibling can drain the shared budget and starve an *earlier* arg
    /// into `FuelExhausted` where the sequential reference would have reached a deterministic
    /// non-fuel error first (or vice versa) — schedule-dependent, so it can even disagree with
    /// itself run to run.
    ///
    /// The fix keeps the parallel speedup exactly where it is safe and defers to the reference
    /// everywhere it might not be:
    /// - Snapshot `F0 = self.fuel` and run the batch on a **separate** clone of it (never the real
    ///   counter — there is nothing to corrupt if the attempt is discarded).
    /// - If **every** argument job returns `Ok` (the clone was never exhausted and no argument
    ///   errored), the parallel run cannot have diverged: sequential would have run the same
    ///   arguments to the same values, spending the same total fuel (order-independent for a simple
    ///   decrement-by-1 pool once every request succeeds). Commit that result.
    /// - If **any** argument job returns `Err` (including `FuelExhausted`) the parallel attempt may
    ///   have evaluated (or starved) siblings the sequential reference would never have reached —
    ///   **discard it entirely** (never commit its partially-consumed fuel) and re-evaluate the whole
    ///   node via the trusted sequential [`Interpreter::eval_core`] on a fresh `F0`. That is the
    ///   authoritative, deterministic answer.
    fn eval_top_batch(&self, node: &Node, head: BatchHead) -> Result<CoreValue, EvalError> {
        let args: &[Node] = match node {
            Node::Op { args, .. } | Node::Construct { args, .. } => args,
            // Unreachable: `eval_top_batch` is only reached via a `TopLevelBatch` plan, which is only
            // produced for `Op`/`Construct`. Refuse explicitly rather than panic (never-silent, G2).
            _ => {
                return Err(EvalError::DataMalformed {
                    why: "eval_top_batch reached a non-Op/Construct head".to_owned(),
                })
            }
        };

        // A fuel clone seeded at F0 = self.fuel — never the real/authoritative counter. If any job
        // errors we throw this attempt (and whatever it consumed) away entirely. `Arc`-shared (M-864:
        // `run_indexed` now requires `'static` jobs, so the shared counter can no longer be a plain
        // `&AtomicU64` borrow of a stack local — every job owns an `Arc::clone` of it instead, same
        // shared-counter semantics, just `'static`-owned).
        let fuel = Arc::new(AtomicU64::new(self.fuel));

        // M-864: a job can no longer borrow `&Interpreter` from this stack frame either (the shared
        // pool's worker threads outlive this call) — clone the interpreter once (`Interpreter` is
        // `Clone`: an `Arc`-bumped swap engine + a small `prims` map clone, see its struct docs) and
        // give each job its own cheap `Arc::clone` handle to that one clone, so `prims` is deep-cloned
        // once per batch, not once per job.
        let interp = Arc::new(self.clone());

        // One `run_indexed` fan-out; each job runs the SEQUENTIAL interpreter on its (now owned)
        // argument (no nested `run_indexed` — the interim top-level-only bound; M-864 made nested
        // submission cheap at the scheduler level, but this evaluator does not yet exploit it — see
        // module docs). Outputs come back in spawn order.
        let jobs: Vec<_> = args
            .iter()
            .map(|arg| {
                let interp = Arc::clone(&interp);
                let fuel = Arc::clone(&fuel);
                let arg = arg.clone();
                move || interp.eval_to_normal_node(&arg, &fuel)
            })
            .collect();
        let results: Vec<Result<Node, EvalError>> = Scheduler::new().run_indexed(jobs, None, None);

        // Any error anywhere in the batch (a genuine semantic error, or fuel starved by a sibling) —
        // discard this parallel attempt wholesale and defer to the trusted sequential reference,
        // which alone has the correct short-circuiting order. Never partially trust the clone.
        if results.iter().any(Result::is_err) {
            return self.eval_core(node);
        }
        let normals: Vec<Node> = results
            .into_iter()
            .map(|r| r.expect("checked all Ok above"))
            .collect();

        match head {
            BatchHead::Op => {
                let prim = match node {
                    Node::Op { prim, .. } => prim,
                    _ => unreachable!("head == Op ⇒ node is Op"),
                };
                // Apply δ exactly as the sequential `step` (E-Op-Apply) does: all args are now values.
                // A1: `wild:` routes through the host-op registry (unreachable for pure batches —
                // `is_pure` excludes `wild:` — kept for parity with sequential dispatch, G2).
                let values = collect_values(&normals)?;
                let result = if prim.starts_with("wild:") {
                    crate::wild::dispatch_wild(&self.host_ops, self.host_caps, prim, &values)?
                } else {
                    let f = self
                        .prims
                        .get(prim)
                        .ok_or_else(|| EvalError::UnknownPrim(prim.clone()))?;
                    f(prim, &values)?
                };
                tick(&fuel)?;
                Ok(CoreValue::Repr(result))
            }
            BatchHead::Construct => {
                let ctor = match node {
                    Node::Construct { ctor, .. } => ctor,
                    _ => unreachable!("head == Construct ⇒ node is Construct"),
                };
                // A saturated Construct of values is itself a normal form (a data value); read it off
                // exactly as `node_to_core_value` does for the sequential path.
                let rebuilt = Node::Construct {
                    ctor: ctor.clone(),
                    args: normals,
                };
                // RFC-0041 W4: the read-off is budgeted (fresh per-batch budget — this worker owns a
                // disjoint sub-value); a deep field spine refuses with `DepthLimit`, never a `SIGABRT`.
                node_to_core_value(&rebuilt, &RecursionBudget::default())
            }
        }
    }

    /// Reduce `node` to a **normal-form [`Node`]** (a `Const`, or a saturated `Construct` of values)
    /// by iterating the trusted sequential small-step [`Interpreter::step`] under the shared `fuel`
    /// clock. This is the exact [`Interpreter::eval_core`] loop, returning the normal-form node
    /// instead of reading it off — so a batch worker uses the trusted semantics verbatim, never a
    /// reimplementation. Runs entirely on one thread (no nested parallelism).
    fn eval_to_normal_node(&self, node: &Node, fuel: &AtomicU64) -> Result<Node, EvalError> {
        // RFC-0041 W4: this batch worker mirrors `eval_core`'s loop, so it runs on the growable deep
        // worker stack too — the budgeted `step` refuses a deep sub-value with `DepthLimit` well within
        // the stack, never a host-stack `SIGABRT` on a scheduler pool thread.
        let sizing = RecursionBudget::default();
        ensure_sufficient_stack(&sizing, move || {
            let mut current = node.clone();
            loop {
                match self.step(&current)? {
                    Step::Value => return Ok(current),
                    Step::Next(next) => {
                        tick(fuel)?;
                        current = *next;
                    }
                }
            }
        })
    }
}

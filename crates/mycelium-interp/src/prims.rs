//! Primitive operators for the interpreter (M-110).
//!
//! A [`Node::Op`](mycelium_core::Node::Op) names a primitive; the registry maps that name to an
//! implementation. The built-in set here is deliberately the **exact, elementwise** fragment — the
//! identity and per-element logical ops — so the reference interpreter is testable end-to-end
//! without prejudging **M-111** (balanced-ternary *arithmetic* — add/mul with carries, property-
//! tested against an integer oracle), which extends this registry.
//!
//! Every built-in is `guarantee = Exact` and type-checks its operands; a wrong arity, paradigm, or
//! width is an explicit [`EvalError::PrimType`], never a silent coercion (SC-3; G2). Result metadata
//! is threaded honestly: provenance becomes `Derived{ op: hash(prim), inputs: [hash(arg)…] }`
//! (RFC-0001 §4.6) and the guarantee is the `meet` of the inputs and the prim's intrinsic strength
//! (RFC-0001 §4.7). Exact inputs through an exact built-in stay `Exact` / `bound = None`.
//!
//! **Composing an approximate input (M-204; ADR-010).** With the verified-numerics kernels landed
//! (`mycelium-numerics`, E2-4), a built-in over an *approximate* input no longer refuses outright: it
//! composes the input's `Error` bound through the affine ε-kernel and meets the strength to the
//! weakest input ([`mycelium_numerics::compose_error_bound`]). Each prim declares its
//! `ApproxRule`: the additive ternary arithmetic (`trit.add/sub/neg`) carries a sound affine
//! composition; `core.id` passes the bound through unchanged; the logical `bit.*` ops and `trit.mul`
//! have **no defined ε-propagation rule over approximate inputs** and so still refuse
//! (`EvalError::ApproxCompositionUnsupported`) — refusing remains the honest choice over fabricating a
//! bound (G2/VR-5).

use std::cmp::Ordering;
use std::collections::BTreeMap;

use mycelium_core::{
    binary, operation_hash, ternary, Bound, BoundBasis, BoundKind, FloatWidth, GuaranteeStrength,
    Meta, NormKind, Payload, Provenance, Repr, Trit, Value, WrappingOpt,
};
use mycelium_dense::{DenseError, DenseSpace};
use mycelium_numerics::{compose_error_bound, ErrorOp};
use mycelium_vsa::{capacity, Bsc, CleanupMemory, Fhrr, MapI, Match, VsaError, VsaModel};

use crate::EvalError;

/// How a built-in composes an *approximate* input's bound (M-204). Exact inputs never reach this —
/// they short-circuit to an `Exact`/`bound = None` result.
#[derive(Debug, Clone, Copy)]
enum ApproxRule {
    /// No defined ε-propagation over an approximate input — refuse (honest; `bit.*`, `trit.mul`).
    Refuse,
    /// Unary identity (`core.id`): pass the single input's bound and strength through unchanged.
    Passthrough,
    /// Compose the inputs' `Error` bounds through the affine ε-kernel under this op (the additive
    /// ternary arithmetic — sound 1-Lipschitz propagation).
    Error(ErrorOp),
}

/// A primitive implementation: a pure function from argument values to a result value (or an error).
pub type PrimFn = fn(prim: &str, args: &[&Value]) -> Result<Value, EvalError>;

/// The name→implementation table the interpreter dispatches `Op` nodes through. Extensible: M-111
/// (arithmetic) and later passes register additional prims here.
#[derive(Clone, Default)]
pub struct PrimRegistry {
    table: BTreeMap<String, PrimFn>,
}

impl PrimRegistry {
    /// An empty registry.
    #[must_use]
    pub fn empty() -> Self {
        PrimRegistry {
            table: BTreeMap::new(),
        }
    }

    /// The default registry: the exact built-ins — elementwise logical (`core.id`,
    /// `bit.not/and/or/xor`), fixed-width balanced-ternary arithmetic (`trit.neg/add/sub/mul`,
    /// M-111), the reduce-to-`Bool` comparison prims (`cmp.eq`/`cmp.lt` → `Binary{1}`, RFC-0032 D1,
    /// M-747), never-silent fixed-width binary arithmetic (`bit.add`/`bit.sub`, RFC-0032 D2,
    /// M-748), the never-silent `Binary` width-cast (`bit.width_cast` — zero-extend widen / checked
    /// narrow, DN-41, M-798), the explicit total-but-lossy `Binary` narrow (`bit.truncate` —
    /// unconditionally drops the high bits, never refuses, `Declared` tag, DN-51 §2 D3/§6),
    /// never-silent indexed-sequence access (`seq.len`/`seq.get`, RFC-0032 D3,
    /// M-749), never-silent byte-string access
    /// (`bytes.len`/`bytes.get`/`bytes.slice`/`bytes.concat`, RFC-0032 D4, M-750), the never-silent
    /// two's-complement `Binary` multiply (`bin.mul`, RFC-0033 §4.1.2/§4.1.3, M-887 — the first
    /// `enb` Gap-B prim; sets the registry pattern the sibling Gap-B/C prims mirror), the
    /// never-silent **unsigned** `Binary` division/remainder (`bin.div`/`bin.rem`, RFC-0033
    /// §4.1.2/§4.1.3, M-888 — div-by-zero is an explicit refusal; the signed variant rides M-767),
    /// the never-silent **logical** `Binary` left/right shift (`bin.shl`/`bin.shr`, RFC-0033
    /// §4.1.2/§4.1.3, M-889 — a shift amount `>= N` is an explicit refusal; the arithmetic/signed
    /// variant rides M-767), and the never-silent two's-complement `add`/`sub`/`neg`
    /// (`bin.add`/`bin.sub`/`bin.neg`, RFC-0033 §4.1.2/§4.1.3, M-766 — completes the shared
    /// two's-complement set `bin.mul` started; distinct from `bit.add`/`bit.sub`'s unsigned overflow
    /// criterion), and the **signedness-split signed op set**
    /// (`bin.div_s`/`bin.rem_s`/`bin.shr_s`/`cmp.lt_s`, RFC-0033 §4.1.2/§4.1.3, M-767 — signed
    /// truncated-toward-zero division/remainder, the arithmetic sign-extending right shift, and
    /// the two's-complement ordering, each a distinct named op from its unsigned counterpart per
    /// ADR-028; the `min ÷ −1` overflow, div-by-zero, and an out-of-range shift amount are
    /// explicit refusals), and the **dense group**
    /// (`dense.add`/`dense.sub`/`dense.neg`/`dense.scale`, RFC-0001 §4.1/RFC-0002 §5, M-890 —
    /// `enb` Gap C; the first tensor-valued prims — plus the M-891 measurement pair
    /// `dense.dot`/`dense.similarity`; all delegate to the `mycelium-dense` kernel and
    /// carry its per-op guarantee tags unchanged — `dense.neg` `Exact`, the rest `Proven`; see
    /// the module note at the dense section below), and the **scalar-float arithmetic group**
    /// (`flt.add`/`flt.sub`/`flt.mul`/`flt.div`/`flt.neg`, ADR-040 §2.5, M-898 — `enb` Gap A;
    /// IEEE-754 binary64 under RNE over `Repr::Float`, arithmetic specials in-band per the
    /// ratified FLAG-2, per-op tag `Empirical` per ADR-040 §2.6 — see the float section note),
    /// and the **scalar-float comparison group** (`flt.lt`/`flt.le`/`flt.gt`/`flt.ge`/`flt.eq` —
    /// the IEEE-754 §5.11 partial-order predicates, any NaN operand → the *defined* result
    /// `false` — plus the named opt-in total order `flt.total_le` (IEEE-754 §5.10 `totalOrder`),
    /// ADR-040 §2.4, M-899 — `enb` Gap A; the total-order *property* stays `Empirical` until the
    /// M-511 proof debt is discharged — see the float comparison section note), and the
    /// **VSA bind group** (`vsa.bind`/`vsa.unbind`/`vsa.permute`, RFC-0003 §3/§4/ADR-008, M-892 —
    /// `enb` Gap C; **model-dispatched** MAP-I/FHRR/BSC on the operand's `Repr::Vsa` model id, an
    /// out-of-set model an explicit refusal; the `mycelium-vsa` kernel constructs the result with
    /// its honest per-model tag — MAP-I/BSC ops `Exact`, FHRR `unbind` `Empirical` with its
    /// trial-validated δ — carried unchanged; see the vsa section note below), and the
    /// **certified VSA superposition** (`vsa.bundle`, RFC-0003 §4/§5/ADR-008, M-893 — `enb`
    /// Gap C; MAP-I's `bundle_values_certified` over a `Seq` of hypervectors + a `Float` δ,
    /// dispatch set the certified singleton {MAP-I}; the kernel issues the `Proven`
    /// `CapacityBound` only under its checked side-conditions, carried unchanged — see the vsa
    /// section note below), and the **VSA cleanup/reconstruction pair + capacity query**
    /// (`vsa.cleanup`/`vsa.reconstruct`/`vsa.required_dim`, RFC-0003 §3/§5/§6/ADR-008, M-894 —
    /// `enb` Gap C; the FR-S4 cleanup-memory retrieval and the §6 compositional
    /// role-reconstruction, each returning the `[index, confidence, margin]` decision triple
    /// with the query/record's own (strength, bound) pair carried through, plus the M-131
    /// `requiredDim` query whose result carries the kernel-checked `Proven` `CapacityBound` —
    /// see the M-894 section note below).
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut r = PrimRegistry::empty();
        r.register("core.id", prim_id);
        r.register("bit.not", prim_bit_not);
        r.register("bit.and", prim_bit_and);
        r.register("bit.or", prim_bit_or);
        r.register("bit.xor", prim_bit_xor);
        r.register("trit.neg", prim_trit_neg);
        r.register("trit.add", prim_trit_add);
        r.register("trit.sub", prim_trit_sub);
        r.register("trit.mul", prim_trit_mul);
        // RFC-0032 D1 (M-747): reduce-to-`Bool` comparison/equality over `Binary{N}`/`Ternary{N}`.
        r.register("cmp.eq", prim_cmp_eq);
        r.register("cmp.lt", prim_cmp_lt);
        // RFC-0032 D2 (M-748): never-silent fixed-width binary arithmetic over `Binary{N}`.
        r.register("bit.add", prim_bit_add);
        r.register("bit.sub", prim_bit_sub);
        // RFC-0033 §4.1.2/§4.1.3 (M-887, `enb` Gap B): never-silent two's-complement multiply.
        r.register("bin.mul", prim_bin_mul);
        // RFC-0033 §4.1.2 (CU-1): never-silent UNSIGNED multiply — the `bit.*` unsigned family's
        // genuinely-missing member (math.myc FLAG-math-1), overflow-distinct from signed `bin.mul`.
        r.register("bit.mul", prim_bit_mul);
        // CU-6: width-preserving bit-manipulation counts (Rust count_ones/leading_zeros/
        // trailing_zeros) — total, signedness-free. popcount/clz/ctz as kernel prims (single host
        // instruction, not efficiently derivable in `.myc`); rotate/reverse ride `std.math`.
        r.register("bit.popcount", prim_bit_popcount);
        r.register("bit.clz", prim_bit_clz);
        r.register("bit.ctz", prim_bit_ctz);
        // RFC-0033 §4.1.2/§4.1.3 (M-888, `enb` Gap B): never-silent unsigned division/remainder.
        r.register("bin.div", prim_bin_div);
        r.register("bin.rem", prim_bin_rem);
        // RFC-0033 §4.1.2/§4.1.3 (M-889, `enb` Gap B): never-silent logical left/right shift.
        r.register("bin.shl", prim_bin_shl);
        r.register("bin.shr", prim_bin_shr);
        // RFC-0033 §4.1.2/§4.1.3 (M-766, `enb` Gap B): never-silent two's-complement add/sub/neg —
        // completes the shared two's-complement op set `bin.mul` started.
        r.register("bin.add", prim_bin_add);
        r.register("bin.sub", prim_bin_sub);
        r.register("bin.neg", prim_bin_neg);
        // RFC-0033 §4.1.2/§4.1.3 (M-767, `enb` Gap B): the signedness-split signed op set —
        // truncated signed division/remainder, the arithmetic (sign-extending) right shift, and
        // the two's-complement ordering, each a distinct named op from its unsigned counterpart
        // (ADR-028). Never-silent: div-by-zero and an out-of-range shift amount refuse as
        // `PrimType`; the single signed-division overflow (`min ÷ −1`) refuses as `Overflow`.
        r.register("bin.div_s", prim_bin_div_s);
        r.register("bin.rem_s", prim_bin_rem_s);
        r.register("bin.shr_s", prim_bin_shr_s);
        r.register("cmp.lt_s", prim_cmp_lt_s);
        // DN-41 (M-798): never-silent width-cast (zero-extend widen / checked narrow) over `Binary`.
        r.register("bit.width_cast", prim_width_cast);
        // DN-51 §2 D3/§6 (maintainer-authorized DN-39 post-freeze promotion): the explicit, total,
        // lossy `Binary` narrow — unconditionally drops the high bits, never refuses.
        r.register("bit.truncate", prim_truncate);
        // RFC-0032 D3 (M-749): never-silent indexed-sequence access over `Repr::Seq`.
        r.register("seq.len", prim_seq_len);
        r.register("seq.get", prim_seq_get);
        // RFC-0032 D4 (M-750): never-silent byte-string access over `Repr::Bytes`.
        r.register("bytes.len", prim_bytes_len);
        r.register("bytes.get", prim_bytes_get);
        r.register("bytes.slice", prim_bytes_slice);
        r.register("bytes.concat", prim_bytes_concat);
        // M-912 (`enb`, folded-in gap): byte-wise equality — flagged missing by the diag/error/
        // recover ports (`bytes.*` had len/get/slice/concat but no equality).
        r.register("bytes.eq", prim_bytes_eq);
        // M-912 (`enb`): the kernel's own BLAKE3 content-addressing hash (M-103) surfaced as a prim.
        r.register("hash.blake3", prim_hash_blake3);
        // DN-138 WU-4 (increment 2, §6): the missing raw `Binary{N} -> Bytes` conversion.
        r.register("bit.to_bytes", prim_bin_to_bytes);
        // DN-58 §A (M-817): the `Binary` `Fuse` semilattice meet (bitwise-AND). The user-`Data` fuse
        // does **not** register a prim — it elaborates to the resolved `Fuse::join` call (DN-58 §A.5);
        // the non-`Binary` reprs have no committed canonical meet in v0 (DN-58 §A.6 F-A3), so only the
        // `Binary` meet is a built-in here. (RFC-0008 RT6; RFC-0027 §10.6 provenance shape.)
        r.register("fuse_join:binary", prim_fuse_join_binary);
        // RFC-0001 §4.1 / RFC-0002 §5 (M-890/M-891, `enb` Gap C): the dense group — the
        // first *tensor-valued* prims. The kernel (`mycelium-dense`) constructs the result with
        // its honest per-op tag/bound; the wrappers carry it through unchanged (VR-5).
        r.register("dense.add", prim_dense_add);
        r.register("dense.sub", prim_dense_sub);
        r.register("dense.neg", prim_dense_neg);
        r.register("dense.scale", prim_dense_scale);
        r.register("dense.dot", prim_dense_dot);
        r.register("dense.similarity", prim_dense_similarity);
        // ADR-040 §2.5 (M-898, `enb` Gap A): the scalar-float arithmetic group over
        // `Repr::Float{F64}` — IEEE-754 binary64 under round-to-nearest-even, in-band specials
        // (±inf/NaN propagate as first-class values — the ratified FLAG-2 policy; never a trap,
        // never a silent wrap onto an ordinary value). Per-op tag `Empirical` per the ratified
        // ADR-040 §2.6 — see the module note at the float section below.
        r.register("flt.add", prim_flt_add);
        r.register("flt.sub", prim_flt_sub);
        r.register("flt.mul", prim_flt_mul);
        r.register("flt.div", prim_flt_div);
        r.register("flt.neg", prim_flt_neg);
        // ADR-040 §2.4 (M-899, `enb` Gap A): the scalar-float comparison group —
        // `flt.lt`/`flt.le`/`flt.gt`/`flt.ge`/`flt.eq` (the IEEE-754 §5.11 partial-order
        // predicates; NaN is explicitly unordered — any NaN operand yields the *defined*
        // predicate value `false`, `flt.eq(NaN, NaN) = false`) plus the **named, opt-in total
        // order** `flt.total_le` (IEEE-754 §5.10 `totalOrder`). Per-op tag `Empirical`; the
        // total-order *property* stays `Empirical` until the M-511 proof debt is discharged —
        // see the float comparison section note below.
        r.register("flt.lt", prim_flt_lt);
        r.register("flt.le", prim_flt_le);
        r.register("flt.gt", prim_flt_gt);
        r.register("flt.ge", prim_flt_ge);
        r.register("flt.eq", prim_flt_eq);
        r.register("flt.total_le", prim_flt_total_le);
        // ADR-040 §2.5 (CU-2): the mandated float classification predicates ("is_nan/is_finite at
        // minimum") — the in-band never-silent tests for the propagating ±inf/NaN sentinels (§2.4).
        r.register("flt.is_nan", prim_flt_is_nan);
        r.register("flt.is_finite", prim_flt_is_finite);
        r.register("flt.is_infinite", prim_flt_is_infinite);
        // ADR-040 §2.4 (CU-3): never-silent Binary↔Float conversions — the "target-width prim"
        // shape of `bit.width_cast` (DN-41). `bin.to_flt` is checked-exact; `flt.to_bin` reads its
        // target width off a witness operand, exactly as `bit.width_cast` does.
        r.register("bin.to_flt", prim_bin_to_flt);
        r.register("flt.to_bin", prim_flt_to_bin);
        // RFC-0003 §3/§4 / ADR-008 (M-892, `enb` Gap C): the VSA bind group — model-dispatched
        // (MAP-I/FHRR/BSC) on the operand's `Repr::Vsa` model id. The kernel (`mycelium-vsa`)
        // constructs the result `Value` with its honest per-model tag; the wrappers carry it
        // through unchanged (VR-5). See the module note at the vsa section below.
        r.register("vsa.bind", prim_vsa_bind);
        r.register("vsa.unbind", prim_vsa_unbind);
        r.register("vsa.permute", prim_vsa_permute);
        // RFC-0003 §4/§5 / ADR-008 (M-893, `enb` Gap C): `vsa.bundle` — superposition via the
        // **certified path** (`MapI::bundle_values_certified`, the M-131 checked-instantiation
        // pattern). The kernel constructs the result `Value` with its `Proven` tag + checked
        // `CapacityBound`; the wrapper carries it through unchanged (VR-5). Dispatch set for
        // bundle: the certified singleton {MAP-I} — FHRR/BSC (whose kernel bundles are
        // `Empirical`-profile ops) are explicit refusals, never a silent re-tag. See the module
        // note at the vsa section below.
        r.register("vsa.bundle", prim_vsa_bundle);
        // RFC-0003 §3/§5/§6 / ADR-008 (M-894, `enb` Gap C): the cleanup/reconstruction pair +
        // the capacity-bound query. `vsa.cleanup` snaps a (possibly noisy) hypervector to the
        // nearest codebook atom and returns the `[index, confidence, margin]` decision triple —
        // never a silent nearest-neighbour pick (FR-S4/G2); `vsa.reconstruct` is the §6
        // compositional role-reconstruction (unbind → cleanup → an explicit caller threshold);
        // `vsa.required_dim` surfaces the M-131 `requiredDim`/`proven_capacity_bound` checked
        // instantiation, its result carrying the kernel's `Proven` `CapacityBound`. See the
        // M-894 section note below for the tag/carry rules.
        r.register("vsa.cleanup", prim_vsa_cleanup);
        r.register("vsa.reconstruct", prim_vsa_reconstruct);
        r.register("vsa.required_dim", prim_vsa_required_dim);
        r
    }

    /// Register (or replace) a primitive.
    pub fn register(&mut self, name: &str, f: PrimFn) {
        self.table.insert(name.to_owned(), f);
    }

    /// Register a **host / FFI** op under the `wild:` namespace (RFC-0028 §4.3).
    ///
    /// `name` is the host-call form without the `wild:` prefix (e.g. `"fs_read"` →
    /// prim key `wild:fs_read`). The default [`PrimRegistry::with_builtins`] grants
    /// **no** host ops — the `@std-sys` host (via `mycelium-std-sys-host` + CLI) must
    /// call this to populate the table. Empty-by-design until install (never silent G2).
    ///
    /// See [`crate::host`] for the install contract (spike resolution 2026-08-01).
    pub fn register_host(&mut self, name: &str, f: PrimFn) {
        let key = if name.starts_with("wild:") {
            name.to_owned()
        } else {
            format!("wild:{name}")
        };
        self.table.insert(key, f);
    }

    /// True if a `wild:{name}` (or fully-qualified `wild:…`) host op is registered.
    #[must_use]
    pub fn has_host(&self, name: &str) -> bool {
        let key = if name.starts_with("wild:") {
            name.to_owned()
        } else {
            format!("wild:{name}")
        };
        self.table.contains_key(&key)
    }

    /// Look up a primitive by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<PrimFn> {
        self.table.get(name).copied()
    }

    /// The registered primitive names (sorted).
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.table.keys().map(String::as_str).collect()
    }
}

/// Build a result value with honest provenance/guarantee threading (RFC-0001 §4.6/§4.7). The
/// intrinsic strength of every built-in is `Exact`, so the result strength is the `meet` over the
/// inputs. Exact inputs → an `Exact`/`bound = None` result; an approximate input is composed per the
/// prim's [`ApproxRule`] (M-204) — or explicitly refused when no rule applies (never a fabricated
/// bound; G2).
fn compose_result(
    prim: &str,
    inputs: &[&Value],
    repr: Repr,
    payload: Payload,
    rule: ApproxRule,
) -> Result<Value, EvalError> {
    let strength = GuaranteeStrength::propagate(
        GuaranteeStrength::Exact,
        inputs.iter().map(|v| v.meta().guarantee()),
    );
    let provenance = Provenance::Derived {
        op: operation_hash(prim),
        inputs: inputs.iter().map(|v| v.content_hash()).collect(),
    };
    let (guarantee, bound) = if strength == GuaranteeStrength::Exact {
        // All inputs exact ⇒ exact result, no bound (M-I1).
        (GuaranteeStrength::Exact, None)
    } else {
        compose_approx(prim, inputs, rule)?
    };
    // The `EvalError::Wf` arms below are *defensive*, not reachable from the public API with the
    // built-in prims (A4-04). `compose_approx` only ever pairs a non-`Exact` strength with a
    // present `Bound` (and `Exact` with `None`), so `Meta::new`'s M-I1 coupling check passes; and
    // every built-in produces a `payload` whose length matches the cloned input `repr`, so
    // `Value::new`'s payload↔repr check passes. They stay as explicit errors so that a *future*
    // prim (or a custom `PrimRegistry` registered via `Interpreter::new`) whose output is
    // internally inconsistent refuses honestly rather than panicking (G2 — never silent, never a
    // crash on constructed input).
    let meta = Meta::new(provenance, guarantee, bound, None, None, None).map_err(EvalError::Wf)?;
    Value::new(repr, payload, meta).map_err(EvalError::Wf)
}

/// Compose the bound + strength for a result over at least one *approximate* input (M-204; ADR-010).
/// The honest upgrade over the Phase-1 blanket refusal: a defined rule composes a *checked* bound; an
/// undefined one still refuses rather than guessing.
fn compose_approx(
    prim: &str,
    inputs: &[&Value],
    rule: ApproxRule,
) -> Result<(GuaranteeStrength, Option<Bound>), EvalError> {
    let refuse = || EvalError::ApproxCompositionUnsupported {
        prim: prim.to_owned(),
    };
    match rule {
        ApproxRule::Refuse => Err(refuse()),
        ApproxRule::Passthrough => {
            // Identity preserves the bound exactly (citation included) — clone it through.
            let v = inputs.first().ok_or_else(refuse)?;
            Ok((v.meta().guarantee(), v.meta().bound().cloned()))
        }
        ApproxRule::Error(op) => {
            // The non-exact inputs carry the Error bounds; exact inputs contribute the ε/strength
            // identity, so collecting only the present bounds is exactly the composition input set.
            let bounds: Vec<&Bound> = inputs.iter().filter_map(|v| v.meta().bound()).collect();
            let composed = compose_error_bound(&bounds, op).ok_or_else(refuse)?;
            Ok((composed.strength, Some(composed.bound)))
        }
    }
}

fn expect_arity(prim: &str, args: &[&Value], n: usize) -> Result<(), EvalError> {
    if args.len() == n {
        Ok(())
    } else {
        Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("expected {n} argument(s), got {}", args.len()),
        })
    }
}

fn as_bits<'a>(prim: &str, v: &'a Value) -> Result<&'a [bool], EvalError> {
    match (v.repr(), v.payload()) {
        (Repr::Binary { .. }, Payload::Bits(b)) => Ok(b),
        _ => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "expected a Binary operand".to_owned(),
        }),
    }
}

// --- built-ins ---------------------------------------------------------------------------------

/// `core.id : a → a`. Identity (re-stamps provenance); useful as a no-op and a test fixture.
fn prim_id(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let v = args[0];
    compose_result(
        prim,
        args,
        v.repr().clone(),
        v.payload().clone(),
        ApproxRule::Passthrough,
    )
}

/// `bit.not : Binary{n} → Binary{n}` — elementwise complement.
fn prim_bit_not(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let bits = as_bits(prim, args[0])?;
    let out: Vec<bool> = bits.iter().map(|&b| !b).collect();
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

/// Shared unary bit-manipulation kernel for `bit.popcount`/`bit.clz`/`bit.ctz` (CU-6) — reads one
/// `Binary{N}` operand, delegates to the width-preserving [`mycelium_core::binary`] count codec, and
/// composes the `Binary{N}` count result. Total (no operand refuses — a bit-count always fits `N`
/// bits), width-preserving, signedness-free. `count` selects the codec.
fn bit_count(
    prim: &str,
    args: &[&Value],
    count: fn(&[bool]) -> Vec<bool>,
) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let bits = as_bits(prim, args[0])?;
    let out = count(bits);
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

/// `bit.popcount : (Binary{N}) → Binary{N}` — population count (CU-6; Rust `count_ones`).
fn prim_bit_popcount(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    bit_count(prim, args, binary::popcount)
}

/// `bit.clz : (Binary{N}) → Binary{N}` — count leading zeros; `N` for all-zero (CU-6; Rust
/// `leading_zeros`).
fn prim_bit_clz(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    bit_count(prim, args, binary::clz)
}

/// `bit.ctz : (Binary{N}) → Binary{N}` — count trailing zeros; `N` for all-zero (CU-6; Rust
/// `trailing_zeros`).
fn prim_bit_ctz(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    bit_count(prim, args, binary::ctz)
}

/// Shared elementwise binary-logical kernel for `bit.and/or/xor`.
fn bit_binop(prim: &str, args: &[&Value], op: fn(bool, bool) -> bool) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bits(prim, args[0])?;
    let b = as_bits(prim, args[1])?;
    if a.len() != b.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("width mismatch: {} vs {}", a.len(), b.len()),
        });
    }
    let out: Vec<bool> = a.iter().zip(b).map(|(&x, &y)| op(x, y)).collect();
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

fn prim_bit_and(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    bit_binop(prim, args, |x, y| x & y)
}
fn prim_bit_or(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    bit_binop(prim, args, |x, y| x | y)
}
fn prim_bit_xor(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    bit_binop(prim, args, |x, y| x ^ y)
}

fn as_trits<'a>(prim: &str, v: &'a Value) -> Result<&'a [Trit], EvalError> {
    match (v.repr(), v.payload()) {
        (Repr::Ternary { .. }, Payload::Trits(t)) => Ok(t),
        _ => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "expected a Ternary operand".to_owned(),
        }),
    }
}

/// `trit.neg : Ternary{m} → Ternary{m}` — digit-wise sign flip. Exactly `value(−t) = −value(t)`
/// (balanced ternary has no sign asymmetry; `docs/spec/swaps/binary-ternary.md` §1). Always in range.
fn prim_trit_neg(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let out = ternary::neg(as_trits(prim, args[0])?);
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Trits(out),
        ApproxRule::Error(ErrorOp::Neg),
    )
}

/// Shared kernel for the fixed-width balanced-ternary binary arithmetic prims (`trit.add/sub/mul`).
/// Operands must be equal-width `Ternary`; an out-of-range result is an explicit
/// [`EvalError::Overflow`], never a silent wrap (M-111; `binary-ternary.md` §1).
fn trit_binop(
    prim: &str,
    args: &[&Value],
    op: fn(&[Trit], &[Trit]) -> Option<Vec<Trit>>,
    rule: ApproxRule,
) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_trits(prim, args[0])?;
    let b = as_trits(prim, args[1])?;
    if a.len() != b.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("width mismatch: {} vs {} trits", a.len(), b.len()),
        });
    }
    let out = op(a, b).ok_or_else(|| EvalError::Overflow {
        prim: prim.to_owned(),
    })?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Trits(out),
        rule,
    )
}

/// `trit.add`: balanced-ternary addition is exact on the values, so an approximate input's ε
/// propagates additively (1-Lipschitz; affine `Add`) — sound (M-204).
fn prim_trit_add(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    trit_binop(prim, args, ternary::add, ApproxRule::Error(ErrorOp::Add))
}
/// `trit.sub`: same additive ε propagation as `trit.add` (affine `Sub`).
fn prim_trit_sub(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    trit_binop(prim, args, ternary::sub, ApproxRule::Error(ErrorOp::Sub))
}
/// `trit.mul`: multiplicative ε propagation needs the central operand magnitudes (affine `Mul`); that
/// plumbing lands with the Dense numerics (E2-1), so an approximate input is refused for now — honest,
/// not a fabricated bound (G2).
fn prim_trit_mul(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    trit_binop(prim, args, ternary::mul, ApproxRule::Refuse)
}

// --- RFC-0032 D1 (M-747): reduce-to-`Bool` comparison/equality ---------------------------------
//
// `eq`/`lt` are the two kernel comparison prims (RFC-0032 §5 D1). Each takes two equal-width
// operands of the **same paradigm** (`Binary{N}` or `Ternary{N}`) and reduces to a one-bit truth
// value. **Realization note (engineering call, RFC-0032 Q1):** a kernel prim returns a
// representation [`Value`], never a `.myc` data value, so the `Bool` of D1 bottoms out here as
// **`Binary{1}`** (`0b1` = true, `0b0` = false); the `.myc` `std.cmp` surface lifts that bit into
// the `Bool` ADT (`match eq(a, b) { 0b1 => True, _ => False }`) — a one-line bridge that lands with
// the E13-1 `std.cmp` port (M-718), demonstrated by the M-752 smoke ports. Guarantee **`Exact`**: a
// total decidable relation over a finite domain. Mismatched widths/paradigms are an explicit
// never-silent [`EvalError::PrimType`] — never a silent `false`/`0b0` (G2).

/// Compare two representation values for the comparison prims. Requires equal width and the same
/// paradigm; a mismatch is an explicit error (never a silent ordering). The orderings are the D1
/// total orders: **unsigned magnitude** for `Binary{N}` (MSB-first lexicographic over the bits) and
/// **balanced-integer value** for `Ternary{N}` (MSB-first lexicographic over the signed digits — for
/// fixed-width balanced ternary the most-significant differing digit dominates, so this equals the
/// integer-value order).
fn cmp_repr_operands(prim: &str, a: &Value, b: &Value) -> Result<Ordering, EvalError> {
    match (a.repr(), b.repr()) {
        (Repr::Binary { width: wa }, Repr::Binary { width: wb }) => {
            if wa != wb {
                return Err(EvalError::PrimType {
                    prim: prim.to_owned(),
                    why: format!("width mismatch: Binary{{{wa}}} vs Binary{{{wb}}}"),
                });
            }
            let xa = as_bits(prim, a)?;
            let xb = as_bits(prim, b)?;
            Ok(xa.iter().cmp(xb.iter()))
        }
        (Repr::Ternary { trits: ta }, Repr::Ternary { trits: tb }) => {
            if ta != tb {
                return Err(EvalError::PrimType {
                    prim: prim.to_owned(),
                    why: format!("width mismatch: Ternary{{{ta}}} vs Ternary{{{tb}}}"),
                });
            }
            let xa = as_trits(prim, a)?;
            let xb = as_trits(prim, b)?;
            Ok(xa
                .iter()
                .map(|t| ternary::digit(*t))
                .cmp(xb.iter().map(|t| ternary::digit(*t))))
        }
        // ADR-040 §2.4 (M-899): float comparison is *partial* (NaN has no order), so it never
        // routes through this D1 total order — it has its own explicit prims: the IEEE-754 §5.11
        // predicates `flt.lt`/`flt.le`/`flt.gt`/`flt.ge`/`flt.eq` (NaN unordered → false) and the
        // named opt-in total order `flt.total_le`. Refused explicitly with the real routing,
        // never funneled into the generic same-paradigm message below (and never a
        // silently-wrong bitwise order).
        (Repr::Float { .. }, Repr::Float { .. }) => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "float comparison is partial (NaN is unordered — ADR-040 §2.4) and does not use \
                  the D1 Binary/Ternary total order; use the explicit predicates \
                  flt.lt/flt.le/flt.gt/flt.ge/flt.eq (any NaN operand → false) or the named \
                  total order flt.total_le for sorting/keying (M-899)"
                .to_owned(),
        }),
        _ => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "comparison requires two operands of the same paradigm (both Binary or both \
                  Ternary)"
                .to_owned(),
        }),
    }
}

/// Build the `Binary{1}` truth value for a comparison result (`0b1` = true). Threads provenance/
/// guarantee honestly via [`compose_result`]; comparison has no defined ε-propagation over an
/// approximate input, so an approximate operand is refused (`ApproxRule::Refuse`) rather than
/// fabricating a bound (G2/VR-5).
fn bool_result(prim: &str, inputs: &[&Value], truth: bool) -> Result<Value, EvalError> {
    compose_result(
        prim,
        inputs,
        Repr::Binary { width: 1 },
        Payload::Bits(vec![truth]),
        ApproxRule::Refuse,
    )
}

/// `cmp.eq : (T{N}, T{N}) → Binary{1}` — structural width-typed equality (RFC-0032 D1).
fn prim_cmp_eq(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let truth = cmp_repr_operands(prim, args[0], args[1])? == Ordering::Equal;
    bool_result(prim, args, truth)
}

/// `cmp.lt : (T{N}, T{N}) → Binary{1}` — the D1 total order (`a < b`).
fn prim_cmp_lt(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let truth = cmp_repr_operands(prim, args[0], args[1])? == Ordering::Less;
    bool_result(prim, args, truth)
}

// --- RFC-0032 D2 (M-748): never-silent fixed-width binary arithmetic ---------------------------
//
// `bit.add`/`bit.sub` are unsigned fixed-width ripple-carry add/subtract over `Binary{N}` (bits
// MSB-first), exactly mirroring the `trit.*` in-range contract: a result outside `[0, 2^N)` is an
// explicit [`EvalError::Overflow`], **never** a silent wrap (RFC-0032 §5 D2; G2). Guarantee
// **`Exact`** on the in-range result. (A wrapping/modular `add` is intentionally absent — it would
// be a separate, *declared* op, never this never-silent one.)

/// Shared kernel for the never-silent binary arithmetic prims. `subtract == false` is ripple-carry
/// addition (carry-out ⇒ overflow); `subtract == true` is ripple-borrow subtraction (borrow-out ⇒
/// underflow, i.e. a negative result with no unsigned representation). Operands must be equal-width
/// `Binary`; a width mismatch is an explicit [`EvalError::PrimType`].
fn bin_arith(prim: &str, args: &[&Value], subtract: bool) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bits(prim, args[0])?;
    let b = as_bits(prim, args[1])?;
    if a.len() != b.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("width mismatch: {} vs {} bits", a.len(), b.len()),
        });
    }
    let n = a.len();
    let mut out = vec![false; n];
    if subtract {
        let mut borrow = 0i8;
        for i in (0..n).rev() {
            let d = i8::from(a[i]) - i8::from(b[i]) - borrow;
            if d < 0 {
                out[i] = (d + 2) == 1;
                borrow = 1;
            } else {
                out[i] = d == 1;
                borrow = 0;
            }
        }
        if borrow != 0 {
            return Err(EvalError::Overflow {
                prim: prim.to_owned(),
            });
        }
    } else {
        let mut carry = 0u8;
        for i in (0..n).rev() {
            let s = u8::from(a[i]) + u8::from(b[i]) + carry;
            out[i] = (s & 1) == 1;
            carry = s >> 1;
        }
        if carry != 0 {
            return Err(EvalError::Overflow {
                prim: prim.to_owned(),
            });
        }
    }
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

/// `bit.add : (Binary{N}, Binary{N}) → Binary{N}` — never-silent unsigned addition (RFC-0032 D2).
fn prim_bit_add(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    bin_arith(prim, args, false)
}

/// `bit.sub : (Binary{N}, Binary{N}) → Binary{N}` — never-silent unsigned subtraction (RFC-0032 D2).
fn prim_bit_sub(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    bin_arith(prim, args, true)
}

// --- RFC-0033 §4.1.2/§4.1.3 (M-887, `enb` Gap B): never-silent two's-complement multiply --------
//
// `bin.mul` is the first Gap-B prim of the RFC-0033 shared two's-complement arithmetic set
// (ADR-028: signedness lives in the *op*, not the `Repr` — `add`/`sub`/`mul`/`neg` are bit-identical
// across the signed/unsigned reading of the operands; only division, comparison, shift, and
// overflow *detection* are signedness-split). This prim reads its equal-width `Binary{N}` operands
// under the two's-complement (**signed**) interpretation — "two's-complement multiply" per the
// M-887 task naming — distinct from `bit.add`/`bit.sub`'s existing **unsigned** overflow contract
// (RFC-0032 D2). The kernel codec lives in [`mycelium_core::binary`] (the M-120 two's-complement
// home, shared with the binary↔ternary swap), mirroring how `trit_binop` delegates to
// [`mycelium_core::ternary`]. Never-silent: an out-of-range product is an explicit
// [`EvalError::Overflow`], never a wrap (RFC-0033 §4.1.3; G2).
//
// **Registry pattern for the rest of Gap B/C (M-766/M-767/M-888/M-889/…):** kernel codec in
// `mycelium-core` (arithmetic + the never-silent bound, `Option<Vec<bool>>`) → a thin prim wrapper
// here that checks arity/width, calls the codec, and maps `None` to `EvalError::Overflow` →
// registered under a `bin.*`/`bit.*`/`trit.*`-namespaced kernel name → surfaced in
// `mycelium-l1/src/checkty.rs` (`prim_family`/`prim_sig`/`prim_kernel_name`) under a distinct
// surface name → pinned in `mycelium-core::PrimTable::builtins()` (the content-addressed `Π`,
// DN-10 §3.4 equivalence) and `mycelium-l1/tests/prim_table.rs`'s `surface_cases()`.
//
// **Width cap (current scope).** [`mycelium_core::binary::mul`] is exact for `n ≤
// `[`mycelium_core::binary::MUL_MAX_WIDTH`]` (an `i128` intermediate product — the same cap
// `bits_to_int`/`int_to_bits` already declare); a wider operand refuses with an explicit
// [`EvalError::PrimType`] naming the cap, never a silent truncation. Arbitrary-width `Binary`
// multiply (matching `bit.add`/`bit.sub`'s width-unbounded ripple-carry) is out of scope for
// M-887 — FLAGged for the Gap-B follow-ons that reconcile the full shared op set (M-766/M-767).

/// `bin.mul : (Binary{N}, Binary{N}) → Binary{N}` — never-silent two's-complement multiply
/// (RFC-0033 §4.1.2/§4.1.3, M-887). Equal-width operands, `N ≤
/// `[`mycelium_core::binary::MUL_MAX_WIDTH`]` (see the module note above); a width mismatch or an
/// over-cap width is an explicit [`EvalError::PrimType`], and an out-of-range product is an
/// explicit [`EvalError::Overflow`] — never a silent wrap (G2).
fn prim_bin_mul(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bits(prim, args[0])?;
    let b = as_bits(prim, args[1])?;
    if a.len() != b.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("width mismatch: {} vs {} bits", a.len(), b.len()),
        });
    }
    if a.len() > binary::MUL_MAX_WIDTH {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "width {} exceeds the {}-bit two's-complement multiply cap (M-887 scope)",
                a.len(),
                binary::MUL_MAX_WIDTH
            ),
        });
    }
    let out = binary::mul(a, b).ok_or_else(|| EvalError::Overflow {
        prim: prim.to_owned(),
    })?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

/// `bit.mul : (Binary{N}, Binary{N}) → Binary{N}` — never-silent **unsigned** fixed-width multiply
/// (CU-1; RFC-0033 §4.1.2 — overflow detection is signedness-distinct ⇒ a distinct named op from the
/// signed `bin.mul`; `lib/std/math.myc` FLAG-math-1). The unsigned counterpart of the `bit.add`/
/// `bit.sub`/`bin.div`/`bin.rem` unsigned family: operands read as unsigned bitvectors, an out-of-
/// `U_N` product an explicit [`EvalError::Overflow`], a width mismatch or over-`MUL_MAX_WIDTH` width an
/// explicit [`EvalError::PrimType`] — never a silent wrap or truncation (G2). Delegates to the shared
/// [`mycelium_core::binary::mul_unsigned`] codec.
fn prim_bit_mul(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bits(prim, args[0])?;
    let b = as_bits(prim, args[1])?;
    if a.len() != b.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("width mismatch: {} vs {} bits", a.len(), b.len()),
        });
    }
    if a.len() > binary::MUL_MAX_WIDTH {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "width {} exceeds the {}-bit multiply cap (CU-1 scope)",
                a.len(),
                binary::MUL_MAX_WIDTH
            ),
        });
    }
    let out = binary::mul_unsigned(a, b).ok_or_else(|| EvalError::Overflow {
        prim: prim.to_owned(),
    })?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

// --- RFC-0033 §4.1.2/§4.1.3 (M-888, `enb` Gap B): never-silent unsigned division/remainder -------
//
// `bin.div`/`bin.rem` are the second Gap-B prims of the RFC-0033 arithmetic set. Division *differs*
// by signedness (§4.1.2: "operations whose result differs by signedness … MUST be distinct named
// ops"), unlike `bin.mul` (shared/signedness-agnostic bits). This lands the **unsigned** reading
// first — the signed/two's-complement reading rides M-767 under its own distinct name. The kernel
// codec ([`binary::div_rem`]) reads operands as **unsigned** bitvectors ([`binary::bits_to_uint`]),
// never through the two's-complement [`binary::bits_to_int`] `bin.mul` uses.
//
// **Naming FLAG (maintainer call).** The M-888 task text names these prims `bin.div`/`bin.rem`,
// mirroring `bin.mul`'s `bin.*` kernel namespace — but M-887 established `bin.*` for the *signed*
// two's-complement reading (`bin.mul`), while `bit.*` is the existing *unsigned* arithmetic
// namespace (`bit.add`/`bit.sub`, RFC-0032 D2). Naming an **unsigned** division `bin.div` sits
// somewhat against that emerging convention (`bit.div` would read more consistently with
// `bit.add`/`bit.sub`). Landed as `bin.div`/`bin.rem` per the literal task/issue naming; flagged
// here — and in the leaf report — for a maintainer decision on whether M-767's future *signed*
// `div`/`rem` should instead claim `bin.div`/`bin.rem` and this unsigned pair be renamed
// `bit.div`/`bit.rem` for consistency (a rename is cheap now, before any downstream `.myc` surface
// depends on the name; RFC-0033 §4.1.2 requires the distinct-naming property, not a specific
// spelling).
//
// Never-silent: division by zero is an explicit [`EvalError::PrimType`] (there is no "overflow"
// case for unsigned fixed-width division — see [`binary::div_rem`]'s doc comment), never a panic or
// a silently-defined value (RFC-0033 §4.1.3; G2).
//
// **Width cap (current scope).** Mirrors `bin.mul`: [`binary::div_rem`] is exact for `n ≤
// `[`binary::DIV_MAX_WIDTH`]`; a wider operand refuses with an explicit [`EvalError::PrimType`]
// naming the cap.

/// Shared arity/width validation + kernel dispatch for `bin.div`/`bin.rem`: checks arity 2,
/// extracts equal-width `Binary` operand bits (width mismatch/over-cap → [`EvalError::PrimType`]),
/// and calls [`binary::div_rem`] (div-by-zero → an explicit [`EvalError::PrimType`], never a
/// panic). Returns `(quotient_bits, remainder_bits)` so `bin.div`/`bin.rem` share exactly one
/// division per call rather than each recomputing it.
fn bin_div_rem(prim: &str, args: &[&Value]) -> Result<(Vec<bool>, Vec<bool>), EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bits(prim, args[0])?;
    let b = as_bits(prim, args[1])?;
    if a.len() != b.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("width mismatch: {} vs {} bits", a.len(), b.len()),
        });
    }
    if a.len() > binary::DIV_MAX_WIDTH {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "width {} exceeds the {}-bit unsigned division cap (M-888 scope)",
                a.len(),
                binary::DIV_MAX_WIDTH
            ),
        });
    }
    binary::div_rem(a, b).ok_or_else(|| EvalError::PrimType {
        prim: prim.to_owned(),
        why: "division by zero".to_owned(),
    })
}

/// `bin.div : (Binary{N}, Binary{N}) → Binary{N}` — never-silent **unsigned** division
/// (RFC-0033 §4.1.2/§4.1.3, M-888). Equal-width operands, `N ≤ `[`binary::DIV_MAX_WIDTH`]`; a width
/// mismatch, an over-cap width, or division by zero is an explicit [`EvalError::PrimType`] — never
/// a panic or a silently-defined value (G2).
fn prim_bin_div(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (q, _r) = bin_div_rem(prim, args)?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(q),
        ApproxRule::Refuse,
    )
}

/// `bin.rem : (Binary{N}, Binary{N}) → Binary{N}` — never-silent **unsigned** remainder
/// (RFC-0033 §4.1.2/§4.1.3, M-888). Same never-silent contract as [`prim_bin_div`]; together they
/// satisfy the Euclidean identity `a == (a/b)*b + (a%b)` for `b ≠ 0` (property-tested).
fn prim_bin_rem(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (_q, r) = bin_div_rem(prim, args)?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(r),
        ApproxRule::Refuse,
    )
}

// --- RFC-0033 §4.1.2/§4.1.3 (M-889, `enb` Gap B): never-silent logical shift ---------------------
//
// `bin.shl`/`bin.shr` are the third Gap-B prim pair of the RFC-0033 arithmetic set — the
// signedness-split `shift` op set (§4.1.2). This lands the **logical** (unsigned) reading first —
// bits shifted off either end are dropped, zero bits are shifted in, never a wrap/rotate — mirroring
// how [`bin_div_rem`] lands the unsigned division reading first. The **arithmetic** (sign-extending)
// right shift is the distinct signed op M-767 lands under its own name (§4.1.2's signedness-split
// requirement applies to `shift` exactly as it does to `div`).
//
// Both operands are `Binary{N}`: the value and the shift amount (itself read as an unsigned `N`-bit
// bitvector via [`binary::bits_to_uint`], never through the two's-complement
// [`binary::bits_to_int`]). A shift amount `>= N` is an explicit [`EvalError::PrimType`] refusal —
// never UB, a silently wrapped/modulo shift amount, or a silently-zeroed result (RFC-0033 §4.1.3;
// G2) — the same refusal *shape* as [`bin_div_rem`]'s div-by-zero (a value-precondition violation
// on an operand, not an out-of-range *result*, so `PrimType` rather than `Overflow`).
//
// **Width cap (current scope).** Mirrors `bin.mul`/`bin.div`: [`binary::shl`]/[`binary::shr`] are
// exact for `n ≤ `[`binary::SHIFT_MAX_WIDTH`]`; a wider operand refuses with an explicit
// [`EvalError::PrimType`] naming the cap.

/// Shared arity/width validation + kernel dispatch for `bin.shl`/`bin.shr` (M-889) and the
/// arithmetic `bin.shr_s` (M-767 — same operand shape and refusal contract, sign-extending
/// kernel): checks arity 2, extracts equal-width `Binary` operand bits (width mismatch/over-cap →
/// [`EvalError::PrimType`]), and calls `op` (`None` — a shift amount `>= N` — → an explicit
/// [`EvalError::PrimType`], never a panic).
fn bin_shift(
    prim: &str,
    args: &[&Value],
    op: fn(&[bool], &[bool]) -> Option<Vec<bool>>,
) -> Result<Vec<bool>, EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bits(prim, args[0])?;
    let shift = as_bits(prim, args[1])?;
    if a.len() != shift.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("width mismatch: {} vs {} bits", a.len(), shift.len()),
        });
    }
    if a.len() > binary::SHIFT_MAX_WIDTH {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "width {} exceeds the {}-bit shift cap (M-889/M-767 scope)",
                a.len(),
                binary::SHIFT_MAX_WIDTH
            ),
        });
    }
    op(a, shift).ok_or_else(|| EvalError::PrimType {
        prim: prim.to_owned(),
        why: format!("shift amount >= width ({} bits)", a.len()),
    })
}

/// `bin.shl : (Binary{N}, Binary{N}) → Binary{N}` — never-silent **logical** left shift
/// (RFC-0033 §4.1.2/§4.1.3, M-889). Equal-width operands, `N ≤ `[`binary::SHIFT_MAX_WIDTH`]`; a
/// width mismatch, an over-cap width, or a shift amount `>= N` is an explicit
/// [`EvalError::PrimType`] — never UB or a silent wrap/rotate (G2).
fn prim_bin_shl(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let out = bin_shift(prim, args, binary::shl)?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

/// `bin.shr : (Binary{N}, Binary{N}) → Binary{N}` — never-silent **logical** (zero-filling) right
/// shift (RFC-0033 §4.1.2/§4.1.3, M-889). Same never-silent contract as [`prim_bin_shl`]; the
/// **arithmetic**/sign-extending right shift is the distinct signed op M-767 lands separately.
fn prim_bin_shr(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let out = bin_shift(prim, args, binary::shr)?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

// --- RFC-0033 §4.1.2/§4.1.3 (M-766, `enb` Gap B): never-silent two's-complement add/sub/neg --------
//
// `bin.add`/`bin.sub`/`bin.neg` complete the *shared* two's-complement arithmetic set `bin.mul`
// (M-887) started (ADR-028: `add`/`sub`/`mul`/`neg` are bit-identical across the signed/unsigned
// reading of the operands, so they MAY be a single named op each). They read equal-width `Binary{N}`
// operands under the two's-complement (**signed**) interpretation, exactly mirroring `bin.mul`.
//
// **Inventory (verified against the registry before landing these, per the M-766 task text —
// "reconcile against the kpr-landed add/sub").** The pre-existing `bit.add`/`bit.sub` (kpr/E19-1,
// RFC-0032 D2, registered above) are a **different, unsigned-committed** family: their overflow
// criterion is the unsigned carry/borrow-out, which *under-refuses* relative to the signed range
// `B_n` — e.g. at `Binary{4}`, `5 + 3 = 8` is unsigned-in-range `[0,15]` but signed-out-of-range
// `B_4 = [-8,7]`, so `bit.add` would accept a sum a signed/two's-complement caller must not silently
// receive. They therefore do **not** satisfy the RFC-0033 §4.1.2 shared two's-complement `add`/`sub`
// this task names, and `bin.add`/`bin.sub` are genuinely missing (not a re-land of E19-1's work).
// `bin.neg` has no pre-existing counterpart to reconcile against (negation is inherently a signed
// concept) — it is unambiguously the shared set's missing fourth member.
//
// Never-silent: an out-of-range sum/difference/negation is an explicit [`EvalError::Overflow`],
// never a wrap (RFC-0033 §4.1.3; G2) — the same posture as `bin.mul`.
//
// **Width cap (current scope).** Mirrors `bin.mul`: [`binary::add`]/[`binary::sub`]/[`binary::neg`]
// are exact for `n ≤ `[`binary::TC_MAX_WIDTH`]`; a wider operand refuses with an explicit
// [`EvalError::PrimType`] naming the cap.

/// Shared arity/width validation + kernel dispatch for the two's-complement `bin.add`/`bin.sub`
/// prims (M-766): checks arity 2, extracts equal-width `Binary` operand bits (width mismatch/
/// over-cap → [`EvalError::PrimType`]), and calls `op` (`None` — the exact result does not fit
/// `B_n` — → an explicit [`EvalError::Overflow`], never a silent wrap).
fn bin_add_sub(
    prim: &str,
    args: &[&Value],
    op: fn(&[bool], &[bool]) -> Option<Vec<bool>>,
) -> Result<Vec<bool>, EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bits(prim, args[0])?;
    let b = as_bits(prim, args[1])?;
    if a.len() != b.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("width mismatch: {} vs {} bits", a.len(), b.len()),
        });
    }
    if a.len() > binary::TC_MAX_WIDTH {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "width {} exceeds the {}-bit two's-complement arithmetic cap (M-766 scope)",
                a.len(),
                binary::TC_MAX_WIDTH
            ),
        });
    }
    op(a, b).ok_or_else(|| EvalError::Overflow {
        prim: prim.to_owned(),
    })
}

/// `bin.add : (Binary{N}, Binary{N}) → Binary{N}` — never-silent two's-complement add (RFC-0033
/// §4.1.2/§4.1.3, M-766). Equal-width operands, `N ≤ `[`binary::TC_MAX_WIDTH`]`; a width mismatch or
/// an over-cap width is an explicit [`EvalError::PrimType`], and an out-of-range sum is an explicit
/// [`EvalError::Overflow`] — never a silent wrap (G2). Distinct from `bit.add` (RFC-0032 D2), whose
/// unsigned carry-out overflow criterion under-refuses relative to the signed domain `B_N` (see the
/// module-level inventory note above).
fn prim_bin_add(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let out = bin_add_sub(prim, args, binary::add)?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

/// `bin.sub : (Binary{N}, Binary{N}) → Binary{N}` — never-silent two's-complement subtract
/// (RFC-0033 §4.1.2/§4.1.3, M-766). Same never-silent contract as [`prim_bin_add`]; distinct from
/// `bit.sub`'s unsigned borrow-out overflow criterion for the same reason.
fn prim_bin_sub(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let out = bin_add_sub(prim, args, binary::sub)?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

/// `bin.neg : Binary{N} → Binary{N}` — never-silent two's-complement negate (RFC-0033
/// §4.1.2/§4.1.3, M-766): the shared op set's genuinely-missing member (unlike `add`/`sub`/`mul`,
/// there is no pre-existing unsigned "negate" to reconcile against). An over-cap width is an
/// explicit [`EvalError::PrimType`]; negating `B_N`'s minimum value `-2^(N-1)` (which has no positive
/// two's-complement counterpart in `B_N`) is an explicit [`EvalError::Overflow`] — never a silent
/// wrap (G2), the classic two's-complement negate-overflow case.
fn prim_bin_neg(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let a = as_bits(prim, args[0])?;
    if a.len() > binary::TC_MAX_WIDTH {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "width {} exceeds the {}-bit two's-complement arithmetic cap (M-766 scope)",
                a.len(),
                binary::TC_MAX_WIDTH
            ),
        });
    }
    let out = binary::neg(a).ok_or_else(|| EvalError::Overflow {
        prim: prim.to_owned(),
    })?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

// --- RFC-0034 §10 (CU-5): the executable `wrapping` construct — eval-mode dispatch --------------
//
// M-791 landed the named, explicit Axis-B `wrapping` opt-out at the representation layer
// (`Meta::with_wrapping`/`WrappingOpt`, `mycelium-core`) but explicitly flagged the **op-layer
// wiring** — the evaluation path that actually *honors* the marker by electing modular wraparound
// instead of the non-wrapping `Option`/`Result`/`SwapError` refusal — as a downstream task. This
// closes that gap for `bin.add`/`bin.sub`/`bin.mul` (RFC-0034 §10). **No new `wrapping_*` prim
// names are added** — `wrapping` is a *mode* dispatched here over the existing three ops, never a
// distinct entry in the `Π` table (per the CU-5 task ruling, DN-34 §8.16).
//
// **Surface-syntax FLAG (CU-5 leaf report).** `mycelium-l1`'s `elab`/`checkty` have no parser/AST/
// elaborator support for a `wrapping { … }` surface construct yet (grepped before landing this —
// only the M-791 representation-layer marker exists anywhere in the crate). [`eval_wrapping`] is
// the runtime half a future surface construct will dispatch through once that parser/elaborator
// work lands; until then it is reachable directly (this function, and any test/tool that calls it)
// rather than via `.myc` source — never faked as "wired end-to-end" when only the eval half is.

/// Evaluate `prim` (one of `bin.add`/`bin.sub`/`bin.mul`) under the **named** `wrapping` opt-out
/// (RFC-0034 §10): the result wraps modulo `2^n` into `B_n` instead of refusing out-of-range —
/// mirroring [`prim_bin_add`]/[`prim_bin_sub`]/[`prim_bin_mul`]'s operand contract (equal-width,
/// the same per-op width cap) but never their range refusal. Tagged **`Declared`** (a
/// zero-magnitude `UserDeclared` bound, M-I4 — the explicit, developer-opted-into semantics; never
/// a fabricated `Proven`/`Empirical` claim, VR-5) and carrying the [`WrappingOpt`] marker so the
/// opt-out stays inspectable on the result ([`Meta::wrapping_opt`]). Any other `prim` name, or a
/// structural mismatch (unequal widths, an over-cap width), is an explicit
/// [`EvalError::PrimType`] — never a silent fallback to the non-wrapping refusal.
pub fn eval_wrapping(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bits(prim, args[0])?;
    let b = as_bits(prim, args[1])?;
    if a.len() != b.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("width mismatch: {} vs {} bits", a.len(), b.len()),
        });
    }
    /// A wrapping-mode binary bitvector op (`wrapping_add`/`wrapping_sub`/`wrapping_mul`'s shared
    /// shape) — named to keep `eval_wrapping`'s dispatch tuple within clippy's complexity budget.
    type WrappingOp = fn(&[bool], &[bool]) -> Option<Vec<bool>>;
    let (op, cap): (WrappingOp, usize) = match prim {
        "bin.add" => (binary::wrapping_add, binary::TC_MAX_WIDTH),
        "bin.sub" => (binary::wrapping_sub, binary::TC_MAX_WIDTH),
        "bin.mul" => (binary::wrapping_mul, binary::MUL_MAX_WIDTH),
        _ => {
            return Err(EvalError::PrimType {
                prim: prim.to_owned(),
                why: "eval_wrapping only supports the wrapping mode over bin.add/bin.sub/bin.mul \
                      (RFC-0034 §10 — no new wrapping_* prims)"
                    .to_owned(),
            });
        }
    };
    if a.len() > cap {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "width {} exceeds the {}-bit wrapping arithmetic cap",
                a.len(),
                cap
            ),
        });
    }
    // `op` never refuses on range — only ever on the width checks already validated above.
    let out = op(a, b).expect("width already validated; wrapping ops never refuse on range");
    wrapping_result(prim, args, args[0].repr().clone(), Payload::Bits(out))
}

/// Build a `wrapping`-tagged result (RFC-0034 §10; CU-5): `Declared` guarantee with a
/// zero-magnitude `UserDeclared` bound (M-I4) — the explicit, developer-opted-into semantics,
/// never a fabricated `Proven`/`Empirical` claim (VR-5) — and the [`WrappingOpt`] marker attached
/// so the opt-out stays inspectable ([`Meta::wrapping_opt`]).
fn wrapping_result(
    prim: &str,
    inputs: &[&Value],
    repr: Repr,
    payload: Payload,
) -> Result<Value, EvalError> {
    let provenance = Provenance::Derived {
        op: operation_hash(prim),
        inputs: inputs.iter().map(|v| v.content_hash()).collect(),
    };
    let bound = Bound {
        kind: BoundKind::Error {
            eps: 0.0,
            norm: NormKind::Linf,
        },
        basis: BoundBasis::UserDeclared,
    };
    let meta = Meta::new(
        provenance,
        GuaranteeStrength::Declared,
        Some(bound),
        None,
        None,
        None,
    )
    .map_err(EvalError::Wf)?
    .with_wrapping(WrappingOpt::new());
    Value::new(repr, payload, meta).map_err(EvalError::Wf)
}

// --- RFC-0033 §4.1.2/§4.1.3 (M-767, `enb` Gap B): the signedness-split signed op set -------------
//
// `bin.div_s`/`bin.rem_s`/`bin.shr_s`/`cmp.lt_s` are the **signed** counterparts to the landed
// unsigned `bin.div`/`bin.rem`/`bin.shr` and the D1 unsigned-magnitude `cmp.lt` — the operations
// §4.1.2 requires as *distinct named ops* because their result differs by signedness (ADR-028:
// signedness lives in the *op*, not the `Repr`; its Consequences pin the SMT-LIB split —
// `bvsdiv`/`bvudiv`, `bvslt`/`bvult`, `bvashr`/`bvlshr`). The kernel codecs live in
// [`mycelium_core::binary`] (`div_signed`/`rem_signed`/`shr_signed`/`cmp_signed`), reading
// operands through the two's-complement [`binary::bits_to_int`] codec `bin.mul`/`bin.add` already
// use — never the unsigned [`binary::bits_to_uint`] their `_u` twins read.
//
// **Rounding convention (grounding — see the `mycelium_core::binary` module note):** division is
// **truncated toward zero**, the remainder's sign follows the dividend (`a == q·b + r`, `|r| <
// |b|`) — the ADR-028-cited SMT-LIB `bvsdiv`/`bvsrem` semantics (also Rust `/`/`%`). RFC-0033's
// text does not literally pin the rounding; the choice is grounded on the ADR-028 citation and
// FLAGged in the M-767 report, never silently made (VR-5/G2).
//
// **Overflow-detect (§4.1.3), the signedness-split part:** unsigned fixed-width division never
// overflows, but signed division has exactly one overflow — `B_N`'s minimum ÷ −1 (true quotient
// `+2^(N-1)`, out of `B_N`) — which is an explicit [`EvalError::Overflow`], never SMT-LIB's
// defined wrap (the never-silent rule outranks it) and never a panic. `rem_s(min, −1) = 0` fits
// `B_N` exactly and succeeds (see [`binary::rem_signed`] — deliberately not Rust `checked_rem`'s
// hardware-motivated over-refusal). Div-by-zero and a shift amount `>= N` remain value-
// precondition [`EvalError::PrimType`] refusals, mirroring the unsigned pair (`Overflow` is
// reserved for an out-of-range *result*).
//
// **Width cap (current scope).** Mirrors the unsigned pair: [`binary::div_signed`]/
// [`binary::rem_signed`] share [`binary::DIV_MAX_WIDTH`], [`binary::shr_signed`] shares
// [`binary::SHIFT_MAX_WIDTH`]; a wider operand refuses with an explicit [`EvalError::PrimType`]
// naming the cap. `cmp.lt_s` is **width-unbounded** (the kernel order is purely structural — the
// sign-bit-flip lexicographic compare), exactly as the D1 `cmp.lt` unsigned order is.

/// Shared arity/width/divisor validation for `bin.div_s`/`bin.rem_s`: checks arity 2, extracts
/// equal-width `Binary` operand bits (width mismatch/over-cap → [`EvalError::PrimType`]), and
/// refuses an all-zero divisor (div-by-zero → [`EvalError::PrimType`], covering `N == 0`, whose
/// only representable divisor is zero) — so the kernel's remaining `None` (quotient overflow,
/// `div_s` only) maps unambiguously to [`EvalError::Overflow`] at the call sites (G2: the refusal
/// reasons stay honestly distinct).
fn bin_div_rem_s_operands<'a>(
    prim: &str,
    args: &'a [&Value],
) -> Result<(&'a [bool], &'a [bool]), EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bits(prim, args[0])?;
    let b = as_bits(prim, args[1])?;
    if a.len() != b.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("width mismatch: {} vs {} bits", a.len(), b.len()),
        });
    }
    if a.len() > binary::DIV_MAX_WIDTH {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "width {} exceeds the {}-bit signed division cap (M-767 scope)",
                a.len(),
                binary::DIV_MAX_WIDTH
            ),
        });
    }
    if b.iter().all(|&bit| !bit) {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "division by zero".to_owned(),
        });
    }
    Ok((a, b))
}

/// `bin.div_s : (Binary{N}, Binary{N}) → Binary{N}` — never-silent **signed** (two's-complement)
/// division, truncated toward zero (RFC-0033 §4.1.2/§4.1.3, ADR-028, M-767; SMT-LIB `bvsdiv` —
/// see the section note). Equal-width operands, `N ≤ `[`binary::DIV_MAX_WIDTH`]`; a width
/// mismatch, an over-cap width, or division by zero is an explicit [`EvalError::PrimType`], and
/// the single signed-division overflow — `B_N`'s minimum ÷ −1 — is an explicit
/// [`EvalError::Overflow`], never a silent wrap (G2).
fn prim_bin_div_s(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = bin_div_rem_s_operands(prim, args)?;
    // After the operand checks the kernel's only remaining refusal is the min ÷ −1 overflow.
    let out = binary::div_signed(a, b).ok_or_else(|| EvalError::Overflow {
        prim: prim.to_owned(),
    })?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

/// `bin.rem_s : (Binary{N}, Binary{N}) → Binary{N}` — never-silent **signed** remainder, sign
/// following the dividend (RFC-0033 §4.1.2/§4.1.3, ADR-028, M-767; SMT-LIB `bvsrem`). Same
/// operand contract as [`prim_bin_div_s`]; together they satisfy the truncated-division identity
/// `a == div_s(a,b)·b + rem_s(a,b)` with `|r| < |b|` (property-tested in `mycelium-core`). The
/// remainder itself **never overflows** (`|r| < |b| ≤ 2^(N-1)` keeps it inside `B_N`), so
/// `rem_s(min, −1) = 0` succeeds where `div_s(min, −1)` refuses.
fn prim_bin_rem_s(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = bin_div_rem_s_operands(prim, args)?;
    // Defensive: after the operand checks `rem_signed` is total (the remainder always fits B_N —
    // see its doc comment); this arm is unreachable rather than a real refusal path.
    let out = binary::rem_signed(a, b).ok_or_else(|| EvalError::Overflow {
        prim: prim.to_owned(),
    })?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

/// `bin.shr_s : (Binary{N}, Binary{N}) → Binary{N}` — never-silent **arithmetic** (sign-extending)
/// right shift (RFC-0033 §4.1.2/§4.1.3, ADR-028, M-767; SMT-LIB `bvashr`): copies of the sign bit
/// are shifted in where the logical `bin.shr` shifts in zeros, so the result is `⌊a / 2^k⌋` under
/// the two's-complement reading. Same operand shape and refusal contract as [`prim_bin_shr`]
/// (shift amount an unsigned `Binary{N}`; width mismatch / over-cap width / amount `>= N` →
/// explicit [`EvalError::PrimType`], never UB or a silently sign-filled out-of-range shift) —
/// delegated through the same [`bin_shift`] validator.
fn prim_bin_shr_s(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let out = bin_shift(prim, args, binary::shr_signed)?;
    compose_result(
        prim,
        args,
        args[0].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

/// `cmp.lt_s : (Binary{N}, Binary{N}) → Binary{1}` — the **signed** (two's-complement) strict
/// order (RFC-0033 §4.1.2, ADR-028, M-767; SMT-LIB `bvslt`), the distinct named counterpart to
/// `cmp.lt`'s unsigned-magnitude order on `Binary` (e.g. at `Binary{8}`, `0b1111_1111` is `−1 <
/// 0` here but `255 > 0` there). `Binary`-only: balanced ternary is inherently signed, so its D1
/// `cmp.lt` order *is* the signed order — a ternary operand refuses with that routing (explicit,
/// never a silently duplicated order), as does a float (partial order — ADR-040 §2.4) or any
/// width mismatch (never a silent `false`; G2). Width-unbounded, like `cmp.lt`.
fn prim_cmp_lt_s(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    match (args[0].repr(), args[1].repr()) {
        (Repr::Binary { width: wa }, Repr::Binary { width: wb }) => {
            if wa != wb {
                return Err(EvalError::PrimType {
                    prim: prim.to_owned(),
                    why: format!("width mismatch: Binary{{{wa}}} vs Binary{{{wb}}}"),
                });
            }
            let a = as_bits(prim, args[0])?;
            let b = as_bits(prim, args[1])?;
            // Equal widths were just checked, so the kernel order is total (`Some`); the
            // `PrimType` arm is defensive, not a reachable refusal path.
            let ord = binary::cmp_signed(a, b).ok_or_else(|| EvalError::PrimType {
                prim: prim.to_owned(),
                why: "width mismatch".to_owned(),
            })?;
            bool_result(prim, args, ord == Ordering::Less)
        }
        (Repr::Ternary { .. }, Repr::Ternary { .. }) => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "balanced ternary is inherently signed — its cmp.lt order IS the signed order; \
                  use cmp.lt (there is no distinct ternary signed comparison to select)"
                .to_owned(),
        }),
        _ => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "cmp.lt_s is the Binary two's-complement order (ADR-028): both operands must be \
                  Binary{N} of equal width"
                .to_owned(),
        }),
    }
}

// --- DN-41 (M-798): never-silent `Binary` width-cast -------------------------------------------
//
// `bit.width_cast(value: Binary{N}, into: Binary{M}) -> Binary{M}` re-widths an unsigned `Binary`
// value (MSB-first). Because `Binary` is **sign-free** (ADR-028), a re-width is purely a matter of
// the unsigned magnitude:
//   - **Widen** (`M > N`): **zero-extension** — pad `M − N` zero bits on the MSB side. Exact, total,
//     lossless (the unsigned value is unchanged); the guarantee is `Exact`.
//   - **Identity** (`M == N`): a copy. Exact.
//   - **Narrow** (`M < N`): the value fits `Binary{M}` **iff** every dropped high bit (the top
//     `N − M`) is zero. A fitting narrow is exact and lossless; a value that does **not** fit is a
//     never-silent [`EvalError::Overflow`] — never a silent truncation (G2/VR-5), exactly mirroring
//     the `bit.add`/`bit.sub` out-of-range contract.
// **Width witness, not a value operand.** The target width `M` is carried by the *second* operand's
// **width** (`into.repr()` = `Binary{M}`); its *bits are unused*. This threads `M` to the kernel
// through the existing surface→kernel dispatch (`prim_kernel_name`) with no result-type plumbing —
// the motivating call `lt(width_cast(idx8, len32), len32)` reuses the very `Binary{32}` length it is
// about to compare against as the width witness (M-717: widen a `Binary{8}` byte index to compare it
// against a `Binary{32}` `bytes_len`). The result inherits the *first* operand's guarantee/bound by
// the standard `compose_result` threading (an approximate value is refused — width-cast has no
// defined ε-rule; G2). A non-`Binary` operand on either side is an explicit type refusal.

/// `bit.width_cast : (Binary{N}, Binary{M}) → Binary{M}` — never-silent unsigned width-cast (DN-41).
/// The second operand is a **width witness** (only its `Binary{M}` width is read; its bits are
/// ignored). Widening (`M > N`) zero-extends (Exact); narrowing (`M < N`) refuses with
/// [`EvalError::Overflow`] when the value does not fit `M` bits — never a silent truncation (G2).
fn prim_width_cast(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let value = as_bits(prim, args[0])?;
    let witness = as_bits(prim, args[1])?;
    let n = value.len();
    let m = witness.len();
    let out: Vec<bool> = if m >= n {
        // Widen (or identity): zero-extend on the MSB side. `Binary` is sign-free (ADR-028), so the
        // pad bits are always zero; the unsigned magnitude is preserved exactly.
        let mut bits = vec![false; m - n];
        bits.extend_from_slice(value);
        bits
    } else {
        // Narrow: the value fits `Binary{M}` iff the dropped high `N − M` bits are all zero. A set
        // high bit means the magnitude exceeds `2^M − 1`, so the narrow would lose information — an
        // explicit never-silent refusal, never a silent truncation (G2/VR-5).
        let (dropped, kept) = value.split_at(n - m);
        if dropped.iter().any(|&b| b) {
            return Err(EvalError::Overflow {
                prim: prim.to_owned(),
            });
        }
        kept.to_vec()
    };
    // Thread the result off the **value** operand only (the witness contributes its width, not its
    // value/guarantee): compose over `[value]` so the result inherits the value's guarantee/bound
    // (an approximate value is refused — width-cast has no defined ε-propagation rule; G2). The
    // result `Repr` is the witness's own `Binary{M}` (cloned, never reconstructed from a `usize`
    // cast) — its width is `M` by construction, so the output width matches the produced bits.
    compose_result(
        prim,
        &args[..1],
        args[1].repr().clone(),
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

// --- DN-51 §2 D3/§6 (maintainer-authorized DN-39 post-freeze promotion, extends DN-41): `truncate`
//
// The explicit, total, **lossy** counterpart to `width_cast`'s checked narrow: `truncate`
// unconditionally drops the high `N - M` bits and keeps the low `M` — it **never refuses**, unlike
// `width_cast`'s narrow (`EvalError::Overflow` on a set dropped bit). DN-51 §2 D3: "unconditionally
// drops the high `N − M` bits… total but lossy", "only ever via this named op" (so `width_cast`'s
// "never a *silent* truncation" still holds — the loss is only ever opted into by name). Same
// width-witness ABI as `width_cast` (DN-41 §3): the second operand's `Binary{M}` width is the
// result width `M`; its bits are unused. When `M >= N` (widen/identity) no bits are actually
// dropped — this behaves exactly as `width_cast`'s zero-extend/identity; DN-51 names `truncate` as
// the narrowing counterpart to `width_cast` but does not restrict its *domain* to `M < N`, so this
// stays total over every `M` rather than adding a refusal DN-51 never asked for (mirrors
// `width_cast`'s own generality).
//
// **Honesty tag — `Declared`, never `Exact` (DN-51 §4's guarantee matrix, VR-5).** Per CLAUDE.md
// rule 1, `Proven` needs a theorem with *checked* side-conditions and `Empirical` needs trial
// validation; neither applies here — `truncate`'s "loss" is not a numerical approximation of some
// other target quantity with a citable/fittable bound, it is the **explicit, developer-opted-into
// semantics of the named op itself** (the whole point of naming it, per DN-51 §3). That is exactly
// the shape [`wrapping_result`] (RFC-0034 §10/CU-5) already tags `Declared` with a zero-magnitude
// `UserDeclared` bound for the same reason ("the explicit, developer-opted-into semantics… never a
// fabricated `Proven`/`Empirical` claim"), so [`truncate_result`] mirrors it verbatim rather than
// inventing a fresh convention. FLAG (per-instance grading, DN-51 §6 item 4, deferred): a future
// pass could special-case the `M >= N` (no actual loss) instances back to `Exact`/`width_cast`'s own
// tag — not done here; every `bit.truncate` call is uniformly `Declared` today, which is honest
// (never an upgrade) even though it is not maximally precise for the non-narrowing sub-case.

/// Build a `truncate`-tagged result (DN-51 §2 D3): mirrors [`wrapping_result`]'s posture exactly —
/// `Declared` guarantee with a zero-magnitude `UserDeclared` bound, because the lossiness is the
/// op's own declared semantics, not a fittable/provable numeric approximation (see the section note
/// above). Mirrors [`prim_width_cast`]'s witness-invisibility: `inputs` is expected to be just the
/// **value** operand (`&args[..1]`) — the witness contributes its width only, never its
/// guarantee/provenance (DN-41 §3).
fn truncate_result(
    prim: &str,
    inputs: &[&Value],
    repr: Repr,
    payload: Payload,
) -> Result<Value, EvalError> {
    let strength = GuaranteeStrength::propagate(
        GuaranteeStrength::Declared,
        inputs.iter().map(|v| v.meta().guarantee()),
    );
    let provenance = Provenance::Derived {
        op: operation_hash(prim),
        inputs: inputs.iter().map(|v| v.content_hash()).collect(),
    };
    let bound = Bound {
        kind: BoundKind::Error {
            eps: 0.0,
            norm: NormKind::Linf,
        },
        basis: BoundBasis::UserDeclared,
    };
    let meta =
        Meta::new(provenance, strength, Some(bound), None, None, None).map_err(EvalError::Wf)?;
    Value::new(repr, payload, meta).map_err(EvalError::Wf)
}

/// `bit.truncate : (Binary{N}, Binary{M}) → Binary{M}` — DN-51 §2 D3's explicit, total, lossy
/// narrow. Widen/identity (`M >= N`) zero-extends exactly like [`prim_width_cast`] (no bits
/// dropped); narrow (`M < N`) unconditionally keeps the low `M` bits — **never refuses**, unlike
/// `width_cast`'s checked narrow. The second operand is the same width-witness shape as
/// `width_cast` (only its `Binary{M}` width is read; its bits are ignored).
fn prim_truncate(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let value = as_bits(prim, args[0])?;
    let witness = as_bits(prim, args[1])?;
    let n = value.len();
    let m = witness.len();
    let out: Vec<bool> = if m >= n {
        // Widen (or identity): zero-extend, identical to `width_cast` — no bits are dropped.
        let mut bits = vec![false; m - n];
        bits.extend_from_slice(value);
        bits
    } else {
        // Narrow: unconditionally drop the high `N - M` bits and keep the low `M` — total, never
        // refuses (DN-51 §2 D3), unlike `width_cast`'s checked narrow.
        let (_dropped, kept) = value.split_at(n - m);
        kept.to_vec()
    };
    truncate_result(prim, &args[..1], args[1].repr().clone(), Payload::Bits(out))
}

// --- RFC-0032 D3 (M-749): indexed-sequence primitives ------------------------------------------
//
// `seq.get`/`seq.len` are the never-silent indexing surface over `Repr::Seq` (RFC-0032 D3). A kernel
// prim returns a representation [`Value`], not a `.myc` data value, so:
//   - `seq.len(s) -> Binary{32}` is the element count as an unsigned 32-bit value (the seq's `len`).
//   - `seq.get(s, i) -> elem` returns the `i`-th element, with `i` an unsigned `Binary{N}` index. An
//     **out-of-bounds index is an explicit [`EvalError::PrimType`]**, never a panic or a silent
//     default (G2). The `.myc` `Vec::get` surface lifts this into the `Option` the spec names
//     (`get(s, i) -> Option<elem>`); the never-silence is what makes that lift honest.
// Guarantee **`Exact`**: total/decidable over the in-range domain.

/// Interpret an unsigned `Binary{N}` value as a `usize` index (MSB-first bits). A non-Binary operand
/// is an explicit error; a width that cannot fit `usize` (`> 64` here, conservatively the pointer
/// width is ≥ 32) overflowing `usize` is also refused rather than silently truncated (G2).
fn as_index(prim: &str, v: &Value) -> Result<usize, EvalError> {
    let bits = as_bits(prim, v)?;
    if bits.len() > usize::BITS as usize {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "index width {} exceeds the {}-bit usize index space",
                bits.len(),
                usize::BITS
            ),
        });
    }
    // MSB-first accumulate; the width guard above keeps this within `usize`.
    let idx = bits
        .iter()
        .fold(0usize, |acc, &b| (acc << 1) | usize::from(b));
    Ok(idx)
}

/// Extract the elements of a `Repr::Seq` operand; a non-sequence is an explicit error (G2).
fn as_seq<'a>(prim: &str, v: &'a Value) -> Result<&'a [Value], EvalError> {
    v.seq_elems().ok_or_else(|| EvalError::PrimType {
        prim: prim.to_owned(),
        why: "expected a Seq operand".to_owned(),
    })
}

/// `seq.len : Seq<T, N> → Binary{32}` — the element count as an unsigned 32-bit value (RFC-0032 D3).
fn prim_seq_len(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let elems = as_seq(prim, args[0])?;
    // The seq's `len` is a `u32` field (well-formedness caps it at MAX_DIM ≤ 2^30 ≤ 2^32), so 32
    // bits hold it exactly. Use the same *checked* conversion as `bytes.len` rather than a silent
    // `as u32` truncation — defensive parity in the trusted base, so a future path that ever yields
    // an over-2^32 sequence refuses (G2) instead of wrapping.
    let n = u32::try_from(elems.len()).map_err(|_| EvalError::PrimType {
        prim: prim.to_owned(),
        why: format!(
            "sequence length {} exceeds the 32-bit length encoding",
            elems.len()
        ),
    })?;
    u32_as_binary32(prim, args, n)
}

/// `seq.get : (Seq<T, N>, Binary{W}) → T` — never-silent indexed access (RFC-0032 D3). An
/// out-of-bounds index is an explicit [`EvalError::PrimType`] (the `.myc` surface lifts to `Option`),
/// never a panic or a silent default (G2).
fn prim_seq_get(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let elems = as_seq(prim, args[0])?;
    let i = as_index(prim, args[1])?;
    match elems.get(i) {
        Some(e) => {
            // Return the element faithfully at **its own** established basis (VR-5): the result's
            // guarantee is the element's, never upgraded — an `Empirical`/`Declared` element must
            // not re-stamp as `Exact` just because the container/index were `Exact`. It is then
            // `meet`-downgraded by the container and index strengths (you cannot trust an element
            // retrieved from a less-certain container more than that container). Indexing is exact
            // and introduces no error, so the element's `bound` carries through unchanged; if that
            // (guarantee, bound) pairing is internally inconsistent for some exotic container meta,
            // `Meta::new` refuses (`EvalError::Wf`) rather than fabricating one (G2). Provenance is
            // re-stamped `Derived` from the access inputs (lineage), as for any other prim result.
            let guarantee = GuaranteeStrength::propagate(
                e.meta().guarantee(),
                args.iter().map(|v| v.meta().guarantee()),
            );
            let bound = e.meta().bound().cloned();
            let provenance = Provenance::Derived {
                op: operation_hash(prim),
                inputs: args.iter().map(|v| v.content_hash()).collect(),
            };
            let meta =
                Meta::new(provenance, guarantee, bound, None, None, None).map_err(EvalError::Wf)?;
            Value::new(e.repr().clone(), e.payload().clone(), meta).map_err(EvalError::Wf)
        }
        None => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "index {i} out of bounds for a sequence of length {}",
                elems.len()
            ),
        }),
    }
}

// --- RFC-0032 D4 (M-750): byte-string primitives -----------------------------------------------
//
// `bytes.len`/`bytes.get`/`bytes.slice`/`bytes.concat` are the never-silent byte surface over
// `Repr::Bytes` (RFC-0032 D4). UTF-8 decode is written in `.myc` over these prims, never in the
// kernel. Out-of-range access is an explicit refusal (the `.myc` surface lifts to `Option`); a
// non-bytes operand is an explicit type refusal (G2). Guarantee **`Exact`**.

/// Extract the bytes of a `Repr::Bytes` operand; a non-bytes value is an explicit error (G2).
fn as_bytes_payload<'a>(prim: &str, v: &'a Value) -> Result<&'a [u8], EvalError> {
    v.bytes().ok_or_else(|| EvalError::PrimType {
        prim: prim.to_owned(),
        why: "expected a Bytes operand".to_owned(),
    })
}

/// Build a `Binary{32}` value from a `u32`, MSB-first (the never-silent length/index encoding).
fn u32_as_binary32(prim: &str, inputs: &[&Value], n: u32) -> Result<Value, EvalError> {
    let out: Vec<bool> = (0..32).rev().map(|k| (n >> k) & 1 == 1).collect();
    compose_result(
        prim,
        inputs,
        Repr::Binary { width: 32 },
        Payload::Bits(out),
        ApproxRule::Refuse,
    )
}

/// `bytes.len : Bytes → Binary{32}` — the byte count (RFC-0032 D4).
fn prim_bytes_len(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let bytes = as_bytes_payload(prim, args[0])?;
    let n = u32::try_from(bytes.len()).map_err(|_| EvalError::PrimType {
        prim: prim.to_owned(),
        why: format!(
            "byte length {} exceeds the 32-bit length encoding",
            bytes.len()
        ),
    })?;
    u32_as_binary32(prim, args, n)
}

/// `bytes.get : (Bytes, Binary{W}) → Binary{8}` — never-silent indexed byte access (RFC-0032 D4). An
/// out-of-bounds index is an explicit refusal (the `.myc` surface lifts to `Option`), never a silent
/// default (G2). The returned byte is a `Binary{8}` value (MSB-first).
fn prim_bytes_get(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let bytes = as_bytes_payload(prim, args[0])?;
    let i = as_index(prim, args[1])?;
    match bytes.get(i) {
        Some(&byte) => {
            let out: Vec<bool> = (0..8).rev().map(|k| (byte >> k) & 1 == 1).collect();
            compose_result(
                prim,
                args,
                Repr::Binary { width: 8 },
                Payload::Bits(out),
                ApproxRule::Refuse,
            )
        }
        None => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "byte index {i} out of bounds for a byte string of length {}",
                bytes.len()
            ),
        }),
    }
}

/// `bytes.slice : (Bytes, Binary{W}, Binary{W}) → Bytes` — never-silent sub-slice `[start, end)`
/// (RFC-0032 D4). An out-of-range or inverted range is an explicit refusal, never a silently-clamped
/// range (G2).
fn prim_bytes_slice(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 3)?;
    let bytes = as_bytes_payload(prim, args[0])?;
    let start = as_index(prim, args[1])?;
    let end = as_index(prim, args[2])?;
    if start > end || end > bytes.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "slice range [{start}, {end}) is out of bounds or inverted for a byte string of \
                 length {}",
                bytes.len()
            ),
        });
    }
    compose_result(
        prim,
        args,
        Repr::Bytes,
        Payload::Bytes(bytes[start..end].to_vec()),
        ApproxRule::Refuse,
    )
}

/// `bytes.concat : (Bytes, Bytes) → Bytes` — byte concatenation (RFC-0032 D4). Total/`Exact`.
fn prim_bytes_concat(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bytes_payload(prim, args[0])?;
    let b = as_bytes_payload(prim, args[1])?;
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    compose_result(
        prim,
        args,
        Repr::Bytes,
        Payload::Bytes(out),
        ApproxRule::Refuse,
    )
}

// --- M-912 (`enb`, folded-in gap): `bytes.eq` — byte-wise equality ------------------------------
//
// Three ports (diag/error/recover) flagged the gap: `bytes.*` had `len`/`get`/`slice`/`concat` but
// no equality. `bytes.eq(a, b) : (Bytes, Bytes) → Binary{1}` is a straight `[u8]` comparison — no
// approximation involved, total over every pair of `Bytes` operands. Guarantee **`Exact`**.

/// `bytes.eq : (Bytes, Bytes) → Binary{1}` — byte-wise equality (M-912). Total/`Exact`.
fn prim_bytes_eq(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bytes_payload(prim, args[0])?;
    let b = as_bytes_payload(prim, args[1])?;
    bool_result(prim, args, a == b)
}

// --- M-912 (`enb`): `hash.blake3` — the kernel's own content-addressing hash, surfaced -----------
//
// The kernel already content-addresses every definition/value with BLAKE3 (M-103;
// `mycelium-core::content::Canon`/`id::ContentHash`) — deterministic, and the kernel's own basis
// for calling it `Exact` here: this wrapper invokes the *same* algorithm (the `blake3` crate, the
// same workspace-pinned version the kernel depends on) the *same* way — hash the raw bytes, keep
// the raw 32-byte digest — so it adds no additional uncertainty beyond the kernel's own use.
// Conformance is pinned against the official BLAKE3 test vectors (known digests), not merely
// self-consistency (`src/tests/prims.rs`).

/// `hash.blake3 : Bytes → Bytes` — the 32-byte BLAKE3 digest of the input byte string (M-912;
/// M-103). Total and deterministic over every `Bytes` input — there is no failure domain.
/// Guarantee **`Exact`**.
fn prim_hash_blake3(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let bytes = as_bytes_payload(prim, args[0])?;
    let digest = blake3::hash(bytes);
    compose_result(
        prim,
        args,
        Repr::Bytes,
        Payload::Bytes(digest.as_bytes().to_vec()),
        ApproxRule::Refuse,
    )
}

// --- DN-138 WU-4 (increment 2, §6): `bit.to_bytes` -- the missing raw Binary{N} -> Bytes prim ---
//
// `derive(Hash)`'s `ScalarBinary` route had no way to turn a scalar field into `Bytes` for
// `hash.blake3` to consume -- `to_dec`/`digit_byte` (`lib/std/fmt.myc`) render a DECIMAL ASCII
// string, not raw bytes. This prim closes exactly that gap: a total, deterministic bit-repacking.

/// `bit.to_bytes : Binary{N} -> Bytes` (DN-138 WU-4) — zero-pads the `N`-bit unsigned magnitude on
/// the MSB side up to a whole number of bytes (the same zero-extend convention `bit.width_cast`'s
/// widen case uses, DN-41), then packs MSB-first. Total over every `Binary{N}` (`N = 0` produces an
/// empty `Bytes`). Guarantee **`Exact`** — a deterministic bit-repacking, no approximation.
fn prim_bin_to_bytes(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let bits = as_bits(prim, args[0])?;
    let n = bits.len();
    let pad = (8 - n % 8) % 8;
    let mut out = Vec::with_capacity((n + pad) / 8);
    let mut acc: u8 = 0;
    let mut count = 0usize;
    for bit in std::iter::repeat_n(false, pad).chain(bits.iter().copied()) {
        acc = (acc << 1) | u8::from(bit);
        count += 1;
        if count == 8 {
            out.push(acc);
            acc = 0;
            count = 0;
        }
    }
    compose_result(
        prim,
        args,
        Repr::Bytes,
        Payload::Bytes(out),
        ApproxRule::Refuse,
    )
}

// --- DN-58 §A (M-817): the `Binary` `Fuse` semilattice meet ------------------------------------
//
// `fuse(a, b)` is a lawful binary merge over a declared commutative/associative/idempotent meet
// (RFC-0008 RT6). For the `Binary` paradigm the meet is **bitwise-AND** — the greatest-lower-bound of
// the boolean lattice, idempotent (`a ∧ a = a`), commutative, and associative. This is the executable
// **repr** case; the user-`Data` case elaborates instead to the resolved `Fuse::join` call (DN-58
// §A.5), and the non-`Binary` reprs have no committed canonical meet in v0 (DN-58 §A.6 F-A3).
//
// **Provenance shape (DN-58 §A.5 / RFC-0027 §10.6).** A `fuse` result's provenance is the canonical
// `Derived{op:"fuse_join", inputs:[hash(a), hash(b)]}` — the merge node the δ-CRDT Merkle anti-entropy
// story reads downstream — **not** the per-paradigm prim name, so every fusible paradigm shares one
// merge-op identity. The guarantee is the `meet` of the inputs' guarantees (RFC-0001 §4.7); the meet
// op is intrinsically `Exact` (a total greatest-lower-bound). The semilattice laws are **`Empirical`**
// (property-tested over bit-vectors, not mechanized-`Proven` here — VR-5).

/// Compose a `fuse_join` result: the `meet` of the input guarantees + the canonical
/// `Derived{op:"fuse_join", …}` provenance (DN-58 §A.5). The meet op introduces no error, so an
/// **exact** pair yields an exact result with no bound; an **approximate** input has no defined
/// ε-propagation rule for the meet (as for `bit.and`), so it is refused — never a fabricated bound
/// (G2/VR-5).
fn fuse_join_result(
    prim: &str,
    inputs: &[&Value],
    repr: Repr,
    payload: Payload,
) -> Result<Value, EvalError> {
    let strength = GuaranteeStrength::propagate(
        GuaranteeStrength::Exact,
        inputs.iter().map(|v| v.meta().guarantee()),
    );
    if strength != GuaranteeStrength::Exact {
        // No committed ε-rule for the meet over an approximate input — refuse honestly (G2/VR-5),
        // exactly as the underlying `bit.and` does.
        return Err(EvalError::ApproxCompositionUnsupported {
            prim: prim.to_owned(),
        });
    }
    let provenance = Provenance::Derived {
        op: operation_hash("fuse_join"),
        inputs: inputs.iter().map(|v| v.content_hash()).collect(),
    };
    let meta = Meta::new(provenance, GuaranteeStrength::Exact, None, None, None, None)
        .map_err(EvalError::Wf)?;
    Value::new(repr, payload, meta).map_err(EvalError::Wf)
}

/// `fuse_join:binary : (Binary{N}, Binary{N}) → Binary{N}` — the `Binary` `Fuse` meet (bitwise-AND;
/// DN-58 §A). Commutative/associative/idempotent (`Empirical`). A width/paradigm mismatch is an
/// explicit [`EvalError::PrimType`], never a silent coercion (G2). The result carries the canonical
/// `fuse_join` provenance (DN-58 §A.5).
fn prim_fuse_join_binary(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let a = as_bits(prim, args[0])?;
    let b = as_bits(prim, args[1])?;
    if a.len() != b.len() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("width mismatch: {} vs {} bits", a.len(), b.len()),
        });
    }
    let out: Vec<bool> = a.iter().zip(b).map(|(&x, &y)| x & y).collect();
    fuse_join_result(prim, args, args[0].repr().clone(), Payload::Bits(out))
}

// --- RFC-0001 §4.1 / RFC-0002 §5 (M-890, `enb` Gap C): the dense elementwise group ---------------
//
// `dense.add`/`dense.sub`/`dense.neg`/`dense.scale` surface the `mycelium-dense` kernel
// (`add_values`/`sub_values`/`neg_value`/`scale_value`) as prims — the first **tensor-valued**
// prims (operands/results are `Repr::Dense{dim, dtype}` values), and a distinct `dense.*`
// namespace: nothing here touches the integer `bin.*`/`bit.*` conventions.
//
// **Registry pattern for the tensor-valued prims (the rest of Gap C — M-891 dot/similarity,
// M-892..M-894 VSA — mirrors this):** unlike the scalar `bin.*`/`bit.*` prims (kernel codec over
// raw bits → wrapper builds the result via [`compose_result`], whose intrinsic is `Exact`), a
// tensor kernel op **constructs the full result `Value` itself** — payload *and* `Meta` (the
// `Derived{op, inputs}` provenance, the per-op guarantee from `DenseSpace::op_guarantee`, and the
// `Proven` ops' `Bound{Error{eps, Rel}, ProvenThm{citation}}` from `op_rel_eps`). The wrapper
// therefore does NOT re-derive or re-compose the tag: it binds the space off the first operand's
// `Repr`, delegates, **carries the kernel's tag through unchanged** (VR-5: the table/`Π` intrinsic
// mirrors `op_guarantee`, guarded by `tests/prims.rs`), and maps each [`DenseError`] onto the
// interpreter's never-silent error surface:
//   - `Overflow` → [`EvalError::Overflow`] (a result outside the dtype's finite range);
//   - `ApproximateSource` → [`EvalError::ApproxCompositionUnsupported`] (composing an approximate
//     input's own bound with the op ε has no defined rule yet — M-204/M-211; refused, never
//     fabricated);
//   - everything else (dim/dtype/shape mismatch, non-dense operand, off-grid/non-finite payloads,
//     subnormal results outside the cited theorem's side-conditions) → [`EvalError::PrimType`]
//     carrying the kernel's own message — an explicit refusal, never a broadcast/coercion (G2).
//
// **The measurement pair `dense.dot`/`dense.similarity` (M-891).** Same bind-space-and-delegate
// shape, same carried-tag contract — with one honesty point worth naming: their result is a
// **`Dense{1, F64}`** value (the f64 the kernel computed, delivered exactly — never re-rounded
// onto the operand grid), and the carried `Proven` bound is the **binary64 accumulation bound**
// (absolute/`Linf`, `DenseSpace::dot_abs_eps`/`similarity_abs_eps`), NOT the dtype's per-element
// `op_rel_eps`: over exact on-grid operands every product is exact in the f64 accumulator, so the
// dtype ε never enters — and a per-element *relative* claim on a dot product would be false under
// cancellation (VR-5: the tag equals what the kernel can prove, nothing else). `F64` has no dense
// op set (kernel v1 scope), so a measurement cannot silently feed back into dense arithmetic —
// re-entry is an explicit `UnsupportedDtype` refusal via `dense_space_of` (G2).
//
// **`dense.scale`'s scalar operand (pre-Gap-A form — FLAG).** The kernel takes the factor as an
// `f64`, but no scalar-float value form exists yet (that is `enb` Gap A, M-895/M-896, design-gated
// behind the float ADR). The only float-bearing value form today is `Dense` itself, so the factor
// is passed as a **`Dense{1, dtype}` value** (same dtype as the vector; must be `Exact` and
// on-grid — the kernel re-checks the grid via `ScalarOffGrid`). When the scalar-float form lands,
// surfacing a true-scalar variant is a maintainer decision (a new distinct op or a migration) —
// FLAGged in the M-890 report, not silently preempted here.

/// Bind the [`DenseSpace`] a dense prim operates in off its **first operand's** `Repr` (the space
/// anchor); the kernel then enforces every other operand agrees (dim/dtype mismatch → explicit
/// error, never a broadcast). A non-`Dense` first operand or an unsupported dtype (`F16`/`F64`,
/// the kernel's v1 scope) is an explicit [`EvalError::PrimType`].
fn dense_space_of(prim: &str, v: &Value) -> Result<DenseSpace, EvalError> {
    let Repr::Dense { dim, dtype } = *v.repr() else {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("expected a Dense operand, got {:?}", v.repr()),
        });
    };
    DenseSpace::new(dim, dtype).map_err(|e| map_dense_err(prim, e))
}

/// Map a kernel [`DenseError`] onto the interpreter's never-silent error surface (see the module
/// note above for the variant-by-variant rationale). Every arm is explicit; nothing is coerced.
fn map_dense_err(prim: &str, e: DenseError) -> EvalError {
    match e {
        DenseError::Overflow { .. } => EvalError::Overflow {
            prim: prim.to_owned(),
        },
        DenseError::ApproximateSource => EvalError::ApproxCompositionUnsupported {
            prim: prim.to_owned(),
        },
        // Shape/dtype/payload-contract refusals: the kernel's Display already names the offense
        // (dim mismatch with expected/got, off-grid element index, subnormal index, …).
        other => EvalError::PrimType {
            prim: prim.to_owned(),
            why: other.to_string(),
        },
    }
}

/// `dense.add : (Dense{d, s}, Dense{d, s}) → Dense{d, s}` — elementwise addition (M-890).
/// **`Proven`**, carried from the kernel: per-element relative ε (`op_rel_eps`) under the
/// round-to-nearest theorem with checked side-conditions; a dim/dtype mismatch is an explicit
/// refusal, never a broadcast (G2).
fn prim_dense_add(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let space = dense_space_of(prim, args[0])?;
    space
        .add_values(args[0], args[1])
        .map_err(|e| map_dense_err(prim, e))
}

/// `dense.sub : (Dense{d, s}, Dense{d, s}) → Dense{d, s}` — elementwise subtraction (M-890).
/// Same **`Proven`** carried tag and never-silent shape contract as [`prim_dense_add`].
fn prim_dense_sub(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let space = dense_space_of(prim, args[0])?;
    space
        .sub_values(args[0], args[1])
        .map_err(|e| map_dense_err(prim, e))
}

/// `dense.neg : Dense{d, s} → Dense{d, s}` — elementwise negation (M-890). **`Exact`**, carried
/// from the kernel: the dtype grids are symmetric, so negation never rounds.
fn prim_dense_neg(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let space = dense_space_of(prim, args[0])?;
    space.neg_value(args[0]).map_err(|e| map_dense_err(prim, e))
}

/// `dense.scale : (Dense{d, s}, Dense{1, s}) → Dense{d, s}` — scalar multiplication (M-890).
/// **`Proven`**, carried from the kernel. The factor rides a `Dense{1, s}` value (the pre-Gap-A
/// scalar form — see the module note): it must be the **same dtype** as the vector, dim exactly 1,
/// and `Exact` (an approximate factor has no defined composition rule — refused, never fabricated);
/// the kernel re-checks the factor is finite and on-grid (`ScalarOffGrid`).
fn prim_dense_scale(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let space = dense_space_of(prim, args[0])?;
    let factor = args[1];
    let Repr::Dense { dim: 1, dtype } = *factor.repr() else {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "the scale factor must be a Dense{{1, dtype}} scalar (the pre-Gap-A scalar form), \
                 got {:?}",
                factor.repr()
            ),
        });
    };
    if dtype != space.dtype {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "the scale factor dtype {dtype:?} must match the vector dtype {:?} — never a \
                 silent re-round",
                space.dtype
            ),
        });
    }
    if factor.meta().guarantee() != GuaranteeStrength::Exact {
        // Composing the factor's own bound with the op ε has no defined rule (M-204/M-211) —
        // the same honest refusal the kernel makes for an approximate vector operand.
        return Err(EvalError::ApproxCompositionUnsupported {
            prim: prim.to_owned(),
        });
    }
    let Payload::Scalars(xs) = factor.payload() else {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "the scale factor's payload is not scalar data".to_owned(),
        });
    };
    let [c] = xs.as_slice() else {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "the Dense{{1}} scale factor must carry exactly one element, got {}",
                xs.len()
            ),
        });
    };
    space
        .scale_value(args[0], *c)
        .map_err(|e| map_dense_err(prim, e))
}

/// `dense.dot : (Dense{d, s}, Dense{d, s}) → Dense{1, F64}` — the dot-product measurement
/// (M-891). **`Proven`**, carried from the kernel: the absolute (`Linf`) binary64 accumulation
/// bound `dot_abs_eps` with its `ProvenThm` citation rides the result `Value` (see the module
/// note — the dtype's `op_rel_eps` deliberately does NOT appear); a dim/dtype mismatch is an
/// explicit refusal, never a broadcast (G2).
fn prim_dense_dot(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let space = dense_space_of(prim, args[0])?;
    space
        .dot_value(args[0], args[1])
        .map_err(|e| map_dense_err(prim, e))
}

/// `dense.similarity : (Dense{d, s}, Dense{d, s}) → Dense{1, F64}` — the cosine-similarity
/// measurement (M-891). **`Proven`**, carried from the kernel: the input-independent absolute
/// (`Linf`) bound `similarity_abs_eps` (normalization caps the error) with its `ProvenThm`
/// citation. Zero-norm operands yield the kernel's documented convention `0` exactly (disclosed
/// in the citation, never silent).
fn prim_dense_similarity(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let space = dense_space_of(prim, args[0])?;
    space
        .similarity_value(args[0], args[1])
        .map_err(|e| map_dense_err(prim, e))
}

// --- ADR-040 §2.5 (M-898, `enb` Gap A): scalar-float arithmetic over `Repr::Float` -------------
//
// `flt.add`/`flt.sub`/`flt.mul`/`flt.div`/`flt.neg` — IEEE-754 **binary64** arithmetic under
// **round-to-nearest-even only** (RNE, the IEEE default; there is no rounding-mode register —
// rounding is a property of the *operation*, never hidden state; any future non-RNE rounding is a
// distinct named op — ADR-040 §2.2, the ADR-028 signedness-as-operations parallel).
//
// **Never-silent, the float reading (ratified FLAG-2 — ADR-040 §2.4).** Arithmetic specials are
// **in-band, inspectable, propagating values**, not errors: overflow → ±inf, x/±0 → ±inf (sign by
// IEEE), 0/0 and inf−inf → NaN. The rationale (argued in the ADR, not assumed): integer overflow
// is never-silent-by-refusal because wraparound *aliases an ordinary in-range value*; float
// overflow lands on a *distinguished sentinel that propagates* and is directly inspectable — the
// in-band signal IS the never-silent mechanism, and trapping would make standard IEEE algorithms
// inexpressible. So these five ops are **total** over `Float` operands: no `Overflow` refusal, no
// silent alias. (Conversion boundaries — float↔integer etc. — remain explicit-error; they are not
// these ops.) Every NaN result carries the canonical bits by [`Value::new`]'s construction
// invariant (ADR-040 §2.3; `mycelium_core::CANONICAL_NAN_BITS`) — one NaN, one address.
//
// **Per-op tag — `Empirical`, per the ratified ADR-040 §2.6 (VR-5: never upgraded).** The op's
// *definition* is "the correctly-rounded IEEE-754 binary64 result under RNE" (`Exact` as a
// definition — it is the spec). The *implementation claim* — that the host's f64 arithmetic
// delivers exactly that bit pattern — is **`Empirical`** at introduction: it rests on the
// "Rust f64 is IEEE-754 binary64" platform statement (`Declared`, the Rust reference; not
// independently verified) and is pinned by the hand-derived IEEE reference-case corpus in
// `src/tests/prims.rs` (exactly [`FLT_CONFORMANCE_TRIALS`] cases whose expected bit patterns are
// derived from IEEE-754 semantics by hand, not recomputed with the op under test). The disclosed
// bound is therefore a **zero-deviation-vs-spec** claim: `ErrorBound{eps: 0.0, Linf}` against the
// IEEE-defined correctly-rounded result, basis `EmpiricalFit{FLT_CONFORMANCE_TRIALS, method}` —
// deliberately NOT a rounding-error bound vs *real* arithmetic (that claim is a theorem with
// side-conditions nobody checks here; claiming it would be an unearned `Proven` — ADR-040 §2.6
// claims no `Proven` anywhere). libm is NOT involved (§2.5 keeps transcendentals out of the
// kernel), so this is not the Empirical-libm accuracy case: the `Empirical` here is host-RNE
// *conformance*, not an approximation fit.
//
// **Composition (the M-204 posture, float form).** A `flt.*` result is `Empirical`, so chained
// float arithmetic must compose: an input is accepted iff it is `Exact` (e.g. a float literal,
// M-897) or carries exactly this zero-deviation `Empirical` form (a prior `flt.*` result) —
// zero-deviation-vs-spec composes to zero-deviation-vs-spec under any deterministic op, so the
// composed claim stays checked, never fabricated. Any *other* bound (a genuine approximation)
// has no defined float ε-propagation rule yet and is an explicit
// [`EvalError::ApproxCompositionUnsupported`] refusal (G2/VR-5 — refuse, don't guess).

/// The trial count of the M-898 IEEE reference-case corpus (`src/tests/prims.rs`,
/// `flt_reference_case_corpus`) — the evidence behind the `EmpiricalFit` basis every `flt.*`
/// result carries. The corpus test asserts its row count equals this constant, so the recorded
/// trials can never silently drift from the trials actually run (VR-5).
pub const FLT_CONFORMANCE_TRIALS: u64 = 40;

/// The method recorded in the `EmpiricalFit` basis of every `flt.*` result (ADR-040 §2.6).
pub const FLT_CONFORMANCE_METHOD: &str = "bit-reproducibility differential against hand-derived \
     IEEE-754 binary64 RNE reference cases (exact-arithmetic rows, ties-to-even at the 2^53 \
     boundary, overflow/underflow edges, signed zeros, in-band specials algebra, canonical-NaN \
     identity)";

/// The zero-deviation-vs-spec bound every `flt.*` result carries (see the module note above):
/// the delivered bit pattern deviates from the IEEE-754-defined correctly-rounded RNE binary64
/// result by at most 0 (`Linf`), on the `EmpiricalFit` evidence of the reference-case corpus.
fn flt_bound() -> Bound {
    Bound {
        kind: BoundKind::Error {
            eps: 0.0,
            norm: NormKind::Linf,
        },
        basis: BoundBasis::EmpiricalFit {
            trials: FLT_CONFORMANCE_TRIALS,
            method: FLT_CONFORMANCE_METHOD.to_owned(),
        },
    }
}

/// Extract the binary64 scalar of a `Float` operand; any other representation is an explicit
/// [`EvalError::PrimType`] — never a silent coercion (G2). The returned NaN, if any, already
/// carries the canonical bits ([`mycelium_core::CANONICAL_NAN_BITS`]) by `Value::new`'s
/// construction invariant.
fn as_float(prim: &str, v: &Value) -> Result<f64, EvalError> {
    match (v.repr(), v.payload()) {
        (Repr::Float { .. }, Payload::Float(x)) => Ok(*x),
        _ => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("expected a Float operand, got {:?}", v.repr()),
        }),
    }
}

/// Whether a `flt.*` input's tag/bound is composable (the module-note rule): `Exact`, or the
/// zero-deviation `Empirical` form a prior `flt.*` op produced. Anything else — a genuine
/// approximation bound — has no defined float ε-propagation rule yet.
fn flt_input_composable(v: &Value) -> bool {
    match v.meta().guarantee() {
        GuaranteeStrength::Exact => true,
        GuaranteeStrength::Empirical => matches!(
            v.meta().bound(),
            Some(Bound {
                kind: BoundKind::Error { eps, .. },
                basis: BoundBasis::EmpiricalFit { .. },
            }) if *eps == 0.0
        ),
        _ => false,
    }
}

/// Build a `flt.*` result: `Float{F64}` repr, the computed scalar (NaN canonicalized by
/// [`Value::new`] — ADR-040 §2.3), `Derived` provenance, and the honest ADR-040 §2.6 tag —
/// strength `meet(Empirical, inputs)` with the zero-deviation `EmpiricalFit` bound. An input
/// that is neither `Exact` nor the composable zero-deviation form is an explicit
/// [`EvalError::ApproxCompositionUnsupported`] (never a fabricated bound — G2/VR-5).
fn flt_result(prim: &str, inputs: &[&Value], out: f64) -> Result<Value, EvalError> {
    empirical_flt_result(
        prim,
        inputs,
        Repr::Float {
            width: FloatWidth::F64,
        },
        Payload::Float(out),
    )
}

/// Build a result for any op whose **intrinsic** is `Empirical` with the shared zero-deviation-
/// vs-spec bound (ADR-040 §2.6 — arithmetic, comparison, *and* conversion prims all carry this
/// posture, §2.6's "Conversions: range/exactness checks `Empirical`…" clause). Generalizes
/// [`flt_result`] (always a `Float` output) to an arbitrary output `repr`/`payload`, shared with
/// the CU-3 `bin.to_flt`/`flt.to_bin` conversion prims. Composability is the same rule
/// [`flt_result`] always used: every input must be `Exact` or the zero-deviation `Empirical` form
/// a prior `flt.*`/conversion op produced ([`flt_input_composable`]); anything else is an explicit
/// [`EvalError::ApproxCompositionUnsupported`] refusal, never a fabricated bound (G2/VR-5).
fn empirical_flt_result(
    prim: &str,
    inputs: &[&Value],
    repr: Repr,
    payload: Payload,
) -> Result<Value, EvalError> {
    if !inputs.iter().all(|v| flt_input_composable(v)) {
        return Err(EvalError::ApproxCompositionUnsupported {
            prim: prim.to_owned(),
        });
    }
    // Inputs are Exact or the zero-deviation Empirical form, and the op contributes Empirical
    // (ADR-040 §2.6) ⇒ the meet is Empirical, paired with the zero-deviation bound (M-I1/M-I3).
    let strength = GuaranteeStrength::propagate(
        GuaranteeStrength::Empirical,
        inputs.iter().map(|v| v.meta().guarantee()),
    );
    let provenance = Provenance::Derived {
        op: operation_hash(prim),
        inputs: inputs.iter().map(|v| v.content_hash()).collect(),
    };
    // The `Wf` arms are defensive, as in `compose_result`: strength is Empirical-with-bound by
    // construction here, and the payload matches the repr — kept explicit so a future
    // inconsistency refuses honestly instead of panicking (G2).
    let meta = Meta::new(provenance, strength, Some(flt_bound()), None, None, None)
        .map_err(EvalError::Wf)?;
    Value::new(repr, payload, meta).map_err(EvalError::Wf)
}

/// Shared arity/operand extraction for the binary `flt.*` ops.
fn flt_binop(prim: &str, args: &[&Value]) -> Result<(f64, f64), EvalError> {
    expect_arity(prim, args, 2)?;
    Ok((as_float(prim, args[0])?, as_float(prim, args[1])?))
}

/// `flt.add : (Float, Float) → Float` — IEEE-754 binary64 addition, RNE (ADR-040 §2.2/§2.5;
/// M-898). Total: overflow → ±inf, NaN propagates (in-band specials, ratified FLAG-2 — the
/// module note above). Tag `Empirical` with the zero-deviation-vs-spec bound (ADR-040 §2.6).
fn prim_flt_add(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = flt_binop(prim, args)?;
    flt_result(prim, args, a + b)
}

/// `flt.sub : (Float, Float) → Float` — IEEE-754 binary64 subtraction, RNE. Same total/in-band
/// contract and `Empirical` tag as [`prim_flt_add`].
fn prim_flt_sub(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = flt_binop(prim, args)?;
    flt_result(prim, args, a - b)
}

/// `flt.mul : (Float, Float) → Float` — IEEE-754 binary64 multiplication, RNE. Same total/in-band
/// contract and `Empirical` tag as [`prim_flt_add`] (`0 × inf → NaN`, canonical).
fn prim_flt_mul(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = flt_binop(prim, args)?;
    flt_result(prim, args, a * b)
}

/// `flt.div : (Float, Float) → Float` — IEEE-754 binary64 division, RNE. Total: `x/±0 → ±inf`
/// (sign by IEEE), `0/0 → NaN` (canonical) — **in-band, never a trap** (the ratified FLAG-2
/// policy; the distinguished sentinel is the never-silent signal, unlike `bin.div`'s integer
/// div-by-zero, which has no in-band sentinel and must refuse). Tag as [`prim_flt_add`].
fn prim_flt_div(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = flt_binop(prim, args)?;
    flt_result(prim, args, a / b)
}

/// `flt.neg : Float → Float` — IEEE-754 binary64 negation (sign-bit flip; exact in binary64 —
/// negation never rounds, and `neg(neg(x))` is a bit-identity). The tag still carries the group's
/// `Empirical` host-conformance posture (ADR-040 §2.6 tags the whole `flt.*` set; splitting `neg`
/// out at a stronger tag is a maintainer call — FLAGged in the M-898 report, not silently taken).
/// `neg(NaN)` re-canonicalizes to the positive quiet NaN (§2.3 — NaN sign/payload bits are not
/// observable).
fn prim_flt_neg(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let a = as_float(prim, args[0])?;
    flt_result(prim, args, -a)
}

// --- ADR-040 §2.4 (M-899, `enb` Gap A): scalar-float comparison + the named total order --------
//
// `flt.lt`/`flt.le`/`flt.gt`/`flt.ge`/`flt.eq` — the IEEE-754 §5.11 quiet comparison
// **predicates** over two `Float` operands, reducing to a `Binary{1}` truth value (the realized
// `Bool`, the RFC-0032 D1 engineering note). Float ordering is **partial** (ADR-040 §2.4): NaN is
// **unordered against everything, itself included**, and a comparison involving NaN yields the
// IEEE-*defined* predicate value **false** on every one of the five — `flt.lt(NaN, x)`,
// `flt.gt(NaN, x)`, `flt.le(NaN, x)`, `flt.ge(NaN, x)`, and `flt.eq(NaN, NaN)` are all `false`
// (NaN ≠ NaN). This is **not** a silent `false`-as-less-than (the §2.4 clause): `false` from
// `flt.lt` asserts only "no `<` relation holds", and the no-order case is directly *observable*
// from the predicate set — `¬flt.le(a,b) ∧ ¬flt.gt(a,b)` ⟺ unordered, `¬flt.eq(x,x)` ⟺ NaN — so
// nothing is funneled into a fabricated ordering (G2). The `Option`-shaped three-way
// (`partial_cmp → None` on NaN, `docs/spec/stdlib/cmp.md` Q1) is the `std.cmp` *surface* built on
// these predicates, not a kernel prim.
//
// **`flt.total_le` — the named, opt-in total order (IEEE-754 §5.10 `totalOrder`).** For
// sorting/keying, where a partial order cannot place NaN, the ADR ships a *distinct, named* op —
// imposing a total order silently is exactly what cmp.md Q1 rejects. `flt.total_le(a, b)` is
// `totalOrder(a, b)`: true iff `a` precedes-or-equals `b` in the total order
// `−inf < … < −0 < +0 < … < +inf < NaN` (with §2.3's canonical positive quiet NaN, the one NaN
// sorts *last*, deterministically). It is **total and reflexive** — `flt.total_le(NaN, NaN)` is
// `true` where `flt.le(NaN, NaN)` is `false` — and it separates the signed zeros
// (`total_le(−0, +0)` but not `total_le(+0, −0)`) where `flt.eq(−0, +0)` calls them equal: the
// documented ADR-040 FLAG-4 identity-vs-equality seam, made orderable *by name*.
//
// **Per-op tag — `Empirical`, per the ratified ADR-040 §2.6 (VR-5: never upgraded).** Two
// distinct claims, both `Empirical`:
//   - the five predicates: the *definition* is IEEE-754 §5.11 (`Exact` as a definition); the
//     *implementation claim* that the host's `f64` comparison operators deliver it is
//     `Empirical`, pinned by the hand-derived reference corpus (`flt_cmp_reference_case_corpus`,
//     exactly [`FLT_CMP_CONFORMANCE_TRIALS`] rows) atop the `Declared` "Rust f64 is IEEE-754
//     binary64" platform statement;
//   - `flt.total_le`: additionally, the **total-order property itself** (totality,
//     antisymmetry, transitivity, the NaN/−0/+0 placement) is property-tested over the value
//     corpus but **not proven — this is the M-511 proof debt, load-bearing here**. The tag
//     stays `Empirical` until a checked theorem lands (M-511); claiming `Proven` from
//     "Rust's `total_cmp` says it implements totalOrder" would upgrade past the basis (VR-5).
// The disclosed bound is the same zero-deviation-vs-spec shape as the arithmetic group: the
// delivered truth value deviates from the IEEE-defined predicate value by 0 (`Linf`) on the
// `EmpiricalFit` evidence of the comparison corpus.
//
// **Composition (the M-204 posture, unchanged).** Same rule as `flt_result`: inputs must be
// `Exact` or the zero-deviation `Empirical` form; a genuinely-approximate input has no defined
// comparison ε-rule (an ε-ball straddling the compare point flips the bit) and is an explicit
// [`EvalError::ApproxCompositionUnsupported`] refusal — refuse, don't guess (G2/VR-5).

/// The trial count of the M-899 IEEE comparison reference corpus (`src/tests/prims.rs`,
/// `flt_cmp_reference_case_corpus`) — the evidence behind the `EmpiricalFit` basis every
/// float-comparison result carries. The corpus test asserts its row count equals this constant,
/// so the recorded trials can never silently drift from the trials actually run (VR-5).
pub const FLT_CMP_CONFORMANCE_TRIALS: u64 = 44;

/// The method recorded in the `EmpiricalFit` basis of every float-comparison result
/// (ADR-040 §2.4/§2.6).
pub const FLT_CMP_CONFORMANCE_METHOD: &str = "truth-table differential against hand-derived \
     IEEE-754 §5.11 predicate rows (NaN unordered on every predicate, NaN ≠ NaN, signed-zero \
     equality, ±inf ordering) and §5.10 totalOrder rows (−0 ≺ +0, canonical NaN last, NaN \
     reflexive) — total-order property evidence is Empirical pending the M-511 proof";

/// The zero-deviation-vs-spec bound every float-comparison result carries (see the section note):
/// the delivered `Binary{1}` truth value deviates from the IEEE-754-defined predicate value by
/// at most 0 (`Linf`), on the `EmpiricalFit` evidence of the comparison reference corpus.
fn flt_cmp_bound() -> Bound {
    Bound {
        kind: BoundKind::Error {
            eps: 0.0,
            norm: NormKind::Linf,
        },
        basis: BoundBasis::EmpiricalFit {
            trials: FLT_CMP_CONFORMANCE_TRIALS,
            method: FLT_CMP_CONFORMANCE_METHOD.to_owned(),
        },
    }
}

/// Build a float-comparison result: `Binary{1}` (`0b1` = true), `Derived` provenance, and the
/// honest ADR-040 §2.6 tag — strength `meet(Empirical, inputs)` with the zero-deviation
/// `EmpiricalFit` comparison bound. An input that is neither `Exact` nor the composable
/// zero-deviation form is an explicit [`EvalError::ApproxCompositionUnsupported`] (an ε-ball
/// around a compared operand could flip the bit — no fabricated bound, G2/VR-5).
fn flt_cmp_result(prim: &str, inputs: &[&Value], truth: bool) -> Result<Value, EvalError> {
    if !inputs.iter().all(|v| flt_input_composable(v)) {
        return Err(EvalError::ApproxCompositionUnsupported {
            prim: prim.to_owned(),
        });
    }
    let strength = GuaranteeStrength::propagate(
        GuaranteeStrength::Empirical,
        inputs.iter().map(|v| v.meta().guarantee()),
    );
    let provenance = Provenance::Derived {
        op: operation_hash(prim),
        inputs: inputs.iter().map(|v| v.content_hash()).collect(),
    };
    // Defensive `Wf` arms, as in `flt_result`: strength is Empirical-with-bound by construction
    // and a one-bit payload matches `Binary{1}` — kept explicit so a future inconsistency
    // refuses honestly instead of panicking (G2).
    let meta = Meta::new(
        provenance,
        strength,
        Some(flt_cmp_bound()),
        None,
        None,
        None,
    )
    .map_err(EvalError::Wf)?;
    Value::new(Repr::Binary { width: 1 }, Payload::Bits(vec![truth]), meta).map_err(EvalError::Wf)
}

/// `flt.lt : (Float, Float) → Binary{1}` — IEEE-754 §5.11 `compareQuietLess` (ADR-040 §2.4;
/// M-899). **NaN is unordered:** any NaN operand → `false` (the defined predicate value, not a
/// fallback — see the section note). Tag `Empirical` with the zero-deviation comparison bound.
fn prim_flt_lt(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = flt_binop(prim, args)?;
    flt_cmp_result(prim, args, a < b)
}

/// `flt.le : (Float, Float) → Binary{1}` — IEEE-754 §5.11 `compareQuietLessEqual`. **NaN is
/// unordered:** any NaN operand → `false` (`le` is NOT `¬gt` on floats — both are false on
/// unordered). Tag as [`prim_flt_lt`].
fn prim_flt_le(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = flt_binop(prim, args)?;
    flt_cmp_result(prim, args, a <= b)
}

/// `flt.gt : (Float, Float) → Binary{1}` — IEEE-754 §5.11 `compareQuietGreater`. **NaN is
/// unordered:** any NaN operand → `false` (NaN is not "the biggest" under the partial order —
/// only the *named* total order `flt.total_le` places it). Tag as [`prim_flt_lt`].
fn prim_flt_gt(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = flt_binop(prim, args)?;
    flt_cmp_result(prim, args, a > b)
}

/// `flt.ge : (Float, Float) → Binary{1}` — IEEE-754 §5.11 `compareQuietGreaterEqual`. **NaN is
/// unordered:** any NaN operand → `false`. Tag as [`prim_flt_lt`].
fn prim_flt_ge(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = flt_binop(prim, args)?;
    flt_cmp_result(prim, args, a >= b)
}

/// `flt.eq : (Float, Float) → Binary{1}` — IEEE-754 §5.11 `compareQuietEqual`. **NaN ≠ NaN**
/// (`flt.eq(NaN, NaN) = false` — the honest unordered result; `¬flt.eq(x, x)` is the in-band
/// NaN test), and the signed zeros compare **equal** (`flt.eq(−0, +0) = true`) even though they
/// are bit- and address-distinct — the documented ADR-040 FLAG-4 identity-vs-equality seam
/// (`flt.total_le` is the op that *does* separate them). Tag as [`prim_flt_lt`].
fn prim_flt_eq(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = flt_binop(prim, args)?;
    #[allow(clippy::float_cmp)] // IEEE §5.11 compareQuietEqual IS the op's spec, not a bug.
    flt_cmp_result(prim, args, a == b)
}

/// `flt.total_le : (Float, Float) → Binary{1}` — IEEE-754 §5.10 `totalOrder(a, b)`: `a`
/// precedes-or-equals `b` in the **named, opt-in total order**
/// `−inf < … < −0 < +0 < … < +inf < NaN` (ADR-040 §2.4; M-899). Total and reflexive — every
/// pair is ordered, `total_le(NaN, NaN) = true`, the canonical NaN (§2.3) sorts last, and
/// `−0`/`+0` are *distinct* (`total_le(−0, +0) ∧ ¬total_le(+0, −0)`) — so sorting/keying floats
/// is deterministic, by explicit request, never by a silently-imposed order (cmp.md Q1).
///
/// **Tag `Empirical` — the M-511 proof debt (VR-5).** The total-order *property* (totality,
/// antisymmetry, transitivity, the placement rules) is corpus/property-tested here but has **no
/// checked proof**; it stays `Empirical` until M-511 discharges it — never `Proven` on the
/// strength of the host's `total_cmp` documentation alone.
fn prim_flt_total_le(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let (a, b) = flt_binop(prim, args)?;
    flt_cmp_result(prim, args, a.total_cmp(&b) != Ordering::Greater)
}

/// Extract the single `f64` of a unary `Float` classification prim (arity + `Repr::Float` checked,
/// never-silent). The unary analogue of [`flt_binop`].
fn flt_unop(prim: &str, args: &[&Value]) -> Result<f64, EvalError> {
    expect_arity(prim, args, 1)?;
    as_float(prim, args[0])
}

/// `flt.is_nan : (Float) → Binary{1}` — IEEE-754 §5.7.2 `isNaN`: `true` iff the operand is NaN
/// (CU-2; **ADR-040 §2.5** mandates classification "`is_nan`/`is_finite` at minimum"). This is the
/// in-band never-silent test for the invalid-operation sentinel (§2.4): `¬flt.eq(x, x)` and
/// `flt.is_nan(x)` agree, but this is the direct, total predicate. Total — every float is or isn't
/// NaN; no operand refuses. Tag `Empirical` (ADR-040 §2.6 — the float op set's strength).
fn prim_flt_is_nan(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let x = flt_unop(prim, args)?;
    flt_cmp_result(prim, args, x.is_nan())
}

/// `flt.is_finite : (Float) → Binary{1}` — IEEE-754 §5.7.2 `isFinite`: `true` iff the operand is
/// neither ±inf nor NaN (CU-2; **ADR-040 §2.5** mandate). The direct never-silent test for "is this
/// an ordinary finite value" — the guard that distinguishes an in-range result from a propagated
/// ±inf/NaN overflow sentinel (§2.4). Total; tag `Empirical` (ADR-040 §2.6).
fn prim_flt_is_finite(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let x = flt_unop(prim, args)?;
    flt_cmp_result(prim, args, x.is_finite())
}

/// `flt.is_infinite : (Float) → Binary{1}` — IEEE-754 §5.7.2 `isInfinite`: `true` iff the operand is
/// +inf or −inf (CU-2; the third classification predicate rounding out ADR-040 §2.5's "at minimum"
/// pair — `is_infinite ≡ ¬is_finite ∧ ¬is_nan`, provided directly so callers need not compose it).
/// Total; tag `Empirical` (ADR-040 §2.6).
fn prim_flt_is_infinite(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    let x = flt_unop(prim, args)?;
    flt_cmp_result(prim, args, x.is_infinite())
}

// --- ADR-040 §2.4 (CU-3): never-silent Binary↔Float conversions --------------------------------
//
// `bin.to_flt`/`flt.to_bin` are the kernel prims behind the surface `bin_to_flt`/`flt_to_bin`
// (`checkty.rs::try_check_float_prim`) — the "target-width prim" shape of `bit.width_cast`
// (DN-41), crossing the Binary/Float paradigms. Both read `Binary` as the **unsigned** magnitude
// ([`binary::bits_to_uint`]/[`binary::uint_to_bits`], sign-free per ADR-028), mirroring
// `bit.width_cast` rather than the *signed* two's-complement reading `bin.add`/`bin.mul` use — a
// signed variant is a natural, undecided follow-on (flagged in the CU-3 leaf report, never
// guessed). Tag **`Empirical`** on both (ADR-040 §2.6: "Conversions: range/exactness checks
// `Empirical` via property tests on the documented bounds (2^53, target-range edges)"), carrying
// the same zero-deviation-vs-spec bound the `flt.*` arithmetic/comparison groups use
// ([`empirical_flt_result`]).
//
// **The lossy `bin→flt` rounding direction is explicitly out of scope.** For a `Binary{N}`
// magnitude exceeding `2^53`, [`prim_bin_to_flt`] refuses rather than silently rounding — a lossy
// conversion is a reified *swap* carrying its rounding bound (ADR-040 §2.4/§5), not a prim; no
// such swap is added here (see the CU-3 leaf report FLAG: the swap machinery to carry that bound
// does not yet exist in this crate).

/// `bin.to_flt : Binary{N} → Float` — checked-exact never-silent conversion (ADR-040 §2.4; CU-3).
/// A non-`Binary` operand is an explicit [`EvalError::PrimType`]; an over-cap width (`N >`
/// [`binary::FLOAT_CONV_MAX_WIDTH`]) is likewise an explicit [`EvalError::PrimType`]; a magnitude
/// exceeding the binary64 exact-integer bound (`2^53`) is an explicit [`EvalError::Overflow`] —
/// never a silent lossy round.
fn prim_bin_to_flt(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 1)?;
    let bits = as_bits(prim, args[0])?;
    if bits.len() > binary::FLOAT_CONV_MAX_WIDTH {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "width {} exceeds the {}-bit bin.to_flt conversion cap (CU-3 scope)",
                bits.len(),
                binary::FLOAT_CONV_MAX_WIDTH
            ),
        });
    }
    let out = binary::checked_uint_to_f64(bits).ok_or_else(|| EvalError::Overflow {
        prim: prim.to_owned(),
    })?;
    empirical_flt_result(
        prim,
        args,
        Repr::Float {
            width: FloatWidth::F64,
        },
        Payload::Float(out),
    )
}

/// `flt.to_bin : (Float, Binary{M}) → Binary{M}` — never-silent conversion (ADR-040 §2.4; CU-3).
/// The second operand is a **width witness** (only its `Binary{M}` width is read; its bits are
/// ignored — exactly `bit.width_cast`'s shape, DN-41). Refuses with an explicit
/// [`EvalError::PrimType`] on a non-`Float` first operand, an over-cap witness width, NaN, ±inf, a
/// negative value, or a nonzero fractional part (dropping it would be a silent truncation);
/// refuses with an explicit [`EvalError::Overflow`] when the integer magnitude does not fit the
/// unsigned `M`-bit target or exceeds the binary64 exact-integer bound (`2^53`) — never a silent
/// round/truncate-by-default. The result's guarantee threads off the **value** operand only (the
/// witness contributes its width, not its guarantee — mirroring [`prim_width_cast`]).
fn prim_flt_to_bin(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let value = as_float(prim, args[0])?;
    let witness = as_bits(prim, args[1])?;
    if witness.len() > binary::FLOAT_CONV_MAX_WIDTH {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "witness width {} exceeds the {}-bit flt.to_bin conversion cap (CU-3 scope)",
                witness.len(),
                binary::FLOAT_CONV_MAX_WIDTH
            ),
        });
    }
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "flt.to_bin requires a non-negative, finite, integer-valued Float (got {value}) \
                 — never a silent NaN/inf/fractional coercion (ADR-040 §2.4)"
            ),
        });
    }
    // `m` fits u32: `witness.len() <= FLOAT_CONV_MAX_WIDTH == 64` by the cap check above.
    let m = u32::try_from(witness.len()).expect("witness width <= 64 fits u32");
    let magnitude = binary::checked_f64_to_uint(value, m).ok_or_else(|| EvalError::Overflow {
        prim: prim.to_owned(),
    })?;
    let out = binary::uint_to_bits(magnitude, m)
        .expect("checked_f64_to_uint already verified magnitude fits the m-bit unsigned target");
    empirical_flt_result(prim, &args[..1], args[1].repr().clone(), Payload::Bits(out))
}

// --- RFC-0003 §3/§4 / ADR-008 (M-892, `enb` Gap C): the VSA bind group ---------------------------
//
// `vsa.bind`/`vsa.unbind`/`vsa.permute` surface the `mycelium-vsa` Value-level model ops as
// **model-dispatched** prims over `Repr::Vsa{model, dim, sparsity}` values: the wrapper reads the
// **first operand's** model id and dispatches to that model's own Value-level op
// (`MapI::bind_values` / `Fhrr::unbind_values` / `Bsc::permute_value`, …). The **introduction
// dispatch set is MAP-I / FHRR / BSC** (the M-892 model set); an operand naming any other model —
// including kernel models with no Value-level op set surfaced yet (HRR/MAP-B/SBC) — is an explicit
// [`EvalError::PrimType`] refusal naming the model and the set, never a guessed algebra (G2).
// Widening the set is an append-only extension (it must also recompute the Π meet tags — see
// `mycelium-core::PrimTable::builtins`). This mirrors the M-890/M-891 tensor-valued registry
// pattern and sets the shape for the rest of the VSA surface (M-893 bundle, M-894
// cleanup/reconstruct).
//
// **Tags — constructed by the kernel per model, carried unchanged (VR-5).** The kernel op builds
// the full result `Value` (payload + `Meta`): MAP-I bind/unbind/permute and BSC bind/unbind/permute
// are **`Exact`** (algebraic — the bipolar/XOR self-inverse identities and the cyclic shift), FHRR
// bind/permute are **`Exact`** (deterministic phasor algebra) and FHRR **unbind is `Empirical`**
// (the RFC-0003 §4 normative weak-link tag) carrying its trial-validated `FHRR_UNBIND_PROFILE` δ
// bound — and it is **regime-gated by the kernel**: the unbound operand must be a single
// `vsa.fhrr.bind` product (provenance-checked), refused explicitly outside the validated regime
// rather than stretching the tag past its evidence. The wrapper never re-derives or re-stamps any
// of this. Note the Π table's single `vsa.unbind` intrinsic is the **meet** over the dispatch set
// (`Empirical`); the runtime value carries the dispatched model's own — possibly stronger — kernel
// tag (guarded by the meet-consistency test in `src/tests/prims.rs`).
//
// **Provenance is model-namespaced.** The kernel stamps `Derived{op: hash("vsa.map_i.bind"), …}`
// (the model's own op id), not `hash("vsa.bind")`: the dispatch *result* is recorded, more specific
// than the dispatching prim name — inspectable and EXPLAIN-able (G2), and exactly what makes the
// FHRR unbind regime gate work through the prim path (it recognizes `vsa.fhrr.bind` products).
//
// **Exact-input guard (the M-204 posture, VSA form).** The kernel Value-level ops stamp their
// per-op tag without meeting the *inputs'* strengths (their alphabet/regime guards make that safe
// for every payload-reachable case — a MAP-I/BSC bundle is out-of-alphabet at unbind, an FHRR
// non-bind-product is regime-refused). One hole remains reachable through the prim surface: an
// **in-alphabet but non-`Exact` value** (e.g. an `Empirical` FHRR unbind result — a valid phase
// vector — fed back into `vsa.bind`) would come out stamped `Exact`, a silent upgrade (VR-5).
// There is **no defined δ-propagation rule through the VSA algebra yet**, so the wrapper refuses
// non-`Exact` operands with [`EvalError::ApproxCompositionUnsupported`] — refuse, never fabricate
// (G2). The honest noisy-decode path is cleanup with its own disclosed confidence (M-894), not a
// silently-Exact re-bind.
//
// **`vsa.permute`'s shift operand.** The kernel takes `shift: i64`; the surface operand is an
// unsigned `Binary{W}` magnitude (read like `seq.get`'s index — MSB-first). Rotation is cyclic
// (`rem_euclid dim`), so every permutation is reachable and the *inverse* permute is the
// complementary shift `dim − s` — no negative-shift form is needed (a signed-shift variant, if
// ever wanted, is a distinct named op per ADR-028's signedness-as-operations rule).
//
// Error mapping ([`map_vsa_err`]): `VsaError::Wf` → [`EvalError::Wf`]; everything else
// (dim/model mismatch, out-of-alphabet component, outside-empirical-profile regime refusals) →
// [`EvalError::PrimType`] carrying the kernel's own message — explicit, never a coercion (G2).

/// The model-dispatch target of a `vsa.*` prim: the M-892 introduction set (MAP-I / FHRR / BSC).
/// Constructed by [`vsa_model_of`] off the first operand's `Repr::Vsa` model id.
enum VsaDispatch {
    /// Multiply-Add-Permute (bipolar) — bind/unbind/permute all `Exact`.
    MapI(MapI),
    /// Fourier HRR (phasor) — bind/permute `Exact`; unbind `Empirical`, regime-gated.
    Fhrr(Fhrr),
    /// Binary Spatter Code — bind/unbind/permute all `Exact`.
    Bsc(Bsc),
}

impl VsaDispatch {
    /// Dispatch `bind` to the model's Value-level op (the kernel constructs the result `Value`).
    fn bind(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        match self {
            VsaDispatch::MapI(m) => m.bind_values(a, b),
            VsaDispatch::Fhrr(m) => m.bind_values(a, b),
            VsaDispatch::Bsc(m) => m.bind_values(a, b),
        }
    }

    /// Dispatch `unbind` (MAP-I/BSC: the self-inverse `Exact` op; FHRR: the regime-gated
    /// `Empirical` op with its δ bound).
    fn unbind(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        match self {
            VsaDispatch::MapI(m) => m.unbind_values(a, b),
            VsaDispatch::Fhrr(m) => m.unbind_values(a, b),
            VsaDispatch::Bsc(m) => m.unbind_values(a, b),
        }
    }

    /// Dispatch `permute` (cyclic shift — `Exact` in every model of the set).
    fn permute(&self, a: &Value, shift: i64) -> Result<Value, VsaError> {
        match self {
            VsaDispatch::MapI(m) => m.permute_value(a, shift),
            VsaDispatch::Fhrr(m) => m.permute_value(a, shift),
            VsaDispatch::Bsc(m) => m.permute_value(a, shift),
        }
    }

    /// **Data-level** unbind for the M-894 reconstruction decode. The record is a *composite*
    /// (typically a bundle), so the Value-level unbind wrappers are the wrong entry: their
    /// `Exact`/regime guards are about tagging a *surfaced* unbind result, while here the noisy
    /// intermediate never surfaces as a `Value` — it flows straight into the cleanup arg-max,
    /// whose own confidence/margin quantify it (FR-S4; exactly `recon.rs::reconstruct_role`'s
    /// kernel posture). Model/dim were already checked by [`vsa_model_of`] + the operand guards.
    fn unbind_data(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        match self {
            VsaDispatch::MapI(m) => m.unbind(a, b),
            VsaDispatch::Fhrr(m) => m.unbind(a, b),
            VsaDispatch::Bsc(m) => m.unbind(a, b),
        }
    }

    /// The cleanup arg-max under this model's similarity (M-894): delegate to the kernel's
    /// [`CleanupMemory::cleanup`], which scores every atom and reports the best hit **with its
    /// confidence and margin** (never a silent nearest-neighbour pick — FR-S4/G2).
    fn cleanup_match(&self, mem: &CleanupMemory, query: &[f64]) -> Option<Match> {
        match self {
            VsaDispatch::MapI(m) => mem.cleanup(query, m),
            VsaDispatch::Fhrr(m) => mem.cleanup(query, m),
            VsaDispatch::Bsc(m) => mem.cleanup(query, m),
        }
    }
}

/// Bind the VSA model a `vsa.*` prim dispatches to off its **first operand's** `Repr::Vsa` model
/// id + dim (the model anchor); the kernel then enforces every other hypervector operand agrees
/// (a model/dim mismatch is an explicit `NotThisModel`/`DimMismatch` refusal, never a coercion).
/// A non-`Vsa` first operand, or a model outside the M-892 introduction dispatch set
/// (MAP-I / FHRR / BSC), is an explicit [`EvalError::PrimType`] naming the model and the set (G2).
fn vsa_model_of(prim: &str, v: &Value) -> Result<VsaDispatch, EvalError> {
    let Repr::Vsa { model, dim, .. } = v.repr() else {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("expected a Vsa hypervector operand, got {:?}", v.repr()),
        });
    };
    match model.as_str() {
        "MAP-I" => Ok(VsaDispatch::MapI(MapI::new(*dim))),
        "FHRR" => Ok(VsaDispatch::Fhrr(Fhrr::new(*dim))),
        "BSC" => Ok(VsaDispatch::Bsc(Bsc::new(*dim))),
        other => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "model {other:?} is outside the vsa.* prim dispatch set at introduction \
                 (MAP-I, FHRR, BSC — M-892); widening the set is an append-only extension, \
                 never a guessed algebra"
            ),
        }),
    }
}

/// Map a kernel [`VsaError`] onto the interpreter's never-silent error surface (see the section
/// note above). Every arm is explicit; nothing is coerced.
fn map_vsa_err(prim: &str, e: VsaError) -> EvalError {
    match e {
        VsaError::Wf(w) => EvalError::Wf(w),
        // Model/dim/alphabet/regime refusals: the kernel's Display already names the offense
        // (expected model, expected/got dims, the offending component index, the failed
        // empirical-profile side-condition, …).
        other => EvalError::PrimType {
            prim: prim.to_owned(),
            why: other.to_string(),
        },
    }
}

/// The Exact-input guard of the section note: a non-`Exact` operand has no defined δ-propagation
/// rule through the VSA algebra, and the kernel would stamp the op's own tag regardless — an
/// unearned upgrade (VR-5). Refused explicitly, never fabricated (the dense group's
/// `ApproximateSource` posture, enforced wrapper-side because the vsa kernel predates the prim
/// surface).
fn vsa_operands_exact(prim: &str, args: &[&Value]) -> Result<(), EvalError> {
    if args
        .iter()
        .all(|v| v.meta().guarantee() == GuaranteeStrength::Exact)
    {
        Ok(())
    } else {
        Err(EvalError::ApproxCompositionUnsupported {
            prim: prim.to_owned(),
        })
    }
}

/// `vsa.bind : (Vsa{m, d}, Vsa{m, d}) → Vsa{m, d}` — model-dispatched binding (M-892). Tag
/// carried from the kernel: **`Exact`** in every model of the introduction set (MAP-I elementwise
/// product on the guarded `±1` alphabet; FHRR phase addition; BSC XOR). A model/dim mismatch or
/// an out-of-alphabet component is an explicit refusal, never a coercion (G2).
fn prim_vsa_bind(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    vsa_operands_exact(prim, args)?;
    let model = vsa_model_of(prim, args[0])?;
    model
        .bind(args[0], args[1])
        .map_err(|e| map_vsa_err(prim, e))
}

/// `vsa.unbind : (Vsa{m, d}, Vsa{m, d}) → Vsa{m, d}` — model-dispatched factor recovery (M-892).
/// Tag carried from the kernel **per model** (VR-5, never re-stamped): MAP-I/BSC **`Exact`** (the
/// self-inverse identities, alphabet-guarded); FHRR **`Empirical`** with its trial-validated
/// `FHRR_UNBIND_PROFILE` δ bound, **regime-gated by the kernel** (the operand must be a single
/// `vsa.fhrr.bind` product — refused explicitly outside the validated regime). The Π-table
/// intrinsic for this prim is the *meet* over the set (`Empirical`); the runtime value's tag is
/// the dispatched model's own.
fn prim_vsa_unbind(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    vsa_operands_exact(prim, args)?;
    let model = vsa_model_of(prim, args[0])?;
    model
        .unbind(args[0], args[1])
        .map_err(|e| map_vsa_err(prim, e))
}

/// `vsa.permute : (Vsa{m, d}, Binary{W}) → Vsa{m, d}` — model-dispatched cyclic shift (M-892).
/// Tag carried from the kernel: **`Exact`** in every model of the set (a pure component
/// rotation). The shift is an unsigned `Binary{W}` magnitude (see the section note: rotation is
/// cyclic, so the inverse permute is the complementary shift `d − s`); a shift width beyond the
/// index space is an explicit refusal, never a truncation (G2).
fn prim_vsa_permute(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    vsa_operands_exact(prim, args)?;
    let model = vsa_model_of(prim, args[0])?;
    let raw = as_index(prim, args[1])?;
    // `rotate` reduces the shift `rem_euclid dim`, but the i64 conversion itself must not wrap:
    // a usize beyond i64::MAX is refused explicitly (unreachable for any realistic width, kept
    // as the never-silent guard — G2).
    let shift = i64::try_from(raw).map_err(|_| EvalError::PrimType {
        prim: prim.to_owned(),
        why: format!("shift amount {raw} exceeds the i64 shift space"),
    })?;
    model
        .permute(args[0], shift)
        .map_err(|e| map_vsa_err(prim, e))
}

/// `vsa.bundle : (Seq{Vsa{m, d}, N≥1}, Float δ) → Vsa{m, d}` — superposition via the **certified
/// path** (M-893): `MapI::bundle_values_certified(items, δ)`, the M-131 checked-instantiation
/// pattern. The kernel issues a **`Proven`** `CapacityBound` (citing Clarkson/Thomas, targeting
/// failure probability δ) **only** under its *checked* side-conditions — `dim ≥ requiredDim(m, δ)`,
/// bipolar (±1) atoms, distinct items — and refuses explicitly otherwise
/// (`InsufficientCapacity` naming the required dim / `NonAlphabetComponent` /
/// `DuplicateBundleItems`), never an unbacked tag (M-I2/VR-5). The wrapper carries the kernel's
/// tag + bound through unchanged; the disclosed bound is the value's own (`Capacity{items, dim}`
/// = this bundle's m and d, with the checked basis in `ProvenThm`), EXPLAIN-able through the
/// model-namespaced `vsa.map_i.bundle` provenance over all input hashes.
///
/// **The dispatch set for bundle is the certified singleton {MAP-I}.** FHRR/BSC are in the M-892
/// *bind-group* dispatch set but their kernel bundles are **`Empirical`-profile ops**
/// (`bundle_values_empirical` — trial-validated δ, not a checked theorem): routing them through
/// this prim would either silently weaken the prim's certified meaning or silently re-tag their
/// evidence (both VR-5 violations), so an FHRR/BSC first element is an explicit refusal naming
/// the certified set — surfacing the empirical bundles is a distinct, append-only extension
/// under its own name (the M-894 lane), never a silent downgrade. An out-of-set model (HRR/…)
/// refuses via [`vsa_model_of`] as for the bind group.
///
/// δ is a `Float` operand and must be a probability in `(0, 1]` — a non-finite or out-of-range δ
/// is refused here with its own message (the kernel's `required_dim` would fold it into an
/// `InsufficientCapacity` whose "required u64::MAX" obscures the actual offense — G2 prefers the
/// named refusal). The Exact-input guard of the section note applies to the operands **and each
/// seq element** (the kernel stamps `Proven` without meeting element strengths — a non-`Exact`
/// element would be a silent upgrade; in particular a bundle result fed back in refuses:
/// no δ-composition rule for nested certified bundles exists).
fn prim_vsa_bundle(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    vsa_operands_exact(prim, args)?;
    let elems = as_seq(prim, args[0])?;
    let Some(first) = elems.first() else {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "bundle requires at least one item — no superposition is defined over an empty \
                  Seq (RFC-0003 §4; refused, never a defaulted value — G2)"
                .to_owned(),
        });
    };
    // Elementwise Exact-input guard (see the doc comment): the seq value's own meta passed the
    // operand guard above, but the *elements* carry their own tags and the kernel stamps the
    // result `Proven` without meeting them (VR-5 — refuse, never fabricate a composed δ).
    if elems
        .iter()
        .any(|v| v.meta().guarantee() != GuaranteeStrength::Exact)
    {
        return Err(EvalError::ApproxCompositionUnsupported {
            prim: prim.to_owned(),
        });
    }
    let delta = as_float(prim, args[1])?;
    if !(delta.is_finite() && delta > 0.0 && delta <= 1.0) {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "target failure probability δ must be a finite Float in (0, 1], got {delta} — \
                 no capacity theorem instantiates outside that range (M-893; refused, never a \
                 defaulted δ — G2)"
            ),
        });
    }
    match vsa_model_of(prim, first)? {
        VsaDispatch::MapI(m) => {
            let refs: Vec<&Value> = elems.iter().collect();
            m.bundle_values_certified(&refs, delta)
                .map_err(|e| map_vsa_err(prim, e))
        }
        VsaDispatch::Fhrr(_) | VsaDispatch::Bsc(_) => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "model {:?} has no certified Value-level bundle — `vsa.bundle` is the certified \
                 path (MAP-I's checked capacity bound) and its dispatch set at introduction is \
                 the certified singleton {{MAP-I}} (M-893); the FHRR/BSC empirical-profile \
                 bundles are a distinct, append-only surfacing under their own name, never a \
                 silent re-tag (VR-5)",
                match first.repr() {
                    Repr::Vsa { model, .. } => model.as_str(),
                    _ => unreachable!("vsa_model_of admitted a Vsa first element"),
                }
            ),
        }),
    }
}

// ── M-894 (`enb` Gap C): `vsa.cleanup` + `vsa.reconstruct` + `vsa.required_dim` ─────────────────
//
// The cleanup-memory retrieval, the RFC-0003 §6 compositional role-reconstruction, and the M-131
// capacity-bound query (FR-S4; RFC-0003 §3/§5/§6; ADR-008). The retrieval prims return a
// **decision triple** `Seq{Float, 3} = [index, confidence, margin]` rather than the recovered
// atom alone: the confidence (match cosine) and margin (gap to the runner-up) are what make an
// approximate retrieval *usable* (FR-S4) — the decision is inspectable, never a silent
// nearest-neighbour pick (G2). The recovered clean atom itself is the caller's `codebook[index]`.
//
// **Tag rule (the RFC-0010 §4.4 precedent + the RFC-0001 §4.7 meet).** The decision procedure is
// an *exhaustive* arg-max over the codebook guarded by an identifiability refusal (a tie —
// `margin ≤ 0` — is an explicit error, never a coin-flip), so the op's own contribution is
// `Exact`, the same claim shape as the RFC-0010 brute-force decode arm. The **query/record slot
// may be non-`Exact`** — cleanup exists precisely to make a noisy unbind usable — and its own
// (strength, bound) pair passes through to the triple via the meet (the M-204 `Passthrough`
// posture: e.g. an FHRR-unbind query's `Empirical` probability bound, or a certified bundle
// record's `Proven` `CapacityBound`, is re-disclosed on the decode — **the disclosed bound is the
// value's own**, never re-derived or dropped). Every *other* operand (codebook atoms, the role,
// the threshold) must be `Exact` — a second non-`Exact` input would make the carried pair
// ambiguous, and no composition rule exists (refused, never fabricated — G2/VR-5).
//
// **Dispatch sets.** `vsa.cleanup` dispatches across the full M-892 set (MAP-I/FHRR/BSC — the
// model only supplies `similarity`; the procedure is model-generic). `vsa.reconstruct` dispatches
// over **{MAP-I, BSC}** — the models whose unbind is `Exact` self-inverse algebra; FHRR's unbind
// tag is `Empirical` and trial-validated only for a single `vsa.fhrr.bind` product (the kernel's
// `FHRR_UNBIND_PROFILE` regime gate), which a reconstruction record is not, so an FHRR
// reconstruct is an explicit refusal naming that ground (never a stretched profile — VR-5);
// surfacing it is an append-only extension under a reconstruction-regime profile of its own.
// The factor decode (`reconstruct_factors`, RFC-0009/RFC-0010) is likewise NOT surfaced here:
// it routes through the RFC-0005 selector whose mandatory EXPLAIN has no prim-surface carrier
// yet — a distinct, append-only surfacing under its own name, never a silent conflation.
//
// **Provenance** is the standard composed shape `Derived{op: hash(prim), inputs}` (over the
// operands as passed), not the kernel's model-namespaced ids: unlike bind/unbind, the decision
// procedure here is model-generic — the dispatched model is recoverable from the operands'
// `Repr::Vsa` model ids in the provenance chain.

/// Extract the hypervector data of a `Vsa` operand whose repr the caller already checked.
fn vsa_hv_data<'a>(prim: &str, v: &'a Value) -> Result<&'a [f64], EvalError> {
    match v.payload() {
        Payload::Hypervector(h) => Ok(h),
        other => Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!("expected a Hypervector payload, got {other:?}"),
        }),
    }
}

/// The codebook operand: a **non-empty** `Seq` of clean atoms, each `Exact` and each of exactly
/// the query/record's model + dim (`like`'s repr). Every violation is an explicit refusal naming
/// the offense (G2): an empty codebook has nothing to clean up against; a non-`Exact` atom has no
/// carry rule (only the query/record slot passes its pair through — see the section note); a
/// cross-model/dim atom is never coerced.
fn vsa_codebook<'a>(prim: &str, like: &Value, seq: &'a Value) -> Result<&'a [Value], EvalError> {
    let elems = as_seq(prim, seq)?;
    if elems.is_empty() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "the codebook must hold at least one atom — an empty cleanup memory has \
                  nothing to clean up against (FR-S4; M-894 — refused, never a defaulted \
                  retrieval; G2)"
                .to_owned(),
        });
    }
    if seq.meta().guarantee() != GuaranteeStrength::Exact
        || elems
            .iter()
            .any(|v| v.meta().guarantee() != GuaranteeStrength::Exact)
    {
        // Only the query/record slot carries a pair through (section note); a non-Exact codebook
        // atom would make the carried pair ambiguous — refuse, never fabricate (G2/VR-5).
        return Err(EvalError::ApproxCompositionUnsupported {
            prim: prim.to_owned(),
        });
    }
    for (i, item) in elems.iter().enumerate() {
        if item.repr() != like.repr() {
            return Err(EvalError::PrimType {
                prim: prim.to_owned(),
                why: format!(
                    "codebook atom {i} is {:?} but the query/record is {:?} — every atom must \
                     share the query's model and dim (M-894 — a model or dim mismatch is an \
                     explicit refusal, never a cross-model coercion; G2)",
                    item.repr(),
                    like.repr()
                ),
            });
        }
    }
    Ok(elems)
}

/// Run the exhaustive cleanup arg-max over `atoms` and apply the identifiability guard
/// (RFC-0010 §4.4: the retrieval is `Exact` only when the best atom is the *unique* arg-max —
/// a tie, `margin ≤ 0`, is an explicit refusal, never a coin-flip between tied atoms).
fn vsa_cleanup_hit(
    prim: &str,
    dispatch: &VsaDispatch,
    dim: u32,
    query: &[f64],
    atoms: &[Value],
) -> Result<Match, EvalError> {
    let mut mem = CleanupMemory::new(dim);
    for (i, atom) in atoms.iter().enumerate() {
        mem.insert(i.to_string(), vsa_hv_data(prim, atom)?.to_vec())
            .map_err(|e| map_vsa_err(prim, e))?;
    }
    let hit = dispatch
        .cleanup_match(&mem, query)
        .ok_or_else(|| EvalError::PrimType {
            // Defensive: unreachable after the non-empty + equal-dim operand guards; kept as the
            // never-silent twin of the kernel's EmptyCodebook (G2).
            prim: prim.to_owned(),
            why: "cleanup memory holds no usable codebook atom for this model/dim".to_owned(),
        })?;
    if hit.margin <= 0.0 {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "retrieval is non-identifiable: the best atom (index {}, similarity {}) is not \
                 the unique arg-max (margin {}) — a coin-flip between tied atoms is exactly what \
                 the Exact decode claim forbids (RFC-0010 §4.4; M-894 — refused, never guessed; \
                 G2)",
                hit.index, hit.confidence, hit.margin
            ),
        });
    }
    Ok(hit)
}

/// Build the `Seq{Float, 3} = [index, confidence, margin]` decision triple, carrying the
/// query/record's own (strength, bound) pair through the RFC-0001 §4.7 meet (the section note's
/// `Passthrough` rule: the op's own contribution is `Exact` and every other operand is guarded
/// `Exact`, so the meet is exactly `carried`'s strength, and the disclosed bound is the value's
/// own). Provenance is `Derived{op: hash(prim), inputs: the operands as passed}`.
fn vsa_decision_triple(
    prim: &str,
    inputs: &[&Value],
    carried: &Value,
    hit: &Match,
) -> Result<Value, EvalError> {
    let strength = carried.meta().guarantee();
    let bound = if strength == GuaranteeStrength::Exact {
        None // M-I1: Exact carries no bound.
    } else {
        carried.meta().bound().cloned()
    };
    let provenance = Provenance::Derived {
        op: operation_hash(prim),
        inputs: inputs.iter().map(|v| v.content_hash()).collect(),
    };
    let index = u32::try_from(hit.index).map_err(|_| EvalError::PrimType {
        // Defensive: a Seq length is a u32, so a codebook index always fits (kept as the
        // never-silent guard — G2).
        prim: prim.to_owned(),
        why: format!("codebook index {} exceeds the u32 index space", hit.index),
    })?;
    let triple_meta = || -> Result<Meta, EvalError> {
        Meta::new(
            provenance.clone(),
            strength,
            bound.clone(),
            None,
            None,
            None,
        )
        .map_err(EvalError::Wf)
    };
    let float_repr = Repr::Float {
        width: FloatWidth::F64,
    };
    let elems: Vec<Value> = [f64::from(index), hit.confidence, hit.margin]
        .into_iter()
        .map(|x| {
            Value::new(float_repr.clone(), Payload::Float(x), triple_meta()?).map_err(EvalError::Wf)
        })
        .collect::<Result<_, _>>()?;
    Value::new(
        Repr::Seq {
            elem: Box::new(float_repr),
            len: 3,
        },
        Payload::Seq(elems),
        triple_meta()?,
    )
    .map_err(EvalError::Wf)
}

/// `vsa.cleanup : (Vsa{m, d}, Seq{Vsa{m, d}, N≥1}) → Seq{Float, 3}` — the cleanup-memory
/// retrieval (M-894; FR-S4): snap the (possibly noisy) query to the nearest codebook atom by the
/// dispatched model's similarity, returning the `[index, confidence, margin]` decision triple.
/// Dispatch set: the full M-892 set (MAP-I/FHRR/BSC — the procedure is model-generic). The
/// query's own (strength, bound) pair carries through; codebook atoms must be `Exact`; a tie is
/// the explicit identifiability refusal (see the section note).
fn prim_vsa_cleanup(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    let query = args[0];
    let dispatch = vsa_model_of(prim, query)?;
    let atoms = vsa_codebook(prim, query, args[1])?;
    let dim = match query.repr() {
        Repr::Vsa { dim, .. } => *dim,
        _ => unreachable!("vsa_model_of admitted a Vsa first operand"),
    };
    let hit = vsa_cleanup_hit(prim, &dispatch, dim, vsa_hv_data(prim, query)?, atoms)?;
    vsa_decision_triple(prim, args, query, &hit)
}

/// `vsa.reconstruct : (Vsa{m, d}, Vsa{m, d}, Seq{Vsa{m, d}, N≥1}, Float) → Seq{Float, 3}` — the
/// RFC-0003 §6 compositional role-reconstruction (M-894; `recon.rs::reconstruct_role` semantics
/// with the manifest's checks made operand-level): unbind the record by the role atom
/// (data-level — the noisy intermediate never surfaces), clean the result up against the
/// codebook, and **refuse explicitly below the caller's threshold** (the manifest's
/// `cleanup_threshold ∈ [0, 1]` made an explicit `Float` operand — a below-threshold retrieval
/// is an error naming confidence vs threshold, never a silent low-quality answer; G2). The
/// record's own (strength, bound) pair carries through — reconstructing from a certified bundle
/// re-discloses its `Proven` `CapacityBound` (the disclosed bound is the value's own). Dispatch
/// set: **{MAP-I, BSC}** — an FHRR record is an explicit refusal (its `Empirical` unbind profile
/// covers only single bind products, not reconstruction records — VR-5; see the section note).
fn prim_vsa_reconstruct(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 4)?;
    let (record, role, threshold_v) = (args[0], args[1], args[3]);
    let dispatch = vsa_model_of(prim, record)?;
    if matches!(dispatch, VsaDispatch::Fhrr(_)) {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "model \"FHRR\" is outside the vsa.reconstruct dispatch set at introduction \
                  ({MAP-I, BSC}) — FHRR's unbind tag is Empirical and trial-validated only for \
                  a single vsa.fhrr.bind product (FHRR_UNBIND_PROFILE), which a reconstruction \
                  record is not; stretching that profile would be an unearned tag (VR-5), so \
                  surfacing FHRR reconstruction is a distinct, append-only extension under a \
                  reconstruction-regime profile of its own (M-894)"
                .to_owned(),
        });
    }
    if role.repr() != record.repr() {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "the role atom is {:?} but the record is {:?} — both must share one model and \
                 dim (M-894 — a model or dim mismatch is an explicit refusal, never a \
                 cross-model coercion; G2)",
                role.repr(),
                record.repr()
            ),
        });
    }
    if role.meta().guarantee() != GuaranteeStrength::Exact
        || threshold_v.meta().guarantee() != GuaranteeStrength::Exact
    {
        // Only the record slot carries a pair through (section note).
        return Err(EvalError::ApproxCompositionUnsupported {
            prim: prim.to_owned(),
        });
    }
    let atoms = vsa_codebook(prim, record, args[2])?;
    let threshold = as_float(prim, threshold_v)?;
    if !(threshold.is_finite() && (0.0..=1.0).contains(&threshold)) {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "the cleanup threshold must be a finite Float in [0, 1], got {threshold} — the \
                 RFC-0003 §6 manifest domain (`ReconInfo`'s cleanup_threshold), made an \
                 explicit operand (M-894; refused, never a defaulted threshold — G2)"
            ),
        });
    }
    let dim = match record.repr() {
        Repr::Vsa { dim, .. } => *dim,
        _ => unreachable!("vsa_model_of admitted a Vsa first operand"),
    };
    let noisy = dispatch
        .unbind_data(vsa_hv_data(prim, record)?, vsa_hv_data(prim, role)?)
        .map_err(|e| map_vsa_err(prim, e))?;
    let hit = vsa_cleanup_hit(prim, &dispatch, dim, &noisy, atoms)?;
    if hit.confidence < threshold {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "cleanup confidence {} is below the threshold {threshold} (best atom: index {}, \
                 margin {}) — a below-threshold retrieval is an explicit refusal, never a \
                 silent low-quality answer (RFC-0003 §6; FR-S4; M-894; G2)",
                hit.confidence, hit.index, hit.margin
            ),
        });
    }
    vsa_decision_triple(prim, args, record, &hit)
}

/// `vsa.required_dim : (Binary{W}, Float) → Binary{64}` — the M-131 capacity-bound query
/// (M-894; RFC-0003 §5): the sufficient dimension `requiredDim(items, δ) = ⌈(2/μ²)·ln(items/δ)⌉`
/// (μ = 0.1 — the cited Clarkson/Thomas instantiation, `mycelium-vsa::capacity`). The result
/// carries the kernel's **`Proven`** `CapacityBound` for exactly this (items, dim, δ)
/// instantiation ([`capacity::proven_capacity_bound`] — the side-condition `dim ≥ requiredDim`
/// holds *by construction* at the returned dim), so the query is inspectable/EXPLAIN-able: the
/// `ProvenThm` basis records the citation, μ, and the checked side-condition. Degenerate inputs
/// — zero items, or δ outside `(0, 1]` — are explicit refusals naming the offense, never the
/// kernel's `u64::MAX` sentinel (G2). A zero `requiredDim` (items = 1, δ = 1) is disclosed as
/// the smallest well-formed dimension 1 — still sufficient (the bound is monotone in dim), and
/// a zero-dimensional hypervector is not a value form.
fn prim_vsa_required_dim(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
    expect_arity(prim, args, 2)?;
    vsa_operands_exact(prim, args)?;
    let items = as_index(prim, args[0])? as u64;
    if items == 0 {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: "requiredDim is asked for zero items — no dimension certifies an empty bundle \
                  (RFC-0003 §5; M-894 — refused, never the u64::MAX sentinel; G2)"
                .to_owned(),
        });
    }
    let delta = as_float(prim, args[1])?;
    if !(delta.is_finite() && delta > 0.0 && delta <= 1.0) {
        return Err(EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "target failure probability δ must be a finite Float in (0, 1], got {delta} — \
                 no capacity theorem instantiates outside that range (M-894; refused, never a \
                 defaulted δ — G2)"
            ),
        });
    }
    // The smallest *well-formed* sufficient dimension (see the doc comment on the zero case).
    let required = capacity::required_dim(items, delta, capacity::MARGIN_MU).max(1);
    let bound = capacity::proven_capacity_bound(items, required, delta).ok_or_else(|| {
        // Defensive: `required ≥ requiredDim` holds by construction after the guards above;
        // kept as the never-silent twin of the kernel's refusal (G2).
        EvalError::PrimType {
            prim: prim.to_owned(),
            why: format!(
                "no Proven capacity bound instantiates at (items {items}, dim {required}, \
                 δ {delta})"
            ),
        }
    })?;
    let bits: Vec<bool> = (0..64u32).rev().map(|i| (required >> i) & 1 == 1).collect();
    let meta = Meta::new(
        Provenance::Derived {
            op: operation_hash(prim),
            inputs: args.iter().map(|v| v.content_hash()).collect(),
        },
        GuaranteeStrength::Proven,
        Some(bound),
        None,
        None,
        None,
    )
    .map_err(EvalError::Wf)?;
    Value::new(Repr::Binary { width: 64 }, Payload::Bits(bits), meta).map_err(EvalError::Wf)
}

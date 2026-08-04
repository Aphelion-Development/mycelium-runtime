//! Typed prim-call registry (RFC-0028 §4.3 sibling; PKG-LINKAGE, mycelium-lang#44).
//!
//! ## Surface law — S-PRIMSIG-SCHEMA / S-TYPED-PRIM-REGISTRY (PKG-LINKAGE)
//!
//! `wild:` (see [`crate::host`]) is the audited, ascription-on-faith escape hatch: `myc check`
//! trusts the caller's declared result type and never verifies a `wild` op's arity or argument
//! types against anything. This module adds a **second, disjoint** door: a `prim:name` dispatch
//! key whose signature — arity, per-argument type, result type, and declared effects — is
//! `myc check`-verifiable against a registered [`PrimSig`], instead of trusted on ascription.
//!
//! This is an **exact structural mirror** of the shipped, spike-resolved
//! `PrimRegistry`/`register_host`/`install_host_ops`/`wild:` pattern in [`crate::host`] (same
//! empty-by-default posture, same loud [`crate::EvalError::UnknownPrim`] miss) — reusing a
//! proven, tested design rather than inventing a second one. "Distinct from the opaque `wild:`
//! table" is satisfied by the separate `prim:` namespace and the separate [`TypedPrimRegistry`]
//! struct (which additionally carries each entry's [`PrimSig`], not just its [`PrimFn`]), not by
//! a second parallel `PrimRegistry`-cloning effort.
//!
//! ## `TySpec` deliberately does **not** reuse `mycelium_l1::Ty`
//!
//! `mycelium-l1` already depends on `mycelium-interp` (its `eval.rs` imports `mycelium_interp`
//! directly — a one-way, non-circular edge). Reusing `mycelium_l1::Ty` here would force
//! `mycelium-interp` to depend on `mycelium-l1`, inverting that existing dependency direction.
//! [`TySpec`] instead mirrors [`mycelium_core::Repr`]'s vocabulary — the shared runtime type-tag
//! every downstream crate already depends on — so `mycelium-l1`'s checker (a downstream
//! consumer) can build its own `Ty <-> TySpec` bridge without interp ever looking upward.
//!
//! ## Scope note (self-hosting leak guard)
//!
//! [`TySpec`] names only representation-level shapes (`Binary`/`Ternary`/`Bytes`/`Bool`/`Unit`/
//! `Float`/`Seq`/`Adt`) — never `mycelium_core::Value`, `Repr` itself, `Box`, `ErrorOp`, or
//! `Emitted` (PKG-LINKAGE's non-goals: those are Rust-only kernel/compiler-internal types, not
//! `.myc`-nameable ones).
//!
//! ## Empty by design until install
//!
//! [`TypedPrimRegistry::empty`] grants **zero** `prim:` ops, exactly like
//! [`crate::prims::PrimRegistry::with_builtins`] grants zero `wild:` ops. An unresolved `prim:`
//! key is a loud, typed miss (never a panic, never a silent no-op) — see the `unknown_typed_prim`
//! test below.
//!
//! ## Not yet wired into `Interpreter::eval`
//!
//! This module ships the schema + registry only (S-PRIMSIG-SCHEMA / S-TYPED-PRIM-REGISTRY).
//! Routing real `Node::Op { prim: "prim:…" }` IR through a live [`TypedPrimRegistry`] at
//! evaluation time is a separate, later surface (`S-TYPED-PRIM-ENV` / `S-TYPED-PRIM-CALL-CHECK`,
//! `mycelium-l1`; `S-CLI-TYPED-PRIM-WIRING`, `mycelium-cli`) — strictly additive over this one,
//! not implied by it. `Interpreter`/`EvalError` are untouched by this file.

use std::collections::BTreeMap;

use mycelium_core::{FloatWidth, GuaranteeStrength};

use crate::prims::PrimFn;

/// Prefix used for typed-prim dispatch keys (`prim:{name}`) — parallel to
/// [`crate::host::WILD_PREFIX`] (`wild:{name}`), and disjoint from it.
pub const PRIM_PREFIX: &str = "prim:";

/// A width argument for [`TySpec::Binary`]/[`TySpec::Ternary`] — mirrors the `width: u32` /
/// `trits: u32` fields of [`mycelium_core::Repr::Binary`]/[`mycelium_core::Repr::Ternary`].
///
/// v0 is monomorphic (a concrete literal only): the risk that a near-generic signature (e.g.
/// `to_json`'s effectively-polymorphic input) may not cleanly monomorphize into one [`TySpec`]
/// is tracked at the package level (PKG-LINKAGE risks) — registering one [`PrimSig`] per
/// concretely-instantiated width actually exercised by a call site, not widening this type, is
/// the proposed mitigation and is out of this surface's scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WidthSpec(pub u32);

/// A checked prim parameter/result type — the frozen S-PRIMSIG-SCHEMA vocabulary. Mirrors
/// [`mycelium_core::Repr`]'s representation-level tags (never `mycelium_l1::Ty` — see the module
/// doc); adds the two nullary logical shapes (`Bool`, `Unit`) and a named applied-data-type slot
/// (`Adt`) a checked prim signature needs to describe a call site, without naming any
/// Rust-only/self-hosting-internal type (`Value`, `Repr` itself, `Box`, `ErrorOp`, `Emitted`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TySpec {
    /// `Binary{n}` — mirrors [`mycelium_core::Repr::Binary`].
    Binary(WidthSpec),
    /// `Ternary{m}` — mirrors [`mycelium_core::Repr::Ternary`].
    Ternary(WidthSpec),
    /// A first-class byte string — mirrors [`mycelium_core::Repr::Bytes`].
    Bytes,
    /// A boolean truth value (the reduce-to-`Bool` result shape; `Repr` has no dedicated `Bool`
    /// tag of its own — comparison prims collapse to `Binary{1}` at the `Repr` layer — but a
    /// checked prim signature names the logical result directly).
    Bool,
    /// The nullary unit type (an effectful prim with no meaningful result, e.g. a write).
    Unit,
    /// A first-class scalar float — mirrors [`mycelium_core::Repr::Float`], INCLUDING its width.
    ///
    /// The frozen surface listed a bare `Float`. `Repr::Float` is width-carrying via the exported
    /// [`FloatWidth`], so carrying it here keeps `TySpec` a faithful mirror — every other tag
    /// already carries its width. To be precise about the benefit: [`FloatWidth`] currently has a
    /// SINGLE variant (`F64`), so a bare tag could not actually confuse two widths today. This is
    /// mirror fidelity and forward-compatibility, NOT a live soundness fix — when a second width
    /// is added, a bare tag would silently accept the wrong one, and changing the shape then would
    /// break every consumer. Corrected now, while there are none.
    Float(FloatWidth),
    /// A first-class indexed homogeneous sequence of `N` elements of the boxed element type —
    /// mirrors [`mycelium_core::Repr::Seq`]'s `{ elem: Box<Repr>, len: u32 }` shape.
    Seq(Box<TySpec>, u32),
    /// A named applied data type (e.g. a checked HTTP response ADT) — a `.myc`-nameable
    /// constructed type, resolved against the declaring crate's own type registry, never a
    /// Rust-internal type name (see the module scope note).
    Adt(String),
}

/// A registered `prim:` call's checked signature (S-PRIMSIG-SCHEMA): arity (`params.len()`),
/// per-argument [`TySpec`], result [`TySpec`], declared effects (free-form, `Vec<String>` —
/// mirrors `FnSig`'s existing effect storage, no grammar change needed to add a new effect name),
/// and the intrinsic [`GuaranteeStrength`] this prim contributes to a result's guarantee meet
/// (RFC-0001 §4.7 — the same intrinsic-tag discipline [`crate::prims::PrimRegistry`]'s built-ins
/// already carry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimSig {
    /// The prim's checked (kernel) name, e.g. `"std.io.serialize.to_json"` — distinct from the
    /// dispatch key, which is this name under the [`PRIM_PREFIX`] namespace.
    ///
    /// OWNED, not `&'static str`. The frozen surface text originally said `&'static str`, which
    /// contradicted its own stated mitigation for near-polymorphic prims: "registering one
    /// [`PrimSig`] per concretely-instantiated width actually exercised by a call site". Names
    /// generated per instantiation cannot be `&'static str` without leaking, so a provider
    /// (`mycelium-std-io`, `mycelium-std-net`) could not have implemented that mitigation at all.
    /// Corrected before any consumer existed — see S-PRIMSIG-SCHEMA's correction note.
    pub name: String,
    /// Per-argument checked types, in order (the length is the prim's arity).
    pub params: Vec<TySpec>,
    /// The checked result type.
    pub ret: TySpec,
    /// Declared effects (e.g. `["net"]`) a caller must cover — never hardcoded by a consumer;
    /// taken from this registered signature (PKG-LINKAGE adversarial-review item).
    pub effects: Vec<String>,
    /// The intrinsic guarantee this prim contributes to a result's guarantee meet.
    pub guarantee: GuaranteeStrength,
}

/// The `prim:` name→(signature, implementation) table `myc check`/`myc run` typed-prim call
/// sites dispatch through — the checked counterpart to [`crate::prims::PrimRegistry`]'s opaque
/// `wild:` table (S-TYPED-PRIM-REGISTRY). Structurally the same shape (a name-keyed map to a
/// [`PrimFn`]), extended to also carry each entry's [`PrimSig`] so a signature can be consulted
/// without invoking the implementation.
#[derive(Clone, Default)]
pub struct TypedPrimRegistry {
    table: BTreeMap<String, (PrimSig, PrimFn)>,
}

impl TypedPrimRegistry {
    /// An empty registry — grants **zero** `prim:` ops (empty-by-design until install, mirroring
    /// [`crate::prims::PrimRegistry::empty`]/`with_builtins`).
    #[must_use]
    pub fn empty() -> Self {
        TypedPrimRegistry {
            table: BTreeMap::new(),
        }
    }

    /// Register (or replace) a typed prim under `prim:{name}` (S-TYPED-PRIM-REGISTRY).
    ///
    /// `name` is the checked (kernel) name without the `prim:` prefix (e.g.
    /// `"std.io.serialize.to_json"`), or already fully-qualified with it — both land under the
    /// same key, mirroring [`crate::prims::PrimRegistry::register_host`]'s bare-or-prefixed
    /// acceptance.
    pub fn register_typed(&mut self, name: &str, sig: PrimSig, f: PrimFn) {
        let key = qualify(name);
        self.table.insert(key, (sig, f));
    }

    /// Look up a typed prim's signature and implementation by name (bare or `prim:`-prefixed).
    #[must_use]
    pub fn get_typed(&self, name: &str) -> Option<(&PrimSig, PrimFn)> {
        self.table.get(&qualify(name)).map(|(sig, f)| (sig, *f))
    }

    /// True if a `prim:{name}` (or fully-qualified `prim:…`) typed prim is registered.
    #[must_use]
    pub fn has_typed(&self, name: &str) -> bool {
        self.table.contains_key(&qualify(name))
    }

    /// Every registered signature, in dispatch-key order — the inspectable surface a checker
    /// (`mycelium-l1`) resolves `use dep::…` imports against.
    pub fn sigs(&self) -> impl Iterator<Item = &PrimSig> {
        self.table.values().map(|(sig, _)| sig)
    }
}

/// Normalize `name` to its fully-qualified `prim:{name}` dispatch key — accepts a bare name or
/// an already-prefixed one, exactly like [`crate::prims::PrimRegistry`]'s `wild:` helpers.
fn qualify(name: &str) -> String {
    if name.starts_with(PRIM_PREFIX) {
        name.to_owned()
    } else {
        format!("{PRIM_PREFIX}{name}")
    }
}

/// Install helper for typed-prim providers (`mycelium-std-io`, `mycelium-std-net`, …) and
/// embedders — the `prim:` counterpart to [`crate::host::install_host_ops`].
///
/// Registers each `(name, sig, f)` under `prim:{name}`. Last registration for a name wins.
pub fn install_typed_prims(reg: &mut TypedPrimRegistry, ops: &[(&str, PrimSig, PrimFn)]) {
    for (name, sig, f) in ops {
        reg.register_typed(name, sig.clone(), *f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EvalError;
    use mycelium_core::{Meta, Payload, Provenance, Repr, Value};

    fn bin1(bit: bool) -> Value {
        Value::new(
            Repr::Binary { width: 1 },
            Payload::Bits(vec![bit]),
            Meta::exact(Provenance::Root),
        )
        .unwrap()
    }

    fn id_sig(name: &str) -> PrimSig {
        PrimSig {
            name: name.to_owned(),
            params: vec![TySpec::Binary(WidthSpec(1))],
            ret: TySpec::Binary(WidthSpec(1)),
            effects: vec![],
            guarantee: GuaranteeStrength::Exact,
        }
    }

    fn typed_id(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
        match args {
            [v] => Ok((*v).clone()),
            _ => Err(EvalError::PrimType {
                prim: prim.to_owned(),
                why: "typed_id expects 1 arg".into(),
            }),
        }
    }

    /// Emulated `prim:` dispatch (mirrors [`crate::wild::dispatch_wild`]'s
    /// `.ok_or_else(EvalError::UnknownPrim)` shape): the checked counterpart to the interpreter's
    /// own `wild:` dispatch, self-contained here since routing live `Node::Op` IR through a
    /// [`TypedPrimRegistry`] is S-TYPED-PRIM-ENV/S-TYPED-PRIM-CALL-CHECK's job (mycelium-l1), not
    /// this surface's.
    fn dispatch_typed(
        reg: &TypedPrimRegistry,
        prim: &str,
        args: &[&Value],
    ) -> Result<Value, EvalError> {
        let (_, f) = reg
            .get_typed(prim)
            .ok_or_else(|| EvalError::UnknownPrim(prim.to_owned()))?;
        f(prim, args)
    }

    #[test]
    fn empty_registry_grants_no_typed_prims() {
        let r = TypedPrimRegistry::empty();
        assert!(!r.has_typed("smoke_id"));
        assert!(!r.has_typed("prim:smoke_id"));
        assert_eq!(r.sigs().count(), 0);
    }

    #[test]
    fn install_typed_prims_registers_prim_prefix() {
        let mut r = TypedPrimRegistry::empty();
        install_typed_prims(&mut r, &[("smoke_id", id_sig("smoke_id"), typed_id)]);
        assert!(r.has_typed("smoke_id"));
        assert!(r.has_typed("prim:smoke_id"));

        let (sig, f) = r.get_typed("prim:smoke_id").expect("registered");
        assert_eq!(sig.name, "smoke_id");
        assert_eq!(sig.params, vec![TySpec::Binary(WidthSpec(1))]);
        assert_eq!(sig.ret, TySpec::Binary(WidthSpec(1)));
        assert!(sig.effects.is_empty());
        assert_eq!(sig.guarantee, GuaranteeStrength::Exact);

        let v = bin1(true);
        let out = f("prim:smoke_id", &[&v]).expect("eval");
        assert_eq!(out, v);

        // Also reachable via bare-name registration/lookup (register_typed accepts bare or
        // prefixed, mirroring register_host).
        let mut b = TypedPrimRegistry::empty();
        b.register_typed("prim:smoke_id", id_sig("smoke_id"), typed_id);
        assert!(b.has_typed("smoke_id") && b.has_typed("prim:smoke_id"));

        // Exactly one signature is visible via sigs().
        assert_eq!(r.sigs().count(), 1);
    }

    /// Unknown `prim:<name>` is a **typed miss** — never silent, never a panic (mirrors
    /// `crate::host`'s `unknown_wild_name_is_typed_miss_loud_fail`).
    #[test]
    fn unknown_typed_prim_is_loud_miss() {
        let r = TypedPrimRegistry::empty();
        let v = bin1(true);
        let err = dispatch_typed(&r, "prim:not_a_registered_typed_prim", &[&v])
            .expect_err("unknown prim: must miss");
        assert!(
            matches!(&err, EvalError::UnknownPrim(p) if p == "prim:not_a_registered_typed_prim"),
            "expected UnknownPrim typed miss, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("not_a_registered_typed_prim"),
            "Display must name the missing prim; got: {msg:?}"
        );

        // A registered TypedPrimRegistry still misses on a *different*, unregistered name — no
        // silent fallback to "any registered prim will do".
        let mut r2 = TypedPrimRegistry::empty();
        install_typed_prims(&mut r2, &[("smoke_id", id_sig("smoke_id"), typed_id)]);
        assert!(matches!(
            dispatch_typed(&r2, "prim:also_unregistered", &[&v]),
            Err(EvalError::UnknownPrim(ref p)) if p == "prim:also_unregistered"
        ));
    }
    /// A provider must be able to register a signature whose name is built AT RUNTIME — the
    /// per-instantiation mitigation [`WidthSpec`]'s doc proposes. This is impossible if `name`
    /// is `&'static str`, so this test pins the owned-name correction.
    #[test]
    fn signature_names_may_be_generated_at_runtime() {
        let mut reg = TypedPrimRegistry::empty();
        for w in [8u32, 16, 32] {
            let generated = format!("std.io.serialize.to_json.binary{w}");
            let sig = PrimSig {
                name: generated.clone(),
                params: vec![TySpec::Binary(WidthSpec(w))],
                ret: TySpec::Bytes,
                effects: vec![],
                guarantee: GuaranteeStrength::Exact,
            };
            reg.register_typed(&generated, sig, typed_id);
        }
        assert!(reg.has_typed("std.io.serialize.to_json.binary8"));
        assert!(reg.has_typed("std.io.serialize.to_json.binary32"));
        assert_eq!(
            reg.sigs().count(),
            3,
            "one signature per instantiated width"
        );
    }

    /// `TySpec::Float` must CARRY its width so it stays a faithful mirror of `Repr::Float` and so
    /// adding a second [`FloatWidth`] variant later is not a breaking change to this shape.
    /// Note: `FloatWidth` has one variant today, so this pins the shape, not a live distinction.
    #[test]
    fn float_tyspec_carries_its_width() {
        let f = TySpec::Float(FloatWidth::F64);
        match f {
            TySpec::Float(w) => assert_eq!(w, FloatWidth::F64, "the width must round-trip"),
            other => panic!("expected Float, got {other:?}"),
        }
    }
}

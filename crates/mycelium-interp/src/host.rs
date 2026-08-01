//! Host / FFI call registry contract (RFC-0028 §4.3).
//!
//! ## Spike resolution (2026-08-01)
//!
//! | Layer | Owns |
//! |-------|------|
//! | **this crate (`mycelium-interp`)** | Dispatch table for `wild:name` via [`PrimRegistry`] |
//! | **`mycelium-std-sys-host`** | `install_default_host_ops(reg)` — OS-backed default table |
//! | **`myc` CLI** | Installs the default table before host-capable evaluation |
//!
//! No separate host crate for v0 (train maturity / multi-repo cost).
//!
//! ## Surface law — S-HOST-REGISTRY (PKG-A1-RECONCILE)
//!
//! **One public surface:** `PrimRegistry` `wild:` via [`PrimRegistry::register_host`] /
//! [`install_host_ops`]. The A1 dual `HostOpRegistry` path (runtime#6 / `dev`) is **rejected** —
//! do not reintroduce a second host-op table or ignore `PrimRegistry::register("wild:…")`.
//! Valuable A1 lessons (empty-by-default, loud unknown-`wild:` miss, fail-closed host floors)
//! land on this mainline only.
//!
//! ## I/O model — blocking-hypha
//!
//! Host ops **may block** the calling OS thread. The hypha scheduler is compute-poll
//! with no I/O reactor. First ports (gha-runner-ctl, tg-agent-relay) are synchronous
//! poll+sleep loops; a reactor is post-S1 work.
//!
//! ## Naming
//!
//! Elaboration lowers `wild { name(args…) }` → `Node::Op { prim: "wild:name" }`.
//! Installers call [`PrimRegistry::register_host`] with the bare `name` (or a
//! fully-qualified `wild:…` key). Prefer `register_host` over plain
//! [`PrimRegistry::register`] for host ops so the `wild:` namespace is explicit.
//!
//! ## Stateful hosts
//!
//! [`PrimFn`] is a pure function pointer (no context). Stateless ops fit directly.
//! Stateful resources (open FDs, HTTP clients) use process-level host context in a
//! follow-up without changing the `wild:` key namespace.
//!
//! ## Empty by design until install
//!
//! [`PrimRegistry::with_builtins`] grants **zero** `wild:` ops. An unresolved
//! host key is [`EvalError::UnknownPrim`] with an explicit capability message (G2).
//! Catalog names (`time_mono_nanos`, `rand_fill`, `process_*`, …) are owned by
//! `mycelium-std-sys-host` / S-HOST-REGISTRY — new wild names need a zipper surface
//! amendment on mycelium-lang first (do not invent A1-only names like `read_capped`
//! here without that PR).

use crate::prims::{PrimFn, PrimRegistry};

/// Documentation alias: the host-call registry **is** the [`PrimRegistry`]'s
/// `wild:` namespace. Prefer this name in host-install code for clarity.
///
/// **Not** a second table: there is no `HostOpRegistry` dual path (A1 / runtime#6 rejected).
pub type HostCallRegistry = PrimRegistry;

/// Prefix used by elaboration for host ops (`wild:{name}`).
pub const WILD_PREFIX: &str = "wild:";

/// Install helper for `mycelium-std-sys-host` and embedders.
///
/// Registers each `(name, f)` under `wild:{name}`. Last registration for a name wins.
pub fn install_host_ops(reg: &mut PrimRegistry, ops: &[(&str, PrimFn)]) {
    for (name, f) in ops {
        reg.register_host(name, *f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EvalError, IdentitySwapEngine, Interpreter};
    use mycelium_core::{Meta, Node, Payload, Provenance, Repr, Value};

    fn bin1(bit: bool) -> Value {
        Value::new(
            Repr::Binary { width: 1 },
            Payload::Bits(vec![bit]),
            Meta::exact(Provenance::Root),
        )
        .unwrap()
    }

    fn host_id(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
        if args.len() != 1 {
            return Err(EvalError::PrimType {
                prim: prim.to_owned(),
                why: "host_id expects 1 arg".into(),
            });
        }
        Ok(args[0].clone())
    }

    /// IR fixture matching L1 elaboration of `wild { name(args…) }`.
    fn wild_op(name: &str, args: Vec<Value>) -> Node {
        Node::Op {
            prim: format!("{WILD_PREFIX}{name}"),
            args: args.into_iter().map(Node::Const).collect(),
        }
    }

    #[test]
    fn default_registry_grants_no_host_ops() {
        let r = PrimRegistry::with_builtins();
        assert!(!r.has_host("fs_read"));
        assert!(!r.has_host("wild:fs_read"));
    }

    #[test]
    fn install_host_ops_registers_wild_prefix() {
        let mut r = PrimRegistry::empty();
        install_host_ops(&mut r, &[("smoke_id", host_id)]);
        assert!(r.has_host("smoke_id"));
        assert!(r.has_host("wild:smoke_id"));
        let v = bin1(true);
        let out =
            r.get("wild:smoke_id").expect("registered")("wild:smoke_id", &[&v]).expect("eval");
        assert_eq!(out, v);
    }

    /// Unknown `wild:<name>` is a **typed miss** — never silent, never panic (A1 lesson on
    /// PrimRegistry mainline; G2).
    #[test]
    fn unknown_wild_name_is_typed_miss_loud_fail() {
        let interp = Interpreter::default();
        let err = interp
            .eval(&wild_op("not_a_registered_host_op", vec![]))
            .expect_err("unknown wild must miss");
        assert!(
            matches!(&err, EvalError::UnknownPrim(p) if p == "wild:not_a_registered_host_op"),
            "expected UnknownPrim typed miss, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("not_a_registered_host_op"),
            "Display must name the missing op; got: {msg:?}"
        );
        assert!(
            msg.contains("host capability")
                || msg.contains("not granted")
                || msg.contains("prim registry"),
            "Display must explain ungranted host capability (loud fail); got: {msg:?}"
        );
        // Dual-path messaging must not reappear (A1 HostOpRegistry migration text rejected).
        assert!(
            !msg.contains("HostOpRegistry"),
            "Display must not direct consumers to the rejected dual HostOpRegistry path; got: {msg:?}"
        );
    }

    /// Default interpreter (builtins only) refuses every wild: as typed miss — pure fragment
    /// safety (ported from A1 `default_interpreter_refuses_all_wild_as_typed_miss`).
    #[test]
    fn default_interpreter_refuses_all_wild_as_typed_miss() {
        let interp = Interpreter::default();
        // Catalog-shaped names + a foreign name — none are granted until install_host_ops.
        for name in [
            "time_mono_nanos",
            "rand_fill",
            "process_spawn",
            "foreign",
            "not_a_registered_host_op",
        ] {
            let err = interp
                .eval(&wild_op(name, vec![]))
                .expect_err("default must refuse wild");
            assert!(
                matches!(err, EvalError::UnknownPrim(ref p) if p == &format!("wild:{name}")),
                "default path: expected UnknownPrim for wild:{name}, got {err:?}"
            );
            let msg = err.to_string();
            assert!(
                msg.contains(name),
                "loud fail must name the op; got: {msg:?}"
            );
        }
    }

    /// `register_host` is the S-HOST-REGISTRY grant path — eval dispatches through PrimRegistry.
    #[test]
    fn register_host_grants_eval_dispatch() {
        let mut prims = PrimRegistry::with_builtins();
        prims.register_host("smoke_id", host_id);
        let interp = Interpreter::new(prims, Box::new(IdentitySwapEngine));
        let v = bin1(true);
        let out = interp
            .eval(&wild_op("smoke_id", vec![v.clone()]))
            .expect("registered host op must evaluate");
        assert_eq!(out, v);
    }

    /// PrimRegistry **is** the host table (not dual): plain `register("wild:…")` still grants
    /// dispatch. A1 ignored this; mainline must not reintroduce that ignore path.
    #[test]
    fn prim_registry_wild_key_dispatches_on_mainline() {
        let mut prims = PrimRegistry::with_builtins();
        prims.register("wild:echo", |p, args| match args {
            [v] => Ok((*v).clone()),
            _ => Err(EvalError::PrimType {
                prim: p.to_owned(),
                why: "echo expects 1 arg".into(),
            }),
        });
        let interp = Interpreter::new(prims, Box::new(IdentitySwapEngine));
        let v = bin1(false);
        let out = interp
            .eval(&wild_op("echo", vec![v.clone()]))
            .expect("PrimRegistry wild: key must grant L0 dispatch on S-HOST-REGISTRY mainline");
        assert_eq!(out, v);
    }

    /// Prefer `register_host` (doc contract): bare name and `wild:`-prefixed name both land
    /// under the same key.
    #[test]
    fn register_host_accepts_bare_or_prefixed_name() {
        let mut a = PrimRegistry::empty();
        a.register_host("smoke_id", host_id);
        let mut b = PrimRegistry::empty();
        b.register_host("wild:smoke_id", host_id);
        assert!(a.has_host("smoke_id") && a.has_host("wild:smoke_id"));
        assert!(b.has_host("smoke_id") && b.has_host("wild:smoke_id"));
        let v = bin1(true);
        let out_a = a.get("wild:smoke_id").expect("a")("wild:smoke_id", &[&v]).expect("a eval");
        let out_b = b.get("wild:smoke_id").expect("b")("wild:smoke_id", &[&v]).expect("b eval");
        assert_eq!(out_a, out_b);
    }

    /// `HostCallRegistry` is a type alias, not a second registry type.
    #[test]
    fn host_call_registry_is_prim_registry_alias() {
        let mut reg: HostCallRegistry = PrimRegistry::empty();
        install_host_ops(&mut reg, &[("smoke_id", host_id)]);
        assert!(reg.has_host("wild:smoke_id"));
        // Same concrete type as PrimRegistry — install mutates the one table.
        let as_prims: &PrimRegistry = &reg;
        assert!(as_prims.has_host("smoke_id"));
    }
}

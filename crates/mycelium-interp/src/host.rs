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
//! fully-qualified `wild:…` key).
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

use crate::prims::{PrimFn, PrimRegistry};

/// Documentation alias: the host-call registry **is** the [`PrimRegistry`]'s
/// `wild:` namespace. Prefer this name in host-install code for clarity.
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

    fn host_id(prim: &str, args: &[&Value]) -> Result<Value, EvalError> {
        if args.len() != 1 {
            return Err(EvalError::PrimType {
                prim: prim.to_owned(),
                why: "host_id expects 1 arg".into(),
            });
        }
        Ok(args[0].clone())
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
}

//! In-crate test modules for `mycelium-interp` (CLAUDE.md test-layout rule).
//! White-box access via `use crate::…::*`; logic files carry no `#[cfg(test)]` inline code.

mod guard_hole_census;
mod parallel;
mod prims;
mod wild;
mod with_depth_parity;

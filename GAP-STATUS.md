# Gap status — `mycelium-runtime`

| Field | Value |
|-------|--------|
| **Component** | `mycelium-runtime` |
| **Role** | Scheduler, interp, cert, diag, rt-abi |
| **Date** | 2026-07-29 |
| **Umbrella roadmap** | [EXPRESSIBILITY-GAPS.md](https://github.com/tzervas/mycelium-lang/blob/docs/expressibility-gap-roadmap-2026-07-29/docs/EXPRESSIBILITY-GAPS.md) (branch; merges to `mycelium-lang` `docs/EXPRESSIBILITY-GAPS.md`) |
| **Tracking epic** | https://github.com/tzervas/mycelium-lang/issues/27 |
| **Port readiness plan** | [PORT-READINESS-2026-07-22.md](https://github.com/tzervas/mycelium-lang/blob/main/docs/planning/PORT-READINESS-2026-07-22.md) |
| **Native twin** | `tzervas/mycelium-runtime-myc` under [mycelium-lang-myc](https://github.com/tzervas/mycelium-lang-myc) |
| **Local gap notes** | none |
| **Related issues** | Interp hosts wild: registry once #16 lands |

## One-line status

This repo is a **Rust train** component (transitional until `.myc` port is Empirical/stable).
Expressibility, host-effect, and self-host staging are coordinated at the **umbrella** —
do not fork a second roadmap here.

## What to read next

1. Umbrella [EXPRESSIBILITY-GAPS.md](https://github.com/tzervas/mycelium-lang/blob/docs/expressibility-gap-roadmap-2026-07-29/docs/EXPRESSIBILITY-GAPS.md) — full gap map + work packages.
2. Local gap note(s) if listed above.
3. Component [README.md](./README.md) honesty banner (extract / Declared-Empirical tags).

## Honesty

- Pins and draw-in gates live in `tzervas/mycelium-lang` `components.lock`.
- Closing a language-completeness gap requires monorepo/design criteria **and** green
  component draw-in — see umbrella `docs/COMPONENT_READINESS.md`.

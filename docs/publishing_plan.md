# Publishing `jsono` as a DuckDB community extension — decision spike

> **Status: research + decision document, not a decision.** This spike records
> the submission mechanics, inventories what publishing would freeze, and
> proposes a release scheme. Every open choice is phrased as a question for the
> maintainer in [Step 4](#step-4-decision-checklist-for-the-maintainer). No
> submission PR, git tag, or code change is part of this spike.
>
> Publishing freezes surface that is still being actively designed. This
> document is best acted on **after** the in-flight hardening/bugfix plans
> (039–042) land — see question (c). Re-run [Step 1](#step-1-submission-mechanics)
> right before any submission; the community repo's process moves.

## Context

The repository is public, but the README tells every consumer to build from
source (`README.md` Installation section: "not yet published in DuckDB's
extension repository … Build it from source"). The downstream rick/data
pipeline compiles it, and its integration work is explicitly held "until a
version bump" of this extension — so downstream is already waiting on a
versioning/release story, independent of any community publication.

The missing pieces are process, not build capability: a version/tagging
scheme, a stability statement for a format that is strict-versioned and fails
loud on mismatch, and the community-extensions submission mechanics. This
document supplies the mechanics and lays out the freeze so the maintainer can
decide.

**STOP-condition check (required by the plan):** community-extensions is
actively accepting submissions — 281 extensions are listed and PRs were merged
on the day of writing — and the build model still matches this document's
premise (a centrally-driven build via `extension-ci-tools` against the latest
stable DuckDB). No STOP condition triggered.

---

## Step 1: Submission mechanics

Sources:

- Overview / distribution model: <https://duckdb.org/community_extensions/>
- How to contribute: <https://duckdb.org/community_extensions/development.html>
- Repository, descriptor examples, update guide:
  <https://github.com/duckdb/community-extensions> — in particular its
  `UPDATING.md`, the `extensions/<name>/description.yml` descriptors, and
  `scripts/build.py`.
- Build toolchain: <https://github.com/duckdb/extension-ci-tools>
- Extension template this repo is based on:
  <https://github.com/duckdb/extension-template>

### Distribution model

Community extensions follow a "code in your repo, build and distribution
centralized" model, explicitly compared to a package manager like Homebrew
(<https://duckdb.org/community_extensions/>). The maintainer keeps the source
in their own repository; the DuckDB community-extensions CI checks out a pinned
commit, builds it with the shared toolchain, signs the binaries, and serves
them. End users then install without the `-unsigned` flag the README currently
requires:

```sql
INSTALL jsono FROM community;
LOAD jsono;
```

### The descriptor file

Submission is a pull request to `duckdb/community-extensions` adding one file:
`extensions/<name>/description.yml`
(<https://duckdb.org/community_extensions/development.html>). The directory name
**must** equal `extension.name` — `scripts/build.py` fails the build otherwise
("Directory name … does not match extension name"). Fields observed across real
descriptors and `scripts/build.py`:

```yaml
extension:
  name: jsono                # required; must match the directory name; must be unique across the repo
  description: <one line>    # required; shown in the extension list
  version: <string>          # required; free-form (semver "1.5.4", date "2025120401", "0.0.1" all appear)
  language: C++              # required
  build: cmake              # required for this repo's build system
  license: MIT              # required
  maintainers:              # required; GitHub handles
    - <handle>
  excluded_platforms: "..." # optional; ";"-separated platforms to skip
  opt_in_platforms: "..."   # optional; platforms built only on request
  requires_toolchains: "..."# optional; e.g. "rust", "fortran;omp" — NOT needed here
  vcpkg_url / vcpkg_commit  # optional; pin a vcpkg for third-party deps — NOT needed here
  custom_toolchain_script   # optional
  test_config               # optional

repo:
  github: <owner>/<repo>    # required; the source repository
  ref: <commit-sha>         # required; the exact commit built for the latest stable DuckDB
  ref_next: <commit-sha>    # optional; commit built against the UPCOMING DuckDB during a transition
  canonical_name: <...>     # optional

docs:
  hello_world: |            # optional but expected; a runnable example
    SELECT ...
  extended_description: |   # optional; markdown shown on the extension page
    ...
```

The `version` string is descriptive metadata only — it is not parsed for
ordering. `ref` is what actually pins the build (see the release scheme in
[Step 3](#step-3-release-scheme-proposal-not-a-decision)).

### How the build is executed and which DuckDB version it targets

Key nuance (from `scripts/build.py` and the community `UPDATING.md`): **the
community repo — not the descriptor — chooses the DuckDB version.** The
descriptor supplies only `repo.github` + `repo.ref`; the community CI checks out
that commit and builds it against whatever DuckDB version the community pipeline
is currently targeting (the latest stable release), using the same
`extension-ci-tools` reusable workflow the extension template ships. The
`duckdb_version` in this repo's own
`.github/workflows/MainDistributionPipeline.yml` governs only the maintainer's
**own** CI verification, not the community distribution build.

Two rebuild triggers exist (community `UPDATING.md`):

1. **Descriptor update** — whenever the `description.yml` PR is merged (a new
   `ref`), the extension is rebuilt and released against the latest stable
   DuckDB.
2. **New DuckDB release** — on every DuckDB release, **all** community
   extensions are rebuilt, so they are available the moment the release ships.

### Platforms

The community CI builds "all supported DuckDB platforms including Linux, MacOS,
Windows and Wasm" (<https://duckdb.org/community_extensions/development.html>).
The full platform set the toolchain knows is
`linux_amd64; linux_arm64; linux_amd64_musl; osx_amd64; osx_arm64;
windows_amd64; windows_amd64_mingw; wasm_eh; wasm_mvp; wasm_threads`. A
descriptor opts out per-platform via `excluded_platforms`. This is a material
difference from the repo's current CI, which builds **only** `linux_amd64`
(arm64/osx are excluded for runner-billing reasons on a private-style repo;
Windows/Wasm are never built) — see the freeze table and question (e).

### Re-pinning on each DuckDB release

The community `UPDATING.md` describes the maintainer's obligations:

- If the extension compiles against both current and upcoming DuckDB
  (`duckdb-stable-build` and `duckdb-next-build` both green), no action is
  needed — the automatic rebuild on release carries it forward.
- If the upcoming DuckDB needs source changes (only `duckdb-stable-build`
  green), the maintainer creates a `vX.Y-<codename>` branch fixed against the
  upcoming release and points the descriptor's `repo.ref_next` at it **before**
  the release, so it is ready on release day. After a release, fixes go on
  `main` and the descriptor `ref` is bumped as normal.

This obligation follows directly from `docs/UPDATING.md` in this repo: the
extension builds against DuckDB's **internal, unstable** C++ API, so a DuckDB
minor can require code changes, not just a rebuild.

### Policy constraints

- **License** — a `license` field is mandatory; this repo is MIT, which is
  accepted (MIT/Apache-2.0 are the common cases in existing descriptors).
- **Naming** — snake_case, unique across the repo, directory-name must match.
  `jsono` is **available** (no `extensions/jsono` today; nearest existing names
  are `cityjson`, `json_schema`, `jsonata`).
- **Maintenance** — the model presumes an active maintainer who keeps the
  extension compiling across DuckDB releases (the `ref_next` flow). There is no
  formal SLA, but an extension that stops compiling drops out of the
  auto-rebuild on the next DuckDB release.

---

## Step 2: What publishing freezes

Publishing turns "reserve the right to change" (CLAUDE.md: binary layout and
SQL API "не стабилизированы") into a public contract with third parties who do
not read this repo. The table inventories each contract, today's stance, and
what freezing it costs. **Severity here is design/operational, not a bug list.**

| Contract | Today's status | What publishing freezes | Freeze cost / open policy |
|---|---|---|---|
| **Binary format V4** (`jsono::VERSION = 0x04`; readers reject other versions loudly; exposed by `jsono_version()`) | Strict-versioned, fail-loud on mismatch. Repo stance: "backward compat not a constraint." | Third parties will persist `jsono` columns in their own DuckDB/Parquet/DuckLake files. A format bump makes a newer extension reject their stored data loudly. | The current "may change freely" stance is **incompatible with published storage** unless the docs state a policy. Must choose: storage is version-locked (re-write/migrate on upgrade), or old versions stay readable. See question (b). |
| **SQL function surface** (names + signatures of `jsono`, `to_json`, `jsono_transform`, `->`/`->>`, the merge/diff/group aggregates, shredding constructor, introspection) | CLAUDE.md reserves the right to rename/re-signature for a "more correct model." README already tiers the introspection helpers as "less stable." | User queries and downstream code (rick/data) bind to these names. A rename is a breaking change for every consumer. | Decide a stability tier: which functions are stable vs. explicitly-unstable. A `v0.x` tag can declare the whole surface unstable (question a). |
| **Bare-STRUCT layout recognition** (`jsono_storage_type()` DDL; structural — no logical type alias) | Deliberate design: portable across Parquet/DuckLake because there is no user-defined type to strip. | User table DDL and stored columns are recognized purely by the physical struct shape. Changing that shape unbinds existing columns. | This is a *strength* to preserve (portability), but it also means the physical shape itself is a public contract, not just the SQL names. Any layout change is a storage-format change (row 1). |
| **DuckDB version coupling** (internal C++ API build → a rebuild, possibly code changes, per DuckDB minor — `docs/UPDATING.md`) | Handled ad-hoc per bump today (single maintainer, no external consumers waiting on a schedule). | Auto-rebuild on each DuckDB release carries the extension forward **only if it still compiles**; otherwise the `ref_next` flow must land within the ~2-week feature-freeze window or the extension misses the release. | A maintenance-cadence commitment (question d). The upside: the community CI builds arm64/osx/etc. on its own runners, removing this repo's current billing constraint. |
| **Platform matrix** | Repo CI builds `linux_amd64` only; Windows/Wasm never built, arm64/osx excluded for billing. | Community CI attempts **all** platforms unless `excluded_platforms` lists them; a platform that fails to compile fails the submission. | Must verify the extension compiles on osx_arm64, linux_arm64, and decide whether to attempt or exclude Windows/Wasm (never built here). See question (e). Note the macOS-specific recursion-depth limit already in the code (512 vs 1000) shows platform behavior is not uniform. |
| **`jsono_version()` / format-version introspection** | Public function returning `4`. | Consumers may branch on it. | Low cost, but it becomes the sanctioned way to gate on format compatibility — tie it to the release scheme (Step 3). |

---

## Step 3: Release scheme proposal (not a decision)

A concrete proposal to react to, not a chosen path. It has **standalone value
before any community submission**: rick/data's "held until version bump" items
unblock at the **first tag**, because a tag gives downstream a stable commit to
pin and build from source — community publication is a separate, later step.

### Tagging

- Adopt semver with a leading `v0.` to signal instability: `v0.1.0`,
  `v0.2.0`, … . Under semver, `0.x` explicitly means "anything may change,"
  which matches the current CLAUDE.md stance and lets the surface keep moving
  while still giving consumers a stable pin. Reserve `v1.0.0` for the point the
  maintainer is ready to freeze the surface (question a).
- Each tag is the unit the community descriptor pins: a `vX.Y.Z` git tag →
  `repo.ref` = that tag's commit → the descriptor `version` string mirrors it.
- Bump policy tied to the two frozen contracts: a **format bump**
  (`jsono::VERSION`) or a SQL-surface breaking change is at least a minor bump
  while in `0.x`, and would be a major bump post-`1.0`. `jsono_version()` and
  the tag move together.

### Repo additions (minimal)

- **`CHANGELOG.md`** — one entry per tag recording: SQL-surface changes,
  whether `jsono::VERSION` changed, and (if so) the storage-migration note
  (question b). This is the only genuinely new artifact required.
- **No new deploy leg is required for community publication.** The community CI
  builds and signs centrally from `repo.ref`; the existing
  `MainDistributionPipeline.yml` already produces verification artifacts. The
  manual `scripts/extension-upload.sh` (S3-style signed upload, not wired into
  CI today) stays a *separate* channel — either retire it or document it as the
  non-community distribution path (a follow-up, not this spike).
- Optionally add a `duckdb-next-build` leg (against DuckDB `main`) to
  `MainDistributionPipeline.yml`, matching the extension template, so the
  maintainer gets early warning of upcoming-DuckDB breakage (feeds the
  `ref_next` flow in Step 1). This is a maintainer convenience, not a
  submission requirement.

### Sequencing note

Because tagging alone unblocks downstream, the maintainer can decouple:
**(1) tag now** to unblock rick/data and give a stable build-from-source pin,
then **(2) submit to community** later once the freeze questions below are
answered and the platform matrix is verified.

---

## Step 4: Decision checklist for the maintainer

Every item is a question with options. None is decided here.

**(a) Freeze the SQL surface now, or tag `v0.x-unstable` first?**
- Option A1 — Tag `v0.x` and publish with an explicit "surface is unstable,
  may change between `0.x` releases" note (keeps CLAUDE.md's current freedom;
  matches how many community extensions ship). Freeze at `v1.0.0` later.
- Option A2 — Declare the core surface (`jsono`, `to_json`, `jsono_transform`,
  `->`/`->>`) stable now and freeze it, keeping only the introspection helpers
  as explicitly-unstable (the README already tiers them this way).
- *Prerequisite for A2:* willingness to treat a rename as a breaking change
  from day one.

**(b) Storage-compatibility statement — what does a format bump promise?**
- Option B1 — **Version-locked storage:** "stored `jsono` data is tied to the
  format version; upgrading the extension across a format bump requires
  re-writing the data." The migration story is plan 037's reshred / re-ingest
  recipe. Simplest; matches today's fail-loud reader.
- Option B2 — **Reader back-compat:** commit to reading older format versions
  in newer extensions. Larger, ongoing engineering cost; contradicts the
  current "backward compat not a constraint" stance.
- Whichever is chosen must be **stated in the README and CHANGELOG** before
  publishing — silence here is the actual risk (a consumer's stored data
  becoming unreadable with no documented recourse).

**(c) Publish now, or after plans 039–042 land?**
- Option C1 — Publish now (tag + descriptor), iterate in public under `0.x`.
- Option C2 — Land the in-flight bugfix + hardening plans (039–042) first, so
  the first published build is a stronger first impression, then tag + submit.
- The plan header notes this doc "is better acted on after 039–043 land"; C2 is
  the conservative reading, C1 trades polish for unblocking downstream sooner
  (though (a)/(b) show tagging already unblocks downstream without publishing).

**(d) Maintenance-cadence commitment.**
- The model asks for a build per DuckDB minor (`docs/UPDATING.md` + the
  community `ref_next` flow within the ~2-week feature-freeze window). Is the
  maintainer willing to commit to that cadence? If not reliably, the extension
  will still auto-rebuild while it *happens* to compile, but will silently drop
  out of a release it can't compile against until fixed — acceptable or not?

**(e) Platform matrix — which platforms does the descriptor build?**
- The community CI attempts all platforms by default; this repo has only ever
  built `linux_amd64`. Decide `excluded_platforms`:
  - Verify the extension **compiles** on `osx_arm64` and `linux_arm64` (likely
    fine — no third-party toolchain, empty `vcpkg.json`).
  - Decide whether to attempt `windows_amd64` and the three Wasm targets
    (never built here) or list them in `excluded_platforms` for the first
    submission and add them later.
- Note the upside: publishing removes this repo's runner-billing constraint —
  arm64/osx are built on the community's runners at no cost to the maintainer.
- *Prerequisite:* a compile check on the non-`linux_amd64` targets before the
  descriptor claims them.

**(f) Descriptor metadata — hygiene decisions.**
- `LICENSE` currently carries "Copyright 2018-2025 Stichting DuckDB Foundation"
  (template residue). MIT is satisfied either way, but should the copyright line
  name the actual author before publishing? (Hygiene, not a blocker.)
- Confirm `maintainers` GitHub handle(s), the one-line `description`, and the
  `docs.hello_world` example for the extension's public page.

---

## If the answer is GO

The follow-up **implementation** plan (out of scope for this spike) would
cover: adding `CHANGELOG.md`; cutting the first tag; opening the
`description.yml` PR against `duckdb/community-extensions`; rewriting the
README Installation section from build-from-source to
`INSTALL jsono FROM community`; deciding `excluded_platforms` after a compile
check; and retiring or re-documenting `scripts/extension-upload.sh`.
Re-run [Step 1](#step-1-submission-mechanics)'s fact-check immediately before
submitting — the community repo's process changes over time.

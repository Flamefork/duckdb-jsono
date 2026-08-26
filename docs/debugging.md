# Hunting a bug the suite cannot see

The SQLLogic suite answers "is the behaviour right on this machine". These recipes answer the
harder question — "is the behaviour right on a machine we do not have" — and each catches a class
none of the others do. They are recipes, not gates: only the sanitizer legs in
`.github/workflows/Sanitizer.yml` run automatically.

Run each against a build of the current tree; every one of them was worth its runtime at least once.

## AddressSanitizer + UndefinedBehaviorSanitizer

```bash
uv run --frozen make debug && ./build/debug/test/unittest "test/*"
```

Use-after-free, out-of-bounds on an *allocation*, and undefined behaviour. This is the leg CI runs
on every push. It does **not** see reads of uninitialised memory, data races, or an index past a
container's `size()` that stays inside its `capacity()`.

## ThreadSanitizer

```bash
THREADSAN=1 uv run --frozen make debug && TSAN_OPTIONS=halt_on_error=0 ./build/debug/test/unittest "test/*"
```

Data races — the class the parallel aggregate paths (`group_merge`, `collect`, the advisor) can
regress into, and the one no other tool here reports. Runs weekly in CI.

ASan and TSan cannot coexist; DuckDB's CMake drops ASan when `THREADSAN=1`. `halt_on_error=0`
reports every race in one pass instead of aborting on the first.

**Do not pipe the run through `tail`.** The pipeline then reports `tail`'s exit status, so a crashed
binary reads as a pass — the first TSan run of this repository "passed" that way while actually
dying two thirds of the way through.

## libc++ hardening (macOS)

```bash
rm -rf build/debug
EXT_DEBUG_FLAGS='-DCMAKE_CXX_FLAGS=-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG' \
  uv run --frozen make debug && ./build/debug/test/unittest "test/*"
```

Container bounds against `size()` (not just the allocation), iterator invalidation, and — the
reason to reach for it — **strict weak ordering validation inside `std::sort`**. A comparator that
is not a strict weak ordering sorts differently under different standard libraries and can run off
the end of the range; this turns that from a platform mystery into a message.

The mode is ABI-affecting, so it must reach every translation unit: pass it through
`EXT_DEBUG_FLAGS` (which lands in `CMAKE_CXX_FLAGS` for the whole build including DuckDB), not
through `CXXFLAGS` on an already-configured build directory, and wipe `build/debug` first.

The Linux CI legs build clang against **libstdc++**, where this macro is a silent no-op. The
equivalents there are `_GLIBCXX_ASSERTIONS` (bounds only, ABI-neutral) and `_GLIBCXX_DEBUG` (full,
ABI-affecting). Neither is wired into CI: they cannot be verified from a macOS checkout, and a
sanitizer job that goes red for a reason nobody can reproduce is worse than one check fewer.

## Guard-page allocator (macOS)

```bash
DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib ./build/release/test/unittest "test/sql/<one>.test"
```

One guard page per allocation, so an out-of-bounds read or a use-after-free traps on the
instruction that does it rather than on the corruption it causes later. Tens to hundreds of times
slower — point it at a single suspect file, never the suite.

A cheaper, coarser version of the same idea needs no rebuild and no library:

```bash
MallocScribble=1 MallocPreScribble=1 MallocGuardEdges=1 ./build/release/test/unittest "test/*"
```

Freed memory is filled with `0x55` and fresh memory with `0xAA`, which turns "the freed bytes
happened to still look right" — the reason a dangling read can be invisible on one platform and
fatal on another — into a wrong answer here.

## Uninitialised reads

Neither ASan nor the hardening modes detect a *read* of uninitialised memory; that needs
MemorySanitizer (which requires every dependency, including the standard library, rebuilt
instrumented) or Valgrind's memcheck under Linux. The weekly and on-demand `memcheck` job runs the
optimizer, group-merge, collect, and advisor SQLLogic files under Valgrind. Run the same focused
selection locally on Linux with `uv run --frozen make memcheck`. This is not whole-suite coverage;
new high-risk aggregate or optimizer files must be added to the target explicitly.

## Compiler-conditional code

A `#if` that changes a *computed* value is a platform bug waiting for a platform. If the value ends
up in persisted bytes, the format itself becomes compiler-dependent — which is what happened to
`HashMix64`, whose no-`__int128` fold dropped a carry and stored a different `shape_hash` under
MSVC. The suite cannot catch this: every document it covers stayed inside the operand range where
both folds agree.

The guard that does work is a `static_assert` comparing the branch this build does not take against
the one it does (see `HashMix64Portable` in `src/include/jsono.hpp`). Any new conditional on that
path owes one.

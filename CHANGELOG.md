# Changelog

All notable changes to this extension are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The `JSONO` binary format and SQL API are still being designed and are not
stabilized: storage layout, function names, and return contracts may change
between revisions without a compatibility shim.

## [Unreleased]

### Added

- Hypothesis-based property and fuzz harness (`test/property/jsono_property.py`):
  round-trip idempotency, value parity against the input JSON, object-key sort
  order, and no-crash fuzzing of corrupted JSON text and of arbitrary four-BLOB
  `JSONO` structs.
- `Sanitizer` CI workflow that builds with AddressSanitizer + UndefinedBehavior
  Sanitizer and runs the SQLLogic suite and the property harness against that
  instrumented build.

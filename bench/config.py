from pathlib import Path

BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
PROFILES_DIR = RESULTS_DIR / "profiles"
PROJECT_ROOT = BENCH_DIR.parent

JSONO_EXTENSION_PATH = PROJECT_ROOT / "build" / "release" / "extension" / "jsono" / "jsono.duckdb_extension"
FIELD_SAMPLE_DATA_DIR = DATA_DIR / "field_sample"
FIELD_SAMPLE_EVENTS_NESTED_PATH = FIELD_SAMPLE_DATA_DIR / "events_nested.parquet"
FIELD_SAMPLE_JSONO_EVENTS_PATH = DATA_DIR / "field_sample" / "events_nested_jsono.parquet"
FIELD_SAMPLE_RULES_PATH = FIELD_SAMPLE_DATA_DIR / "rules.json"
FIELD_SAMPLE_ROW_COUNT = 245_760

# Local retail-shaped sample: nested event_properties (root ym:pv fields + nested
# params), wide objects, deep nesting, null/empty/array mix, clientID group sizes.
# Gitignored, local-only.
RETAIL_SAMPLE_EVENTS_PATH = DATA_DIR / "field_sample" / "retail_sample_events.parquet"
RETAIL_SAMPLE_ROW_COUNT = 245_760
RETAIL_SAMPLE_PARSE_COPY_PATH = RESULTS_DIR / "retail_sample_parse_copy.parquet"
# Top-level scalar keys present in the retail sample event_properties (correct casing).
EXTRACT_RETAIL_SAMPLE_SPEC = {
    "URL": "VARCHAR",
    "UTMSource": "VARCHAR",
    "deviceCategory": "VARCHAR",
    "clientID": "VARCHAR",
}

# Deep-nested retail-shaped ecommerce events (root + nested params + nested
# `ecommerce` arrays of products, avg ~2.7KB). Covers the array/deep-nesting
# dimension the page_view sample lacks. Gitignored, local-only.
RETAIL_SAMPLE_ECOMMERCE_PATH = DATA_DIR / "field_sample" / "retail_sample_ecommerce.parquet"
RETAIL_SAMPLE_ECOMMERCE_ROW_COUNT = 65_987
RETAIL_SAMPLE_ECOMMERCE_PARSE_COPY_PATH = RESULTS_DIR / "retail_sample_ecommerce_parse_copy.parquet"
# Scalar paths into the ecommerce shape (one reaches into the nested array head).
EXTRACT_RETAIL_SAMPLE_ECOMMERCE_SPEC = {
    "URL": "VARCHAR",
    "deviceCategory": "VARCHAR",
    "currency": {"type": "VARCHAR", "path": "$.ecommerce[0].currencyCode"},
}
FIELD_SAMPLE_JSONO_ROW_GROUP_SIZE = 32_768
FIELD_SAMPLE_JSONO_COMPRESSION = "zstd"
FIELD_SAMPLE_JSONO_COMPRESSION_LEVEL = 8

# v4: results carry `row_count` so scaling reports can distinguish real parallel
# regressions from small-case overhead.
SCHEMA_VERSION = 4
DEFAULT_RUNS = 5
DEFAULT_PROFILE_RUNS = 1
DEFAULT_TOLERANCE_PCT = 10.0
DEFAULT_MIN_EFFECT_MS = 1.0

SIZES = {
    "1k": 1_000,
    "10k": 10_000,
    "100k": 100_000,
}

URL_SIZES = {
    "1k": 1_000,
    "10k": 10_000,
    "100k": 100_000,
}
URL_DEFAULT_SIZES = ["1k", "10k", "100k"]

# Number-heavy dataset. field_sample event_properties is ~100% small integers
# (int60) and exercises none of the int64/uint64/DEC60/NUMBER paths the format
# rework adds. This synthetic dataset exists solely to cover that gap; it is not
# the primary performance proof. Each row carries one value of every numeric
# class so parse and parse_copy (storage size) both exercise the new ladder.
NUMBERS_SIZES = {
    "100k": 100_000,
}
NUMBERS_PARSE_COPY_PATH = RESULTS_DIR / "numbers_parse_copy.parquet"

# Wide-flat page_view-class dataset: one schema-stable object of ~100 scalar
# top-level keys, no nesting. Merge-family performance is shape-dependent
# (wide-flat sorted-key merge vs deep-nested descent), so merge scenarios need
# this shape represented next to the deep-nested retail sample.
WIDE_FLAT_SIZES = {
    "100k": 100_000,
}

# Array-heavy ecommerce dataset: every row is a purchase event whose `products` array carries
# several item objects (~10 fields each), mirroring the retail_sample_ecommerce shape that
# jsono_entries `array_style` is built for. Exists so the indexed_elements (per-element explosion)
# vs whole_json (one leaf per array) comparison has an array-dominated local dataset — the
# 4%-array synthetic events file is too diluted to show the difference. Generate explicitly
# (outside `--kind all`): `uv run python bench/generate_data.py --kind ecom --size 100k`.
ECOM_SIZES = {
    "10k": 10_000,
    "100k": 100_000,
}

# products shredded into a LIST<STRUCT> lane — the production rick/data event_properties shape
# (the array is physically shredded, not residual). entries whole_json over this exercises the
# reconstruct-to-plain read path, the cost the plain ecom dataset does not cover.
ECOM_PRODUCTS_SHRED = {
    "$.products": (
        "STRUCT(item_id VARCHAR, item_name VARCHAR, item_brand VARCHAR, "
        "item_category VARCHAR, item_variant VARCHAR, affiliation VARCHAR, "
        'price DOUBLE, quantity BIGINT, discount DOUBLE, "position" BIGINT)[]'
    )
}

# Events4 keyed-aggregate proxy. Kept outside `--kind all` generation because it
# is deliberately larger than the routine synthetic datasets.
KEYED_PAIR_SIZES = {
    "1M": 1_000_000,
}
KEYED_PAIR_ROW_GROUP_SIZE = 32_768
KEYED_PAIR_PAGE_GROUPS = 86_207
KEYED_PAIR_USER_GROUPS = 263_000

# Transaction-snapshot diff dataset: per-transaction ordered CUMULATIVE snapshots — the
# rick/data transaction_webhook shape jsono_diff consumes. Each successive snapshot of a
# transaction is the prior one with a few mutated scalar leaves plus a small churning `items`
# array, so jsono_diff(lag(snapshot) OVER (PARTITION BY transaction_id ORDER BY seq), snapshot)
# yields a realistic small delta. Long-tail snapshots-per-transaction (~16 avg, tail to ~112)
# mirrors the keyed_pair group distribution. Kept outside `--kind all` (generate explicitly,
# like keyed_pair): `uv run python bench/generate_data.py --kind diff_pairs --size 100k`.
# Two sizes so the linear jsono_diff can be read against the superlinear SQL-emulation baseline
# across a 10x data step (the plan's ×4.1-data → ×6.8-time flatten+sort blow-up).
DIFF_PAIRS_SIZES = {
    "10k": 10_000,
    "100k": 100_000,
}
# Average snapshots per transaction; the long tail is in DIFF_PAIRS_SNAPSHOT_SEGMENTS below.
DIFF_PAIRS_ROW_GROUP_SIZE = 32_768

# Schema of the wide-flat dataset (generate_data.py reads these to emit rows, the
# shredded merge scenarios read them to build a shred spec that matches the keys
# exactly). group prefix -> [(key suffix, value kind)], flattened to
# f"{group}_{suffix}". Single source of truth so the shred spec never drifts from
# the generated keys.
WIDE_FLAT_GROUPS = {
    "event": [
        ("ts", "int"),
        ("id", "digits"),
        ("session_id", "digits"),
        ("hit_id", "digits"),
        ("counter_id", "digits"),
        ("secondary_id", "digits"),
        ("visit_id", "digits"),
    ],
    "browser": [
        ("name", "token"),
        ("major_version", "int"),
        ("minor_version", "int"),
        ("engine", "token"),
        ("engine_version", "int"),
        ("language", "token"),
        ("country", "token"),
        ("cookies", "flag"),
    ],
    "screen": [
        ("width", "int"),
        ("height", "int"),
        ("colors", "int"),
        ("format", "token"),
        ("orientation", "token"),
        ("physical_width", "int"),
        ("physical_height", "int"),
    ],
    "window": [
        ("client_width", "int"),
        ("client_height", "int"),
    ],
    "region": [
        ("country", "token"),
        ("country_id", "digits"),
        ("area", "token"),
        ("area_id", "digits"),
        ("city", "token"),
        ("city_id", "digits"),
    ],
    "utm": [
        ("source", "token"),
        ("medium", "token"),
        ("campaign", "token"),
        ("content", "token"),
        ("term", "token"),
    ],
    "offline_call": [
        ("talk_duration", "int"),
        ("hold_duration", "int"),
        ("missed", "flag"),
        ("tag", "sparse_token"),
        ("first_time_caller", "flag"),
        ("url", "url"),
    ],
    "share": [
        ("service", "sparse_token"),
        ("url", "url"),
        ("title", "token"),
    ],
    "openstat": [
        ("ad", "sparse_token"),
        ("campaign", "sparse_token"),
        ("service", "sparse_token"),
        ("source", "sparse_token"),
    ],
    "last": [
        ("traffic_source", "sparse_token"),
        ("search_engine", "sparse_token"),
        ("search_engine_root", "sparse_token"),
        ("adv_engine", "sparse_token"),
        ("social_network", "sparse_token"),
        ("social_network_profile", "sparse_token"),
    ],
    "client": [
        ("id", "digits"),
        ("time_zone", "int"),
        ("user_id_hash", "digits"),
        ("ip", "digits"),
    ],
    "page": [
        ("url", "url"),
        ("referer", "url"),
        ("title", "token"),
        ("charset", "token"),
        ("link", "url"),
        ("download", "flag"),
        ("not_bounce", "flag"),
        ("http_error", "flag"),
    ],
    "device": [
        ("category", "token"),
        ("mobile_phone", "sparse_token"),
        ("mobile_phone_model", "sparse_token"),
        ("os", "token"),
        ("os_root", "token"),
    ],
    "flags": [
        ("javascript_enabled", "flag"),
        ("third_party_cookie_enabled", "flag"),
        ("is_turbo_page", "flag"),
        ("is_turbo_app", "flag"),
        ("has_gclid", "flag"),
        ("iframe", "flag"),
        ("messenger", "sparse_token"),
    ],
    "network": [
        ("type", "token"),
    ],
}

WIDE_FLAT_FIELDS = [
    (f"{group}_{suffix}", kind) for group, suffixes in WIDE_FLAT_GROUPS.items() for suffix, kind in suffixes
] + [(f"custom_param_{i:02d}", "token" if i % 2 == 0 else "sparse_token") for i in range(1, 21)]

# Shred spec covering every always-present scalar top-level key (~107 lanes). int
# values shred as BIGINT, every other kind (digits/token/sparse_token/flag/url) is
# a JSON string -> VARCHAR. event_name is the fixed archetype key; the resulting
# value is a wide shredded group-merge base with many typed lanes.
WIDE_FLAT_SHREDDING_SPEC = {
    "event_name": "VARCHAR",
    **{name: ("BIGINT" if kind == "int" else "VARCHAR") for name, kind in WIDE_FLAT_FIELDS},
}
KEYED_PAIR_WIDE_SHREDDING_SPEC = {
    **WIDE_FLAT_SHREDDING_SPEC,
    "URL": "VARCHAR",
    "primaryID": "VARCHAR",
    "secondaryID": "VARCHAR",
}

# Merge-patch inputs are tagged with how they are built, because the construction
# decides the shredded type the merge sees:
#   {"plain": value}    -> jsono('<json text>'): a PLAIN jsono value (no shreds). This is the
#                          `jsono({'payload': nested_payload})` shape where payload holds a JSONO
#                          value, so it is not auto-shredded — the plain argument fast-paths.
#   {"shredded": value} -> jsono(<struct literal>): jsono({...}) auto-shreds scalar top-level
#                          fields, mirroring jsono({'event_name':.., 'goals':..}).
#   {"auto_shred": value} -> jsono(<struct literal>) over a value with NESTED scalars, which
#                          auto-shred into '$.path' lanes. A nested shred disqualifies the fast
#                          path (it is not a whole top-level lane), so this stays on the reshred
#                          fallback even after H1 — it isolates that separate bottleneck.
WIDE_FLAT_NESTED_PATCH = {"plain": {"payload": {"detail_goal_a": "1042", "detail_goal_b": "5530"}}}
WIDE_FLAT_SMALL_PATCH = {"shredded": {"event_name": "page_view_merged", "goals": "g1,g2,g3"}}
WIDE_FLAT_NESTED_SHRED_PATCH = {"auto_shred": {"payload": {"detail_goal_a": "1042", "detail_goal_b": "5530"}}}

# Shred set for shredded scenarios over the field_sample page_view shape:
# always-present scalar keys, used by projection and filter-pushdown scenarios.
FIELD_SAMPLE_SHREDDING_SPEC = {
    "clientID": "VARCHAR",
    "event_name": "VARCHAR",
    "url": "VARCHAR",
    "UTMSource": "VARCHAR",
}

# Filterless multi-extract projection paths: two covered by
# FIELD_SAMPLE_SHREDDING_SPEC, three residual-only, so the shredded variant
# exercises both the struct_extract rewrite and the per-path residual reads
# that O-1/O-2 want to fuse.
FIELD_SAMPLE_PROJECT_PATHS = [
    "$.clientID",
    "$.event_name",
    "$.browser",
    "$.regionCity",
    "$.operatingSystem",
]

# Filtered multi-predicate read: three residual-only paths (none in
# FIELD_SAMPLE_SHREDDING_SPEC), each an IN over its common values so most rows
# pass several predicates and the matcher pays all three residual locates. The
# matcher fuses them into one __jsono_internal_match over a single residual read
# (O-2); without the fuse the shredded variant pays three residual reads + three
# manifest checks per row.
FIELD_SAMPLE_FILTER_PREDICATES = [
    {"path": "$.browser", "values": ["chrome", "alt_browser", "safari_mobile"]},
    {"path": "$.operatingSystem", "values": ["windows10", "ios18", "android"]},
    {"path": "$.regionCity", "values": ["Moscow", "Saint Petersburg", "Kiev"]},
]

PLUCK_CORE_SPEC = {
    "event_name": "VARCHAR",
    "event_id": "VARCHAR",
    "event_ts": "BIGINT",
    "user_id": "VARCHAR",
    "session_id": "VARCHAR",
    "utm_source": "VARCHAR",
    "utm_medium": "VARCHAR",
    "page_url": "VARCHAR",
    "device_type": "VARCHAR",
    "geo_country": "VARCHAR",
}

TRANSFORM_CORE_SPEC = {
    "event_name": "VARCHAR",
    "event_id": "VARCHAR",
    "event_ts": "BIGINT",
    "user_id": "VARCHAR",
    "session_id": "VARCHAR",
    "utm_source": {"type": "VARCHAR", "path": "$.utm.utm_source"},
    "utm_medium": {"type": "VARCHAR", "path": "$.utm.utm_medium"},
    "page_url": {"type": "VARCHAR", "path": "$.page.page_url"},
    "device_type": {"type": "VARCHAR", "path": "$.device.device_type"},
    "geo_country": {"type": "VARCHAR", "path": "$.geo.geo_country"},
}

TRANSFORM_ECOMMERCE_JOIN_SPEC = {
    "event_name": "VARCHAR",
    "product_categories": {
        "type": "VARCHAR",
        "path": "$.ec_items[*].item_name",
        "join_separator": "\n",
    },
}

# Scalar top-level paths present in every json_nested archetype. Both
# jsono_transform and core json_transform accept the same VARCHAR/BIGINT schema,
# so outputs are field-by-field comparable.
EXTRACT_CORE_SPEC = {
    "event_name": "VARCHAR",
    "event_ts": "BIGINT",
    "user_id": "VARCHAR",
    "session_id": "VARCHAR",
    "page_url": "VARCHAR",
    "device_type": "VARCHAR",
}

FIELD_SAMPLE_STRUCT_CORE_SPEC = {
    "title": "VARCHAR",
    "url": "VARCHAR",
    "referer": "VARCHAR",
    "event_name": "VARCHAR",
    "clientID": "VARCHAR",
    "UTMCampaign": "VARCHAR",
    "UTMContent": "VARCHAR",
    "UTMMedium": "VARCHAR",
    "UTMSource": "VARCHAR",
    "UTMTerm": "VARCHAR",
    "browser": "VARCHAR",
    "browserMajorVersion": "VARCHAR",
    "browserCountry": "VARCHAR",
    "browserEngine": "VARCHAR",
    "browserLanguage": "VARCHAR",
    "deviceCategory": "VARCHAR",
    "mobilePhone": "VARCHAR",
    "mobilePhoneModel": "VARCHAR",
    "operatingSystem": "VARCHAR",
    "operatingSystemRoot": "VARCHAR",
    "regionCountry": "VARCHAR",
    "regionArea": "VARCHAR",
    "regionCity": "VARCHAR",
    "screenWidth": "VARCHAR",
    "screenHeight": "VARCHAR",
    "windowClientWidth": "VARCHAR",
    "windowClientHeight": "VARCHAR",
    "product_all_category": "VARCHAR",
}

# Row-group pruning scenario. Shred the monotonic `event_ts` leaf as a typed
# BIGINT sibling and cluster the rows by it on write, so each Parquet row group
# holds a disjoint event_ts range and a selective band filter skips the others.
# The native control column carries the same value as a plain BIGINT, so the
# shred-leaf filter and the native-column filter prune identically — the metric
# (OPERATOR_ROW_GROUPS_SCANNED / OPERATOR_TOTAL_ROW_GROUPS_TO_SCAN, surfaced by
# --row-groups) shows scanned < total under the filter, scanned == total without.
# The band targets ~1 of 9 row groups for the seed-42 events_100k event_ts range
# (≈1.716e9..1.7186e9); it is a property of that generated data, not a runtime
# computation, so regenerating events with a different seed may shift it.
PRUNE_FILTER_COPY_PATH = RESULTS_DIR / "synthetic_prune.parquet"
PRUNE_FILTER_SHRED_LEAF = "$.event_ts"
PRUNE_FILTER_SHRED_TYPE = "BIGINT"
PRUNE_FILTER_NATIVE_COLUMN = "event_ts_native"
PRUNE_FILTER_BAND = (1_717_000_000, 1_717_120_000)
PRUNE_FILTER_ROW_GROUP_SIZE = 12_000

# Type-reconciliation boundary scenarios (plan 033): a facade UNION ALL over a shredded and a
# plain branch, and a union_by_name multi-file read over files with diverging shred sets where
# the measured lane is present in both. Both time the same lane extract; the win comes from the
# set-op pushdown / recovered-statistics totality fold, not from a different operation contract.
SETOP_EXTRACT_SHREDDING_SPEC = {"event_name": "VARCHAR"}
MULTIFILE_EXTRACT_SHREDDING_A = {"event_name": "VARCHAR", "$.page.page_url": "VARCHAR"}
MULTIFILE_EXTRACT_SHREDDING_B = {
    "event_name": "VARCHAR",
    "$.geo.geo_country": "VARCHAR",
}
MULTIFILE_EXTRACT_COPY_PATH_A = RESULTS_DIR / "synthetic_multifile_a.parquet"
MULTIFILE_EXTRACT_COPY_PATH_B = RESULTS_DIR / "synthetic_multifile_b.parquet"

FIELD_SAMPLE_STRUCT_PRODUCTS_SPEC = {
    "event_name": "VARCHAR",
    "url": "VARCHAR",
    "clientID": "VARCHAR",
    "product_all_category": "VARCHAR",
    "products": [
        {
            "name": "VARCHAR",
            "price": "DOUBLE",
            "discount": "DOUBLE",
            "brand": "VARCHAR",
            "category": "VARCHAR",
            "variant": "VARCHAR",
        }
    ],
}


from scenarios.field_sample import FIELD_SAMPLE_SCENARIOS
from scenarios.synthetic import SCENARIOS

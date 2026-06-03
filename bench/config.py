from pathlib import Path

BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
PROFILES_DIR = RESULTS_DIR / "profiles"
PROJECT_ROOT = BENCH_DIR.parent

JSONO_EXTENSION_PATH = (
    PROJECT_ROOT
    / "build"
    / "release"
    / "extension"
    / "jsono"
    / "jsono.duckdb_extension"
)
FIELD_SAMPLE_DATA_DIR = DATA_DIR / "field_sample"
FIELD_SAMPLE_EVENTS_NESTED_PATH = FIELD_SAMPLE_DATA_DIR / "events_nested.parquet"
FIELD_SAMPLE_JSONO_EVENTS_PATH = (
    DATA_DIR / "field_sample" / "events_nested_jsono.parquet"
)
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
RETAIL_SAMPLE_ECOMMERCE_PATH = (
    DATA_DIR / "field_sample" / "retail_sample_ecommerce.parquet"
)
RETAIL_SAMPLE_ECOMMERCE_ROW_COUNT = 65_987
RETAIL_SAMPLE_ECOMMERCE_PARSE_COPY_PATH = (
    RESULTS_DIR / "retail_sample_ecommerce_parse_copy.parquet"
)
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

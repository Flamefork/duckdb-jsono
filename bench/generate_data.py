#!/usr/bin/env python3
"""Generate realistic marketing-telemetry JSON events for benchmarks.

v2 — event archetypes for schema-stable data.

Real marketing data is **schema-clustered**: each `event_name` produces a
consistent key set across millions of rows (a `purchase` always has
ec_revenue, a `page_view` never does). The previous generator rolled
optional groups independently per row, producing ~39k distinct schemas in
100k rows — wildly unrealistic and breaking any schema-aware analysis.

v2 picks one of ~14 fixed archetypes per row; required groups are always
present, optional fields vary only at the leaf level. Target: 10-30
distinct schemas across the whole dataset, ~50-80 keys per row.
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timezone

import duckdb

from config import (
    DATA_DIR,
    KEYED_PAIR_PAGE_GROUPS,
    KEYED_PAIR_ROW_GROUP_SIZE,
    KEYED_PAIR_SIZES,
    KEYED_PAIR_USER_GROUPS,
    NUMBERS_SIZES,
    SIZES,
    URL_DEFAULT_SIZES,
    URL_SIZES,
    WIDE_FLAT_FIELDS,
    WIDE_FLAT_GROUPS,
    WIDE_FLAT_SIZES,
)

# ────────────────────────────────────────────────────────────────────────
# Value pools
# ────────────────────────────────────────────────────────────────────────

UTM_SOURCES = [
    "google",
    "facebook",
    "instagram",
    "local_search",
    "vk",
    "tiktok",
    "email",
    "direct",
    "bing",
    "twitter",
    "youtube",
    "linkedin",
]
UTM_MEDIUMS = [
    "cpc",
    "organic",
    "social",
    "email",
    "referral",
    "display",
    "video",
    "affiliate",
    "push",
]
UTM_CAMPAIGNS = [
    f"{theme}_{year}_{i:02d}"
    for theme in [
        "spring_sale",
        "summer_promo",
        "black_friday",
        "newyear",
        "back_to_school",
        "valentines",
        "brand_awareness",
        "retargeting",
    ]
    for year in ["2023", "2024", "2025"]
    for i in range(1, 6)
]
UTM_CONTENTS = [
    "banner_top",
    "banner_side",
    "card_a",
    "card_b",
    "video_pre",
    "email_cta",
    "popup",
    "footer_link",
    "menu_promo",
]
UTM_TERMS = [
    "buy_shoes",
    "best_laptop",
    "cheap_flights",
    "ai_tools",
    "online_course",
    "summer_dress",
]

DOMAINS = [
    "example.com",
    "shop.example.com",
    "blog.example.com",
    "promo.partner.com",
    "app.startup.io",
]
PAGE_PATHS = [
    "/",
    "/products",
    "/products/shoes",
    "/products/laptops",
    "/cart",
    "/checkout",
    "/blog/seo-tips",
    "/about",
    "/contact",
    "/account",
    "/account/orders",
    "/search?q=running",
]
PAGE_TITLES = [
    "Home – Example Store",
    "All Products",
    "Running Shoes – Best Sellers",
    "Laptops Under $1000",
    "Your Cart",
    "Secure Checkout",
    "SEO Tips for 2025",
    "About Us",
    "Contact Support",
    "My Account",
    "Order History",
]
REFERRERS = [
    "https://www.google.com/search?q=shoes",
    "https://www.google.com/",
    "https://search.example/",
    "https://m.facebook.com/",
    "https://t.me/somechannel",
    "https://news.ycombinator.com/",
    "https://twitter.com/",
]

DEVICE_TYPES = ["mobile", "desktop", "tablet"]
BROWSERS = ["Chrome", "Safari", "Firefox", "Edge", "AltBrowser", "Opera"]
OSES = ["iOS", "Android", "Windows", "macOS", "Linux"]
SCREENS = [
    "390x844",
    "414x896",
    "375x667",
    "1920x1080",
    "2560x1440",
    "1440x900",
    "1366x768",
    "768x1024",
]
LANGUAGES = ["en-US", "ru-RU", "es-ES", "de-DE", "fr-FR", "pt-BR", "ja-JP", "zh-CN"]
UA_TEMPLATES = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/121.0",
]

COUNTRIES = ["US", "RU", "DE", "FR", "BR", "JP", "GB", "CA", "AU", "IN", "ES", "IT"]
REGIONS = {
    "US": ["California", "New York", "Texas", "Florida"],
    "RU": ["Moscow", "St. Petersburg", "Sverdlovsk", "Tatarstan"],
    "DE": ["Berlin", "Bavaria", "Hesse"],
    "FR": ["Île-de-France", "Provence"],
    "BR": ["São Paulo", "Rio de Janeiro"],
    "JP": ["Tokyo", "Osaka"],
    "GB": ["England", "Scotland"],
    "CA": ["Ontario", "Quebec"],
}
CITIES = {
    "California": ["San Francisco", "Los Angeles", "San Jose"],
    "New York": ["New York City", "Buffalo"],
    "Moscow": ["Moscow"],
    "Berlin": ["Berlin"],
    "Tokyo": ["Tokyo"],
}

CURRENCIES = ["USD", "EUR", "RUB", "JPY", "BRL", "GBP"]

ITEM_NAMES = [
    "Running Shoes",
    "Wireless Headphones",
    "Laptop Stand",
    "Coffee Mug",
    "Notebook",
    "USB Cable",
    "T-Shirt",
    "Jeans",
    "Sunglasses",
    "Backpack",
]

LOGIN_METHODS = ["email", "google_oauth", "apple_id", "facebook", "phone_otp"]
SIGNUP_METHODS = ["email", "google_oauth", "apple_id", "phone_otp"]
SHARE_CHANNELS = ["facebook", "twitter", "telegram", "whatsapp", "copy_link", "email"]
SEARCH_QUERIES = [
    "running shoes",
    "laptop",
    "summer dress",
    "wireless headphones",
    "coffee mug",
    "back to school",
    "cheap flights",
]


# ────────────────────────────────────────────────────────────────────────
# Field-group generators — each emits a dict of keys for one group.
# ────────────────────────────────────────────────────────────────────────


def make_page_url(rng: random.Random, path: str) -> str:
    domain = rng.choice(DOMAINS)
    qs = ""
    if rng.random() < 0.40:
        params = []
        if rng.random() < 0.7:
            params.append(f"utm_source={rng.choice(UTM_SOURCES)}")
        if rng.random() < 0.5:
            params.append(f"utm_campaign={rng.choice(UTM_CAMPAIGNS)}")
        if rng.random() < 0.3:
            params.append(f"ref={rng.randint(100, 9999)}")
        if params:
            qs = "?" + "&".join(params)
    return f"https://{domain}{path}{qs}"


def grp_core(rng: random.Random, seed: int, row_idx: int) -> dict:
    return {
        "event_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{seed}:event:{row_idx}")),
        "event_ts": 1_716_000_000 + rng.randint(0, 2_592_000),
        "user_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{seed}:user:{row_idx // 5}")),
        "session_id": str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{seed}:session:{row_idx // 5}")
        ),
    }


def grp_utm(rng: random.Random) -> dict:
    return {
        "utm_source": rng.choice(UTM_SOURCES),
        "utm_medium": rng.choice(UTM_MEDIUMS),
        "utm_campaign": rng.choice(UTM_CAMPAIGNS),
        "utm_content": rng.choice(UTM_CONTENTS),
        "utm_term": rng.choice(UTM_TERMS),
    }


def grp_page(rng: random.Random) -> dict:
    path = rng.choice(PAGE_PATHS)
    return {
        "page_url": make_page_url(rng, path),
        "page_path": path,
        "page_title": rng.choice(PAGE_TITLES),
        "page_referrer": rng.choice(REFERRERS),
    }


def grp_device(rng: random.Random) -> dict:
    return {
        "device_type": rng.choice(DEVICE_TYPES),
        "device_browser": rng.choice(BROWSERS),
        "device_browser_version": f"{rng.randint(80, 125)}.0",
        "device_os": rng.choice(OSES),
        "device_os_version": f"{rng.randint(10, 17)}.{rng.randint(0, 5)}",
        "device_user_agent": rng.choice(UA_TEMPLATES),
        "device_screen": rng.choice(SCREENS),
        "device_language": rng.choice(LANGUAGES),
    }


def grp_geo(rng: random.Random) -> dict:
    country = rng.choice(COUNTRIES)
    region = rng.choice(REGIONS.get(country, ["Unknown"]))
    city = rng.choice(CITIES.get(region, ["Unknown"]))
    return {
        "geo_country": country,
        "geo_region": region,
        "geo_city": city,
    }


def grp_ga(rng: random.Random) -> dict:
    return {
        "ga_client_id": f"{rng.randint(1_000_000_000, 9_999_999_999)}.{rng.randint(1_000_000_000, 9_999_999_999)}",
        "ga_session_id": str(rng.randint(1_700_000_000, 1_750_000_000)),
        "ga_user_pseudo_id": str(rng.randint(1_000_000_000, 9_999_999_999)),
    }


def grp_pixels(rng: random.Random) -> dict:
    return {
        "fbp": f"fb.1.{rng.randint(1_000_000_000, 9_999_999_999)}.{rng.randint(100, 999)}",
        "external_uid": str(rng.randint(1_000_000_000, 9_999_999_999)),
        "ym_uid": str(rng.randint(1_000_000_000, 9_999_999_999)),
        "_ga": f"GA1.2.{rng.randint(100_000_000, 999_999_999)}.{rng.randint(1_000_000_000, 9_999_999_999)}",
    }


def grp_ec(rng: random.Random) -> dict:
    return {
        "ec_currency": rng.choice(CURRENCIES),
        "ec_revenue": round(rng.uniform(5, 500), 2),
        "ec_transaction_id": f"txn_{rng.randint(100000, 999999)}",
        "ec_tax": round(rng.uniform(0, 50), 2),
        "ec_shipping": round(rng.uniform(0, 30), 2),
    }


def grp_ec_items(rng: random.Random, n_items: int) -> dict:
    return {
        "ec_items": [
            {
                "item_id": f"sku_{rng.randint(1000, 9999)}",
                "item_name": rng.choice(ITEM_NAMES),
                "price": round(rng.uniform(5, 200), 2),
                "quantity": rng.randint(1, 3),
            }
            for _ in range(n_items)
        ]
    }


# ────────────────────────────────────────────────────────────────────────
# Event archetypes — each maps event_name → (weight, [field-groups], [extras])
# Within an archetype the present key SET is fixed; only leaf values vary.
# ────────────────────────────────────────────────────────────────────────


def custom_extras(rng: random.Random, keys: list[str]) -> dict:
    """Emit a fixed set of custom_event_* keys for an archetype."""
    out = {}
    for k in keys:
        if k in ("video_position", "page_depth", "scroll_pct"):
            out[f"custom_event_{k}"] = rng.randint(0, 100)
        else:
            out[f"custom_event_{k}"] = f"val_{rng.randint(0, 100)}"
    return out


# (weight, base_groups, extras_fn or None)
ARCHETYPES = {
    "page_view": (
        45,
        ["core", "utm", "page", "device", "geo", "ga", "pixels"],
        None,
    ),
    "click": (
        15,
        ["core", "utm", "page", "device", "geo", "ga"],
        lambda rng: custom_extras(rng, ["button_text", "section_id", "element_id"]),
    ),
    "scroll": (
        8,
        ["core", "page", "device", "geo"],
        lambda rng: custom_extras(rng, ["scroll_pct", "page_depth"]),
    ),
    "form_submit": (
        4,
        ["core", "utm", "page", "device", "geo", "ga"],
        lambda rng: custom_extras(rng, ["form_id", "form_step", "form_type"]),
    ),
    "purchase": (
        4,
        ["core", "utm", "page", "device", "geo", "ga", "ec", "ec_items", "pixels"],
        None,
    ),
    "add_to_cart": (
        6,
        ["core", "utm", "page", "device", "geo", "ga"],
        lambda rng: {
            "ec_item_id": f"sku_{rng.randint(1000, 9999)}",
            "ec_item_name": rng.choice(ITEM_NAMES),
            "ec_item_price": round(rng.uniform(5, 200), 2),
            "ec_currency": rng.choice(CURRENCIES),
        },
    ),
    "view_item": (
        5,
        ["core", "utm", "page", "device", "geo", "ga"],
        lambda rng: {
            "ec_item_id": f"sku_{rng.randint(1000, 9999)}",
            "ec_item_name": rng.choice(ITEM_NAMES),
            "ec_item_price": round(rng.uniform(5, 200), 2),
        },
    ),
    "sign_up": (
        2,
        ["core", "utm", "page", "device", "geo"],
        lambda rng: custom_extras(rng, ["signup_method"])
        | {
            "custom_event_signup_method": rng.choice(SIGNUP_METHODS),
        },
    ),
    "login": (
        3,
        ["core", "page", "device", "geo"],
        lambda rng: {
            "custom_event_login_method": rng.choice(LOGIN_METHODS),
        },
    ),
    "search": (
        4,
        ["core", "page", "device", "geo"],
        lambda rng: {
            "custom_event_search_query": rng.choice(SEARCH_QUERIES),
            "custom_event_filter_applied": f"val_{rng.randint(0, 20)}",
            "custom_event_sort_order": rng.choice(
                ["relevance", "price_asc", "price_desc", "newest"]
            ),
        },
    ),
    "share": (
        2,
        ["core", "page", "device"],
        lambda rng: {
            "custom_event_share_channel": rng.choice(SHARE_CHANNELS),
            "custom_event_content_id": f"content_{rng.randint(100, 9999)}",
        },
    ),
    "video_play": (
        1,
        ["core", "page", "device", "geo"],
        lambda rng: {
            "custom_event_video_id": f"video_{rng.randint(100, 999)}",
            "custom_event_video_position": rng.randint(0, 600),
            "custom_event_video_duration": rng.randint(60, 1800),
        },
    ),
    "video_complete": (
        1,
        ["core", "page", "device", "geo"],
        lambda rng: {
            "custom_event_video_id": f"video_{rng.randint(100, 999)}",
            "custom_event_video_duration": rng.randint(60, 1800),
        },
    ),
}


GROUP_GENERATORS = {
    "core": grp_core,
    "utm": lambda rng, *_: grp_utm(rng),
    "page": lambda rng, *_: grp_page(rng),
    "device": lambda rng, *_: grp_device(rng),
    "geo": lambda rng, *_: grp_geo(rng),
    "ga": lambda rng, *_: grp_ga(rng),
    "pixels": lambda rng, *_: grp_pixels(rng),
    "ec": lambda rng, *_: grp_ec(rng),
    "ec_items": lambda rng, *_: grp_ec_items(rng, n_items=2),
}


# Pre-build cumulative weights for archetype selection
_ARCHETYPE_NAMES = list(ARCHETYPES.keys())
_ARCHETYPE_WEIGHTS = [ARCHETYPES[n][0] for n in _ARCHETYPE_NAMES]


def make_rng(seed: int, row_idx: int) -> random.Random:
    return random.Random(f"{seed}:{row_idx}")


def generate_event(seed: int, row_idx: int) -> dict:
    rng = make_rng(seed, row_idx)
    archetype = rng.choices(_ARCHETYPE_NAMES, weights=_ARCHETYPE_WEIGHTS, k=1)[0]
    _, base_groups, extras_fn = ARCHETYPES[archetype]

    e: dict = {"event_name": archetype}
    for grp in base_groups:
        gen = GROUP_GENERATORS[grp]
        if grp == "core":
            e.update(gen(rng, seed, row_idx))
        else:
            e.update(gen(rng))

    if extras_fn is not None:
        e.update(extras_fn(rng))

    return e


def flatten_event(event: dict) -> dict:
    out: dict = {}
    for k, v in event.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                out[f"{k}.{sub_k}"] = sub_v
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    for sub_k, sub_v in item.items():
                        out[f"{k}[{i}].{sub_k}"] = sub_v
                else:
                    out[f"{k}[{i}]"] = item
        else:
            out[k] = v
    return out


def generate_row(row_idx: int, seed: int) -> dict:
    event = generate_event(seed, row_idx)
    flat = flatten_event(event)
    return {
        "json_nested": json.dumps(event, separators=(",", ":")),
        "json_flat": json.dumps(flat, separators=(",", ":")),
        "g1e1": f"g_{row_idx % 10}",
        "g1e3": f"g_{row_idx % 1000}",
        "g1e4": f"g_{row_idx % 10000}",
    }


def generate_data(size_name: str, num_rows: int, seed: int) -> None:
    print(f"Generating {size_name} ({num_rows:,} rows) with seed={seed}...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"events_{size_name}.parquet"

    conn = duckdb.connect()

    batch_size = 50_000
    rows_generated = 0
    temp_table_created = False

    while rows_generated < num_rows:
        batch_rows = min(batch_size, num_rows - rows_generated)
        batch = [generate_row(rows_generated + i, seed) for i in range(batch_rows)]

        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE batch (
                json_nested JSON,
                json_flat JSON,
                g1e1 VARCHAR,
                g1e3 VARCHAR,
                g1e4 VARCHAR
            )
            """
        )
        conn.executemany(
            "INSERT INTO batch VALUES (?, ?, ?, ?, ?)",
            [
                (r["json_nested"], r["json_flat"], r["g1e1"], r["g1e3"], r["g1e4"])
                for r in batch
            ],
        )

        if not temp_table_created:
            conn.execute("CREATE OR REPLACE TABLE data AS SELECT * FROM batch")
            temp_table_created = True
        else:
            conn.execute("INSERT INTO data SELECT * FROM batch")

        rows_generated += batch_rows
        print(f"  {rows_generated:,} / {num_rows:,}")

    generated_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        f"""
        COPY data TO '{output_path}' (
            FORMAT PARQUET,
            KV_METADATA {{
                seed: '{seed}',
                size: '{size_name}',
                rows: '{num_rows}',
                generated_at: '{generated_at}',
                schema_version: 'marketing_telemetry_v2_archetypes'
            }}
        )
        """
    )
    conn.close()

    print(f"  Saved to {output_path}")


# ────────────────────────────────────────────────────────────────────────
# Number-heavy dataset — covers the JSONO numeric ladder that field_sample does
# not. Raw JSON text is assembled by hand (not json.dumps) so byte-exact tokens
# survive: trailing zeros and high-precision decimals that Python would
# otherwise normalize.
#
# Scope is deliberately limited to values the *current* (simdjson) parser
# accepts, so the same scenario yields a comparable before/after number across
# every phase: int60, signed int64 (|v|<2^63), dec60-fit fractionals, and
# high-precision fractionals (parsed lossy-as-double today, byte-exact NUMBER
# after the rework). uint64>2^63 and bignum>2^64 are NOT included: the baseline
# rejects them outright, so they cannot produce a same-session baseline. Their
# correctness is covered by SQLLogic tests in later phases, not by this bench.
# ────────────────────────────────────────────────────────────────────────


def generate_numbers_row(row_idx: int, seed: int) -> dict:
    rng = make_rng(seed, row_idx)
    int60 = rng.randint(-(2**58), 2**58)
    int64 = rng.randint(2**59, 2**63 - 1) * rng.choice([-1, 1])
    # dec60-fit: small mantissa, modest scale; keep trailing zero to test byte-exact
    dec60 = f"{rng.randint(0, 999999)}.{rng.randint(10, 99)}0"
    # high-precision: too many significant digits for DEC60 (mantissa >= 2^53)
    highprec = f"{rng.randint(10**16, 10**17)}.{rng.randint(10**16, 10**17)}"
    raw = (
        "{"
        f'"i60":{int60},'
        f'"i64":{int64},'
        f'"dec":{dec60},'
        f'"hp":{highprec},'
        f'"arr":[{int60},{dec60},{int64}]'
        "}"
    )
    return {
        "json_numbers": raw,
        "g1e1": f"g_{row_idx % 10}",
    }


def generate_numbers_data(size_name: str, num_rows: int, seed: int) -> None:
    print(f"Generating numbers {size_name} ({num_rows:,} rows) with seed={seed}...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"numbers_{size_name}.parquet"

    conn = duckdb.connect()

    batch_size = 50_000
    rows_generated = 0
    temp_table_created = False

    while rows_generated < num_rows:
        batch_rows = min(batch_size, num_rows - rows_generated)
        batch = [
            generate_numbers_row(rows_generated + i, seed) for i in range(batch_rows)
        ]

        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE batch (
                json_numbers VARCHAR,
                g1e1 VARCHAR
            )
            """
        )
        conn.executemany(
            "INSERT INTO batch VALUES (?, ?)",
            [(r["json_numbers"], r["g1e1"]) for r in batch],
        )

        if not temp_table_created:
            conn.execute("CREATE OR REPLACE TABLE data AS SELECT * FROM batch")
            temp_table_created = True
        else:
            conn.execute("INSERT INTO data SELECT * FROM batch")

        rows_generated += batch_rows
        print(f"  {rows_generated:,} / {num_rows:,}")

    generated_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        f"""
        COPY data TO '{output_path}' (
            FORMAT PARQUET,
            KV_METADATA {{
                seed: '{seed}',
                size: '{size_name}',
                rows: '{num_rows}',
                generated_at: '{generated_at}',
                schema_version: 'numbers_ladder_v1'
            }}
        )
        """
    )
    conn.close()

    print(f"  Saved to {output_path}")


# ────────────────────────────────────────────────────────────────────────
# Wide-flat dataset — page_view-class telemetry hit: one schema-stable object
# of ~100 scalar top-level keys, no nesting. Key names share group prefixes
# (browser_*, screen_*, ...) like real hit payloads, so sorted-key comparison
# cost is realistic. Exists because merge-family performance is shape-dependent:
# wide-flat objects stress the sorted-key merge in a way the deep-nested retail
# sample does not.
# ────────────────────────────────────────────────────────────────────────

WIDE_FLAT_TOKENS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "direct",
    "organic",
    "internal",
    "none",
    "unknown",
    "mobile",
    "desktop",
    "tablet",
]


def wide_flat_value(rng: random.Random, kind: str) -> object:
    match kind:
        case "int":
            return rng.randint(0, 4096)
        case "digits":
            return str(rng.randint(1_000_000_000, 9_999_999_999))
        case "token":
            return rng.choice(WIDE_FLAT_TOKENS)
        case "sparse_token":
            return "" if rng.random() < 0.75 else f"tok_{rng.randint(0, 99)}"
        case "flag":
            return rng.choice(["0", "1"])
        case "url":
            return make_page_url(rng, rng.choice(PAGE_PATHS))
        case _:
            raise ValueError(f"unknown wide_flat value kind: {kind}")


def generate_wide_flat_row(row_idx: int, seed: int) -> dict:
    rng = make_rng(seed, row_idx)
    event: dict = {"event_name": "page_view"}
    for key, kind in WIDE_FLAT_FIELDS:
        event[key] = wide_flat_value(rng, kind)
    return {
        "json_wide_flat": json.dumps(event, separators=(",", ":")),
        "g1e1": f"g_{row_idx % 10}",
        "g1e4": f"g_{row_idx % 10000}",
    }


def generate_wide_flat_data(size_name: str, num_rows: int, seed: int) -> None:
    print(f"Generating wide_flat {size_name} ({num_rows:,} rows) with seed={seed}...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"wide_flat_{size_name}.parquet"

    conn = duckdb.connect()

    batch_size = 50_000
    rows_generated = 0
    temp_table_created = False

    while rows_generated < num_rows:
        batch_rows = min(batch_size, num_rows - rows_generated)
        batch = [
            generate_wide_flat_row(rows_generated + i, seed) for i in range(batch_rows)
        ]

        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE batch (
                json_wide_flat JSON,
                g1e1 VARCHAR,
                g1e4 VARCHAR
            )
            """
        )
        conn.executemany(
            "INSERT INTO batch VALUES (?, ?, ?)",
            [(r["json_wide_flat"], r["g1e1"], r["g1e4"]) for r in batch],
        )

        if not temp_table_created:
            conn.execute("CREATE OR REPLACE TABLE data AS SELECT * FROM batch")
            temp_table_created = True
        else:
            conn.execute("INSERT INTO data SELECT * FROM batch")

        rows_generated += batch_rows
        print(f"  {rows_generated:,} / {num_rows:,}")

    generated_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        f"""
        COPY data TO '{output_path}' (
            FORMAT PARQUET,
            KV_METADATA {{
                seed: '{seed}',
                size: '{size_name}',
                rows: '{num_rows}',
                generated_at: '{generated_at}',
                schema_version: 'wide_flat_pageview_v1'
            }}
        )
        """
    )
    conn.close()

    print(f"  Saved to {output_path}")


def sql_string(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def keyed_pair_value_sql(kind: str) -> str:
    match kind:
        case "int":
            return "(i % 4097)::BIGINT"
        case "digits":
            return "(1000000000 + ((i * 17) % 9000000000))::VARCHAR"
        case "token":
            return "'tok_' || (i % 97)::VARCHAR"
        case "sparse_token":
            return "CASE WHEN i % 4 = 0 THEN 'tok_' || (i % 97)::VARCHAR ELSE '' END"
        case "flag":
            return "CASE WHEN i % 2 = 0 THEN '1' ELSE '0' END"
        case "url":
            return "'https://example.com/page/' || (i % 1000)::VARCHAR"
        case _:
            raise ValueError(f"unknown keyed_pair value kind: {kind}")


def json_object_sql(fields: list[tuple[str, str]]) -> str:
    arguments = []
    for key, expression in fields:
        arguments.append(sql_string(key))
        arguments.append(expression)
    return "json_object(" + ", ".join(arguments) + ")::JSON"


def group_rows_sql(group_column: str, segments: list[tuple[int, int]]) -> str:
    start_group = 0
    start_row = 0
    selects = []
    for group_count, group_size in segments:
        selects.append(
            f"""
            SELECT
                row_num: {start_row}::UBIGINT + group_offset * {group_size} + row_offset,
                {group_column}: {start_group}::UBIGINT + group_offset
            FROM range({group_count}) group_offsets(group_offset)
            CROSS JOIN range({group_size}) row_offsets(row_offset)
            """
        )
        start_group += group_count
        start_row += group_count * group_size
    return "\nUNION ALL\n".join(selects)


def generate_keyed_pair_data(size_name: str, num_rows: int, seed: int) -> None:
    print(f"Generating keyed_pair {size_name} ({num_rows:,} rows) with seed={seed}...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"keyed_pair_{size_name}.parquet"

    conn = duckdb.connect()

    generated_at = datetime.now(timezone.utc).isoformat()
    root_fields = [
        ("event_name", "'page_view'"),
        ("URL", "'https://example.com/page/' || (i % 1000)::VARCHAR"),
        (
            "primaryID",
            "(row_num % {groups} + 1)::VARCHAR".format(groups=KEYED_PAIR_PAGE_GROUPS),
        ),
        ("secondaryID", "(row_num % 1000000 + 1)::VARCHAR"),
        *[(key, keyed_pair_value_sql(kind)) for key, kind in WIDE_FLAT_FIELDS],
    ]
    detail_fields = [
        ("browser", "'browser_' || (i % 7)::VARCHAR"),
        ("device", "'device_' || (i % 3)::VARCHAR"),
        ("screen", "'screen_' || (i % 8)::VARCHAR"),
        ("language", "'lang_' || (i % 6)::VARCHAR"),
        ("referer", "'https://ref.example/' || (i % 100)::VARCHAR"),
        ("scroll_depth", "(i % 101)::BIGINT"),
        ("duration", "(1 + (i % 600))::BIGINT"),
        ("is_bounce", "i % 2 = 0"),
    ]
    user_fields = [
        ("client_type", "'type_' || (i % 3)::VARCHAR"),
        ("segment", "'segment_' || (i % 4)::VARCHAR"),
        ("country", "'country_' || (i % 12)::VARCHAR"),
        ("visits", "(1 + (i % 25))::BIGINT"),
        ("lead_score", "(i % 1001)::BIGINT"),
    ]
    page_segments = [
        (44_000, 9),
        (33_000, 12),
        (9_000, 22),
        (206, 48),
        (1, 112),
    ]
    user_segments = [
        (100_000, 3),
        (130_000, 4),
        (30_000, 5),
        (2_900, 9),
        (100, 39),
    ]
    conn.execute(
        f"""
        COPY (
            WITH
            page_rows AS (
                {group_rows_sql("page_group", page_segments)}
            ),
            user_rows AS (
                {group_rows_sql("user_group", user_segments)}
            )
            SELECT
                json_wide_payload: {json_object_sql(root_fields)},
                json_detail_payload: {json_object_sql(detail_fields)},
                json_small_payload: {json_object_sql(user_fields)},
                g_medium_group_rows: 'pv_' || page_group::VARCHAR,
                g_few_group_rows: 'up_' || user_group::VARCHAR,
            FROM page_rows
            JOIN user_rows USING (row_num)
            CROSS JOIN LATERAL (
                SELECT (row_num + {seed})::UBIGINT AS i
            )
        )
        TO '{output_path}' (
            FORMAT PARQUET,
            ROW_GROUP_SIZE {KEYED_PAIR_ROW_GROUP_SIZE},
            KV_METADATA {{
                seed: '{seed}',
                size: '{size_name}',
                rows: '{num_rows}',
                generated_at: '{generated_at}',
                row_group_size: '{KEYED_PAIR_ROW_GROUP_SIZE}',
                schema_version: 'keyed_pair_v1'
            }}
        )
        """
    )
    conn.close()

    print(f"  Saved to {output_path}")


def clean_path(path: str) -> str:
    return path.split("?", 1)[0]


def make_short_url(rng: random.Random) -> str:
    return f"https://{rng.choice(DOMAINS)}{clean_path(rng.choice(PAGE_PATHS))}"


def make_marketing_url(rng: random.Random, row_idx: int) -> str:
    params = [
        ("utm_source", rng.choice(UTM_SOURCES)),
        ("utm_medium", rng.choice(UTM_MEDIUMS)),
        ("utm_campaign", rng.choice(UTM_CAMPAIGNS)),
        ("utm_content", rng.choice(UTM_CONTENTS)),
        ("gclid", f"g_{row_idx}_{rng.randint(100000, 999999)}"),
        ("fbclid", f"fb_{rng.randint(100000000, 999999999)}"),
        (
            "term",
            rng.choice(
                [
                    "running+shoes",
                    "summer%20dress",
                    "coffee%20mug",
                    "wireless+headphones",
                ]
            ),
        ),
        ("flag", ""),
    ]
    query = "&".join(
        key if key == "flag" else f"{key}={value}" for key, value in params
    )
    return f"https://{rng.choice(DOMAINS)}{clean_path(rng.choice(PAGE_PATHS))}?{query}"


def make_wide_query_url(rng: random.Random, row_idx: int) -> str:
    params = [("session_id", f"s_{row_idx}"), ("dup", "first")]
    for i in range(40):
        key = f"p{i:02d}"
        value = rng.choice(
            [
                f"v{rng.randint(1000, 9999)}",
                f"item%20{rng.randint(100, 999)}",
                f"q+{rng.randint(10, 99)}",
            ]
        )
        params.append((key, value))
    params.extend(
        [
            ("utm_source", "first"),
            ("utm_source", rng.choice(UTM_SOURCES)),
            ("dup", "last"),
            ("empty", ""),
        ]
    )
    query = "&".join(f"{key}={value}" for key, value in params)
    return f"https://{rng.choice(DOMAINS)}{clean_path(rng.choice(PAGE_PATHS))}?{query}#section-{row_idx % 10}"


def generate_url_row(row_idx: int, seed: int) -> dict:
    rng = make_rng(seed, row_idx)
    url_short = make_short_url(rng)
    url_marketing = make_marketing_url(rng, row_idx)
    url_wide_query = make_wide_query_url(rng, row_idx)
    url_mixed = rng.choices(
        [url_marketing, url_short, url_wide_query],
        weights=[50, 35, 15],
        k=1,
    )[0]
    return {
        "url_short": url_short,
        "url_marketing": url_marketing,
        "url_wide_query": url_wide_query,
        "url_mixed": url_mixed,
    }


def generate_url_data(size_name: str, num_rows: int, seed: int) -> None:
    print(f"Generating URL {size_name} ({num_rows:,} rows) with seed={seed}...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"urls_{size_name}.parquet"

    conn = duckdb.connect()

    batch_size = 50_000
    rows_generated = 0
    temp_table_created = False

    while rows_generated < num_rows:
        batch_rows = min(batch_size, num_rows - rows_generated)
        batch = [generate_url_row(rows_generated + i, seed) for i in range(batch_rows)]

        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE batch (
                url_short VARCHAR,
                url_marketing VARCHAR,
                url_wide_query VARCHAR,
                url_mixed VARCHAR
            )
            """
        )
        conn.executemany(
            "INSERT INTO batch VALUES (?, ?, ?, ?)",
            [
                (
                    r["url_short"],
                    r["url_marketing"],
                    r["url_wide_query"],
                    r["url_mixed"],
                )
                for r in batch
            ],
        )

        if not temp_table_created:
            conn.execute("CREATE OR REPLACE TABLE data AS SELECT * FROM batch")
            temp_table_created = True
        else:
            conn.execute("INSERT INTO data SELECT * FROM batch")

        rows_generated += batch_rows
        print(f"  {rows_generated:,} / {num_rows:,}")

    generated_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        f"""
        COPY data TO '{output_path}' (
            FORMAT PARQUET,
            KV_METADATA {{
                seed: '{seed}',
                size: '{size_name}',
                rows: '{num_rows}',
                generated_at: '{generated_at}',
                schema_version: 'url_bench_v1'
            }}
        )
        """
    )
    conn.close()

    print(f"  Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate marketing-telemetry benchmark data (v2)"
    )
    parser.add_argument(
        "--size",
        choices=sorted(set(SIZES) | set(URL_SIZES)),
        help="Generate only this size (default: all)",
    )
    parser.add_argument(
        "--kind",
        choices=["all", "events", "urls", "numbers", "wide_flat", "keyed_pair"],
        default="all",
        help="Dataset kind to generate (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    if args.size:
        if args.kind in ("all", "events") and args.size in SIZES:
            generate_data(args.size, SIZES[args.size], args.seed)
        if args.kind in ("all", "urls") and args.size in URL_SIZES:
            generate_url_data(args.size, URL_SIZES[args.size], args.seed)
        if args.kind in ("all", "numbers") and args.size in NUMBERS_SIZES:
            generate_numbers_data(args.size, NUMBERS_SIZES[args.size], args.seed)
        if args.kind in ("all", "wide_flat") and args.size in WIDE_FLAT_SIZES:
            generate_wide_flat_data(args.size, WIDE_FLAT_SIZES[args.size], args.seed)
        if args.kind == "keyed_pair" and args.size in KEYED_PAIR_SIZES:
            generate_keyed_pair_data(args.size, KEYED_PAIR_SIZES[args.size], args.seed)
    else:
        if args.kind in ("all", "events"):
            for size_name, num_rows in SIZES.items():
                generate_data(size_name, num_rows, args.seed)
        if args.kind in ("all", "urls"):
            for size_name in URL_DEFAULT_SIZES:
                generate_url_data(size_name, URL_SIZES[size_name], args.seed)
        if args.kind in ("all", "numbers"):
            for size_name, num_rows in NUMBERS_SIZES.items():
                generate_numbers_data(size_name, num_rows, args.seed)
        if args.kind in ("all", "wide_flat"):
            for size_name, num_rows in WIDE_FLAT_SIZES.items():
                generate_wide_flat_data(size_name, num_rows, args.seed)
        if args.kind == "keyed_pair":
            for size_name, num_rows in KEYED_PAIR_SIZES.items():
                generate_keyed_pair_data(size_name, num_rows, args.seed)


if __name__ == "__main__":
    main()

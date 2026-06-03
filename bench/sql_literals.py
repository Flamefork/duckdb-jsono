import json


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sql_json(value: object) -> str:
    return sql_string(json.dumps(value, separators=(",", ":")))


def sql_typed_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return sql_string(value)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(sql_typed_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        fields = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"typed struct literal key must be a string, got: {key!r}"
                )
            fields.append(f"{sql_string(key)}: {sql_typed_literal(item)}")
        return "{" + ", ".join(fields) + "}"
    raise ValueError(f"unsupported typed literal value: {value!r}")

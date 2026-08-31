#!/usr/bin/env python3
"""
Test precise token-level alias replacer.
"""
import re

test_schema = {
    "schools": [{"name": "CDSCode"}, {"name": "School"}, {"name": "Street"}, {"name": "Phone"}, {"name": "OpenDate"}, {"name": "Charter"}],
    "frpm": [{"name": "CDSCode"}, {"name": "County Name"}, {"name": "Charter Funding Type"}, {"name": "Charter School (Y/N)"}, {"name": "FRPM Count (K-12)"}],
    "satscores": [{"name": "cds"}, {"name": "NumTstTakr"}, {"name": "AvgScrMath"}, {"name": "NumGE1500"}]
}

col_to_table = {}
for t, cols in test_schema.items():
    for c in cols:
        col_to_table[c["name"].lower()] = t

def fix_table_aliases(sql: str, col_to_table: dict) -> str:
    # 1. Build alias map
    alias_map = {}
    table_matches = re.findall(r'\b(FROM|JOIN)\s+[`"]?([\w\-]+)[`"]?\s+(?:AS\s+)?([a-zA-Z0-9_]+)\b', sql, re.IGNORECASE)
    for _, tbl, alias in table_matches:
        if alias.lower() not in ('inner', 'left', 'right', 'outer', 'cross', 'join', 'on', 'where', 'group', 'order', 'limit'):
            alias_map[alias] = tbl

    def repl(m):
        alias = m.group(1)
        col = m.group(2).strip('`"\'')
        owning = col_to_table.get(col.lower())
        if owning and alias in alias_map:
            current_tbl = alias_map[alias]
            if current_tbl.lower() != owning.lower():
                for a, t in alias_map.items():
                    if t.lower() == owning.lower():
                        return f"{a}.`{col}`"
                return f"`{col}`"
        return m.group(0)

    return re.sub(r'\b([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+|`[^`]+`)\b', repl, sql)

sample = "SELECT DISTINCT s.Phone FROM schools s INNER JOIN frpm f ON s.CDSCode = f.CDSCode WHERE f.OpenDate > '2000-01-01' AND f.`Charter School (Y/N)` = 1"
print("Original:", sample)
print("Fixed:   ", fix_table_aliases(sample, col_to_table))

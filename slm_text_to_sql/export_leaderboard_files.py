#!/usr/bin/env python3
import os
import json

benchmark_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_summary.json")
with open(benchmark_file, "r") as f:
    data = json.load(f)

detailed = data.get("detailed_results", [])

# 1. Export Spider format (line-by-line SQL)
spider_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_spider_format.txt")
with open(spider_file, "w") as f:
    for idx, item in enumerate(detailed):
        sql = item["pred"].replace("\n", " ").strip()
        f.write(f"{sql}\n")
print(f"✅ Generated Spider format prediction file: {spider_file}")

# 2. Export BIRD format (JSON index dict)
bird_dict = {}
for idx, item in enumerate(detailed):
    sql = item["pred"].replace("\n", " ").strip()
    bird_dict[str(idx)] = sql

bird_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_bird_format.json")
with open(bird_file, "w") as f:
    json.dump(bird_dict, f, indent=2)
print(f"✅ Generated BIRD format prediction file: {bird_file}")

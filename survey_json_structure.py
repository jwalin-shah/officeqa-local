#!/usr/bin/env python3
"""Sample 10 random corpus_json files and report structure without dumping content."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

CORPUS = Path(__file__).parent / "corpus_json"


def walk_keys(obj, prefix="", out=None, depth=0, max_depth=4):
    if out is None:
        out = Counter()
    if depth > max_depth:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            out[path] += 1
            walk_keys(v, path, out, depth + 1, max_depth)
    elif isinstance(obj, list) and obj:
        walk_keys(obj[0], prefix + "[]", out, depth + 1, max_depth)
    return out


def survey_one(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)

    elem_types = Counter()
    doc = d.get("document") or {}
    elements = doc.get("elements") or []
    pages = doc.get("pages") or []

    for e in elements:
        if isinstance(e, dict):
            t = e.get("type") or "?"
            elem_types[t] += 1

    tables = [e for e in elements if isinstance(e, dict) and e.get("type") == "table"]
    first_table_keys = list(tables[0].keys()) if tables else []
    has_bbox = any(isinstance(e, dict) and e.get("bbox") for e in elements)

    return {
        "file": path.name,
        "num_elements": len(elements),
        "num_pages": len(pages),
        "element_types": dict(elem_types.most_common()),
        "num_tables": len(tables),
        "first_table_keys": first_table_keys,
        "has_bbox": has_bbox,
    }


def inspect_table_deep(path: Path) -> dict | None:
    with open(path) as f:
        d = json.load(f)
    elements = (d.get("document") or {}).get("elements") or []
    for e in elements:
        if isinstance(e, dict) and e.get("type") == "table":
            rep = {"keys": list(e.keys()), "type": e.get("type")}
            content = e.get("content")
            rep["content_type"] = type(content).__name__
            if isinstance(content, str):
                rep["content_len"] = len(content)
                rep["content_preview"] = content[:400]
            elif isinstance(content, list) and content:
                rep["content_list_len"] = len(content)
                rep["content_first_type"] = type(content[0]).__name__
                if isinstance(content[0], dict):
                    rep["content_first_keys"] = list(content[0].keys())
            elif isinstance(content, dict):
                rep["content_dict_keys"] = list(content.keys())
            rep["bbox"] = e.get("bbox")
            rep["description"] = (e.get("description") or "")[:200]
            return rep
    return None


def main() -> None:
    files = sorted(CORPUS.glob("*.json"))
    sample = random.sample(files, 10)
    print(f"Sampled 10 of {len(files)} corpus_json files\n")

    all_elem_types: Counter = Counter()
    table_key_sets: Counter = Counter()
    bbox_count = 0
    total_elements = 0
    total_tables = 0

    for p in sample:
        try:
            rep = survey_one(p)
        except Exception as e:
            print(f"{p.name}: ERROR {e}")
            continue
        print(f"=== {rep['file']} ===")
        print(f"  num_pages:       {rep['num_pages']}")
        print(f"  num_elements:    {rep['num_elements']}")
        print(f"  num_tables:      {rep['num_tables']}")
        print(f"  element_types:   {rep['element_types']}")
        print(f"  has_bbox:        {rep['has_bbox']}")
        if rep["first_table_keys"]:
            print(f"  table keys:      {rep['first_table_keys']}")
            table_key_sets[tuple(sorted(rep["first_table_keys"]))] += 1
        print()
        all_elem_types.update(rep["element_types"])
        total_elements += rep["num_elements"]
        total_tables += rep["num_tables"]
        if rep["has_bbox"]:
            bbox_count += 1

    print("=" * 60)
    print("AGGREGATE (across 10 sampled files)")
    print("=" * 60)
    print(f"total elements: {total_elements}")
    print(f"total tables:   {total_tables}")
    print(f"files w/ bbox:  {bbox_count}/10")
    print(f"element types:  {dict(all_elem_types.most_common())}")
    print(f"table key signatures seen: {len(table_key_sets)}")
    for keys, n in table_key_sets.most_common():
        print(f"  x{n}: {list(keys)}")

    print()
    print("=" * 60)
    print("DEEP TABLE INSPECTION (first sampled file with a table)")
    print("=" * 60)
    for p in sample:
        rep = inspect_table_deep(p)
        if rep:
            print(f"file: {p.name}")
            for k, v in rep.items():
                print(f"  {k}:")
                if isinstance(v, dict):
                    for kk, vv in list(v.items())[:30]:
                        print(f"    {kk}: {vv}")
                else:
                    print(f"    {v}")
            break


if __name__ == "__main__":
    main()

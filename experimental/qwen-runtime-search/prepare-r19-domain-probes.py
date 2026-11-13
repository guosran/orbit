#!/usr/bin/env python3
"""Prepare diagnostic assignments from the actual admitted domain, without search.

Costs are deliberately absent: the common evaluator must price these assignments
using the supplied fresh catalogue. This file is not an ORBIT search result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.catalogue.read_bytes()
    catalogue = json.loads(raw)
    groups = [group for group in catalogue["choice_groups"] if len(group["choices"]) > 1]
    sizes = [len(group["choices"]) for group in groups]
    if catalogue.get("execution_policy") == "source-dependency-ready":
        sizes.append(len(catalogue["ready_priority_policies"]))
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("Expected a nonempty registered choice domain")
    generator = random.Random(20261113)
    assignments = [[0] * len(sizes)]
    for i in range(49):
        values = [0] * len(sizes)
        d = i * len(sizes) // 49
        values[d] = 1 + i % (sizes[d] - 1) if sizes[d] > 1 else 0
        assignments.append(values)
    assignments.extend([[generator.randrange(size) for size in sizes] for _ in range(50)])
    with args.output.open("x") as stream:
        for i, values in enumerate(assignments):
            stream.write(json.dumps({"evaluation": i + 1, "assignment": values,
                "source_id": catalogue["source_id"],
                "priority_key_identity": "named-ready-priority-seed",
                "purpose": "actual-domain-fast-reference-preflight-not-search"}) + "\n")
    metadata = {"catalogue": str(args.catalogue.resolve()),
                "catalogue_sha256": hashlib.sha256(raw).hexdigest(),
                "source_id": catalogue["source_id"],
                "execution_registration_identity": catalogue["execution_registration_identity"],
                "assignments": len(assignments), "dimensions": len(sizes),
                "generator": "baseline-plus49single-group-plus50mixed-seed20261113",
                "old_costs_reused": False, "search_performed": False}
    with Path(str(args.output) + ".metadata.json").open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build exact-cover neighborhoods from executable source-bound regions.

Regions may overlap in the catalogue, but every choice partitions its source
component exactly once. A compound backend context remains an atomic bundle:
for example, the complete decoder cannot mix two incompatible firmware contexts
between recurrent invocations. No implementation price is synthesized here.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": str(path), "sha256": checksum}


def exact_cover_groups(operator_count, implementations, bundles=(), protected_covers=()):
    """Enumerate all partitions within components of overlapping alternatives.

    Taking the first uncovered operator avoids permutation duplicates. Unioning
    every atom's cover ensures that a choice cannot silently change the ABI of
    a different component. This is a domain constructor, not an optimizer.
    """
    by_id = {row["id"]: row for row in implementations}
    if len(by_id) != len(implementations) or operator_count <= 0:
        raise ValueError("Implementation IDs must be unique and source nonempty")
    covers = {}
    for identifier, row in by_id.items():
        members = row["source_operators"]
        if (not members or members != sorted(set(members))
                or any(type(i) is not int or not 0 <= i < operator_count
                       for i in members)):
            raise ValueError("Invalid typed source cover: " + identifier)
        covers[identifier] = set(members)

    atoms, bundled_ids = [], set()
    for bundle in bundles:
        identifiers = tuple(bundle)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("Empty or duplicated implementation in bundle")
        members = set()
        for identifier in identifiers:
            if identifier not in covers or members & covers[identifier]:
                raise ValueError("Unknown or overlapping compound implementation")
            members.update(covers[identifier])
        atoms.append((identifiers, members))
        bundled_ids.update(identifiers)
    atoms.extend(((identifier,), members) for identifier, members in covers.items()
                 if identifier not in bundled_ids)
    # Protection belongs to source instances, not just known implementation IDs.
    # A newly registered partial context must not escape its atomic bundle.
    for protected in protected_covers:
        protected = set(protected)
        if not protected:
            raise ValueError("Empty protected context cover")
        for identifiers, members in atoms:
            if members & protected and members != protected:
                raise ValueError("Partial implementation invades protected context cover")
            if members & protected and any(i not in bundled_ids for i in identifiers):
                raise ValueError("Protected context requires explicit compound bundle")
    atoms = sorted(set((ids, tuple(sorted(members))) for ids, members in atoms))

    parents = list(range(operator_count))

    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    covered = set()
    for _, members in atoms:
        first = find(members[0])
        for member in members:
            parents[find(member)] = first
        covered.update(members)
    if covered != set(range(operator_count)):
        raise ValueError("Executable implementations leave source coverage holes")
    components = {}
    for member in range(operator_count):
        components.setdefault(find(member), []).append(member)

    groups = []
    for members in sorted(components.values(), key=lambda row: row[0]):
        local_index = {member: i for i, member in enumerate(members)}
        options = [[] for _ in members]
        for identifiers, cover in atoms:
            if cover[0] not in local_index:
                continue
            mask = sum(1 << local_index[member] for member in cover)
            for member in cover:
                options[local_index[member]].append((identifiers, mask))
        full = (1 << len(members)) - 1
        choices = []

        # A complete large graph can have more regions than Python's recursion
        # limit, even with only two possible covers. Keep DFS state explicitly.
        pending = [(0, [])]
        while pending:
            used, selected = pending.pop()
            if used == full:
                ordered = sorted(selected, key=lambda identifier:
                                 (by_id[identifier]["source_operators"][0], identifier))
                choices.append({"id": "cover-" + digest(ordered)[:24],
                                "implementations": ordered})
                continue
            remaining = full ^ used
            first = (remaining & -remaining).bit_length() - 1
            for identifiers, mask in reversed(options[first]):
                if not used & mask:
                    pending.append((used | mask, selected + list(identifiers)))
        if not choices:
            raise ValueError("An overlap component has no executable exact cover")
        unique = {tuple(row["implementations"]): row for row in choices}
        groups.append({"id": "source-cover-" + str(members[0]),
                       "choices": list(unique.values())})
    return groups


def build(source, catalogue, additions, preserved_groups):
    if (source.get("source_id") != digest({k: v for k, v in source.items()
                                           if k != "source_id"})
            or catalogue["source_id"] != source["source_id"]):
        raise ValueError("Source identity mismatch")
    result = copy.deepcopy(catalogue)
    result.pop("execution_registration_identity", None)
    result["implementations"].extend(copy.deepcopy(additions.get("implementations", [])))
    runtime = result["runtime_registration"]["implementations"]
    for identifier, registration in additions.get("runtime_implementations", {}).items():
        if identifier in runtime:
            raise ValueError("Runtime registration would overwrite " + identifier)
        runtime[identifier] = copy.deepcopy(registration)
    for row in result["implementations"]:
        price = row.get("wall_ms")
        if (row.get("source_id") != source["source_id"]
                or not row.get("executable") or not row.get("numeric_admitted")
                or type(price) not in (int, float) or not math.isfinite(price)
                or price < 0 or row["id"] not in runtime):
            raise ValueError("Missing admitted execution, price or source binding: " + row["id"])
    bundles = []
    protected_covers = []
    by_id = {row["id"]: row for row in result["implementations"]}
    found = set()
    for group in catalogue["choice_groups"]:
        if group["id"] in preserved_groups:
            found.add(group["id"])
            choices = [choice["implementations"] for choice in group["choices"]]
            if not choices:
                raise ValueError("Empty protected context group")
            covers = [{i for name in choice for i in by_id[name]["source_operators"]}
                      for choice in choices]
            if any(cover != covers[0] for cover in covers):
                raise ValueError("Protected context alternatives change source coverage")
            protected_covers.append(covers[0])
            bundles.extend(choices)
    if found != set(preserved_groups):
        raise ValueError("Requested compound context group not found")
    bundles.extend(additions.get("compound_choices", []))
    result["choice_groups"] = exact_cover_groups(
        len(source["operators"]), result["implementations"], bundles, protected_covers)
    result["cover_domain"] = {
        "schema": "orbit.executable-cover-components.v1",
        "preserved_compound_context_groups": sorted(found),
        "component_count": len(result["choice_groups"]),
        "choices_per_component": [len(group["choices"]) for group in result["choice_groups"]],
        "all_exact_covers_enumerated_within_components": True,
        "prices_synthesized": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "catalogue", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--additions", type=Path)
    parser.add_argument("--preserve-group", action="append",
                        default=["decode.global-compiled-dataflow-context"])
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    result = build(read(args.source), read(args.catalogue),
                   read(args.additions) if args.additions else {}, args.preserve_group)
    result["cover_domain"].update({
        "producer": file_record(__file__), "parent_catalogue": file_record(args.catalogue),
        "additions": file_record(args.additions) if args.additions else None,
    })
    result["execution_registration_identity"] = digest(result)
    with args.output.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["cover_domain"]))


if __name__ == "__main__":
    main()

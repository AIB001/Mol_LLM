"""Compact structural evidence for free prose, derived only from frozen v2 facts.

This is input evidence, not a writing template or a claim of prose correctness.
Uses existing ring, component, named-group and attachment-path records. No RDKit
reconstruction, molecule-specific lookup, descriptors, model or network calls.
"""
from collections import Counter, defaultdict
from itertools import combinations

from readable_facts_v3 import _letter

SCHEMA_VERSION = "natural_structure_context/1"
_BOND = {"SINGLE": "-", "DOUBLE": "=", "TRIPLE": "#", "AROMATIC": ":",
         "DATIVE": "->", "ZERO": "~0~", "QUADRUPLE": "~quadruple~"}


def natural_structure_context(facts):
    if facts.get("schema_version") != "molecular_graph_facts/2":
        raise ValueError("Expected molecular_graph_facts/2")
    atoms = {atom["id"]: atom for atom in facts["atoms"]}
    bonds = {bond["id"]: bond for bond in facts["bonds"]}
    rings = {ring["id"]: ring for ring in facts["rings"]}
    ring_labels = {identifier: _letter(index) for index, identifier in enumerate(rings)}
    references, component_for_atom = {}, {}
    for ring in facts["rings"]:
        for position, atom_id in enumerate(ring["ordered_atom_ids"], 1):
            references.setdefault(atom_id, f"{ring_labels[ring['id']]}{position}")
    for index, component in enumerate(facts["nonring_components"], 1):
        for position, atom_id in enumerate(component["atom_ids"], 1):
            references[atom_id] = f"G{index}.{position}"
            component_for_atom[atom_id] = f"G{index}"
    if set(references) != set(atoms):
        raise ValueError("Incomplete source ring/nonring partition")

    tagged = {a["id"] for a in atoms.values() if a["chiral_tag"] != "CHI_UNSPECIFIED"
              or a["formal_charge"] or a["isotope"] or a["radical_electrons"]}
    for atom in atoms.values():
        if atom["chiral_tag"] != "CHI_UNSPECIFIED":
            tagged.update(atom["neighbor_ids"])
    for bond in bonds.values():
        if bond["stereo"] != "STEREONONE":
            tagged.update([bond["begin_atom_id"], bond["end_atom_id"], *bond["stereo_atom_ids"]])
    for center in facts["stereo"]["potential_centers"]:
        identifier = center["center_id"]
        if identifier in atoms:
            tagged.add(identifier)
        elif identifier in bonds:
            tagged.update([bonds[identifier]["begin_atom_id"], bonds[identifier]["end_atom_id"]])
    for group in facts["stereo"]["groups"]:
        tagged.update(group["atom_ids"])

    def formula(identifier, annotate=False):
        atom = atoms[identifier]
        value = (str(atom["isotope"]) if atom["isotope"] else "") + atom["element"]
        hydrogen = atom["total_hydrogens"]
        if hydrogen:
            value += "H" + (str(hydrogen) if hydrogen > 1 else "")
        charge = atom["formal_charge"]
        if charge:
            value += ("^" + str(abs(charge)) if abs(charge) > 1 else "") + ("+" if charge > 0 else "−")
        if atom["isotope"] or charge or atom["radical_electrons"]:
            value = "[" + value + "]"
        value += "·" * atom["radical_electrons"]
        if annotate and identifier in tagged and identifier in component_for_atom:
            value += "{" + references[identifier] + "}"
        return value

    def symbol(bond, start):
        if bond["bond_type"] == "DATIVE":
            return "->" if start == bond["begin_atom_id"] else "<-"
        return _BOND.get(bond["bond_type"], "~" + bond["bond_type"].lower() + "~")

    def endpoint(atom_id):
        return references[atom_id] + "[" + formula(atom_id) + "]"

    ring_notes = []
    for ring in facts["rings"]:
        orders = [bonds[identifier]["bond_type"].lower() for identifier in ring["ordered_bond_ids"]]
        ring_notes.append({"ring": ring_labels[ring["id"]], "size": ring["size"],
            "identity": ring["description"], "state": ring["state"],
            "ordered_cycle_atoms": "/".join(formula(identifier) for identifier in ring["ordered_atom_ids"]),
            "heteroatom_positions": {str(position): formula(identifier)
                for position, identifier in enumerate(ring["ordered_atom_ids"], 1) if atoms[identifier]["element"] != "C"},
            "cycle_bonds_including_closure": orders[0] if len(set(orders)) == 1 else orders})
    shared = [{"rings": [ring_labels[r] for r in relation["ring_ids"]],
        "relationship": relation["relationship"],
        "shared_positions": ["=".join(f"{ring_labels[m['ring_id']]}{m['position']}"
            for m in atoms[identifier]["ring_memberships"] if m["ring_id"] in relation["ring_ids"])
            for identifier in relation["shared_atom_ids"]],
        "shared_cycle_bonds": len(relation["shared_bond_ids"])} for relation in facts["ring_relationships"]]

    groups, linkers = [], []
    for number, component in enumerate(facts["nonring_components"], 1):
        included = set(component["atom_ids"])
        graph = {identifier: [] for identifier in included}
        node_text = {identifier: formula(identifier, True) for identifier in included}
        for bond_id in component["bond_ids"]:
            bond = bonds[bond_id]
            left, right = bond["begin_atom_id"], bond["end_atom_id"]
            graph[left].append((right, symbol(bond, left)))
            graph[right].append((left, symbol(bond, right)))
        for index, attachment in enumerate(component["attachments"]):
            leaf = f"endpoint:{index}"
            source_atom, inner = attachment["ring_atom_id"], attachment["fragment_atom_id"]
            bond = bonds[attachment["bond_id"]]
            graph[leaf] = [(inner, symbol(bond, source_atom))]
            graph[inner].append((leaf, symbol(bond, inner)))
            node_text[leaf] = endpoint(source_atom)
        root = "endpoint:0" if component["attachments"] else component["atom_ids"][0]
        visited = set()

        def tree(node, parent=None):
            if node in visited:
                raise ValueError("Nonring component is cyclic; compact tree notation is not valid")
            visited.add(node)
            children = [(neighbor, bond) for neighbor, bond in graph[node] if neighbor != parent]
            value = node_text[node]
            for neighbor, bond in children[:-1]:
                value += "(" + bond + tree(neighbor, node) + ")"
            if children:
                neighbor, bond = children[-1]
                value += bond + tree(neighbor, node)
            return value

        group_formula = tree(root)
        if visited != set(graph):
            raise ValueError("Disconnected source nonring component")
        groups.append({"group": f"G{number}", "name": component.get("named_group"),
                       "bonded_formula": group_formula})
        for left, right in combinations(component["attachments"], 2):
            start, end = left["fragment_atom_id"], right["fragment_atom_id"]
            if start == end:
                path_atoms, path_bonds = [start], []
            else:
                record = next((path for path in component["paths_from_attachments"]
                    if {path["atom_ids"][0], path["atom_ids"][-1]} == {start, end}), None)
                if record is None:
                    raise ValueError("Missing source attachment-to-attachment path")
                path_atoms, path_bonds = list(record["atom_ids"]), list(record["bond_ids"])
                if path_atoms[0] != start:
                    path_atoms.reverse()
                    path_bonds.reverse()
            path_text = formula(path_atoms[0])
            for a, bond_id, b in zip(path_atoms, path_bonds, path_atoms[1:]):
                path_text += symbol(bonds[bond_id], a) + formula(b)
            linkers.append({"group": f"G{number}", "from": endpoint(left["ring_atom_id"]),
                "to": endpoint(right["ring_atom_id"]), "internal_path": path_text,
                "internal_carbon_atoms": sum(atoms[a]["element"] == "C" for a in path_atoms),
                "internal_CH2_atoms": sum(atoms[a]["element"] == "C" and atoms[a]["total_hydrogens"] == 2 for a in path_atoms),
                "entry_bond": left["bond_type"].lower(), "exit_bond": right["bond_type"].lower()})

    multiplicities, relative = [], []
    for ring in facts["rings"]:
        label = ring_labels[ring["id"]]
        fragments = {fragment["id"]: fragment for fragment in ring["external_fragments"]}
        named = defaultdict(list)
        for fragment in fragments.values():
            if fragment.get("named_group"):
                named[fragment["named_group"]].extend(f"{label}{a['ring_position']}" for a in fragment["attachments"])
        for name, sites in named.items():
            multiplicities.append({"ring": label, "group": name, "count": len(sites), "sites": sites})

        def branch(identifier):
            fragment = fragments[identifier]
            attachment = fragment["attachments"][0]
            atom = attachment["fragment_atom_id"]
            location = component_for_atom.get(atom, "ring-site " + references[atom])
            return {"at": f"{label}{attachment['ring_position']}", "branch": location,
                    "name": fragment.get("named_group"), "root": formula(atom)}

        for relation in ring["substituent_relations"]:
            relative.append({"ring": label, "left": branch(relation["left_fragment_id"]),
                "right": branch(relation["right_fragment_id"]), "relationship": relation["relationship"]})

    atom_stereo = [{"at": references[a["id"]], "tag": a["chiral_tag"], "CIP": a.get("cip_code"),
                   "ordered_graph_neighbors": [references[n] for n in a["neighbor_ids"]],
                   "permutation": a.get("chiral_permutation")} for a in atoms.values() if a["chiral_tag"] != "CHI_UNSPECIFIED"]
    bond_stereo = [{"between": [references[b["begin_atom_id"]], references[b["end_atom_id"]]],
                   "assignment": b["stereo"].replace("STEREO", ""),
                   "ordered_references": [references[a] for a in b["stereo_atom_ids"]]}
                  for b in bonds.values() if b["stereo"] != "STEREONONE"]
    unassigned = []
    for center in facts["stereo"]["potential_centers"]:
        if center["specified"] == "Specified":
            continue
        identifier = center["center_id"]
        location = references.get(identifier)
        if location is None:
            bond = bonds[identifier]
            location = references[bond["begin_atom_id"]] + "–" + references[bond["end_atom_id"]]
        unassigned.append({"at": location, "kind": center["type"], "state": center["specified"]})
    enhanced = [{"type": group["type"], "atoms": [references[a] for a in group["atom_ids"]],
                 "bonds": [[references[bonds[b]["begin_atom_id"]], references[bonds[b]["end_atom_id"]]]
                           for b in group["bond_ids"]]} for group in facts["stereo"]["groups"]]
    return {"schema_version": SCHEMA_VERSION,
        "interpretation": "Structural cross-check notes, not a prose template. A1 means ring A local position 1; G labels identify shared groups. Labels are not IUPAC locants and must not appear in the final paragraph. Cycle entries follow consecutive positions and close to the first. Formula H counts include separately listed H atoms; parentheses are branches, not extra molecules. Aromatic bonds are delocalized. Overlapping SSSR cycles are not necessarily independent rings.",
        "rings": ring_notes, "shared_ring_topology": shared,
        "independent_cycle_count": len(bonds) - len(atoms) + len(facts["connected_components"]),
        "groups": groups, "ring_linkers": linkers,
        "direct_ring_connections": [{"from": endpoint(b["begin_atom_id"]), "to": endpoint(b["end_atom_id"]),
                                     "bond": b["bond_type"].lower(), "intervening_atoms": 0}
                                    for b in facts["direct_ring_bonds"]],
        "named_group_multiplicity": multiplicities, "benzene_relations": relative,
        "stereo": {"specified_atoms": atom_stereo, "specified_bonds": bond_stereo,
                   "unassigned_potential_sites": unassigned, "enhanced_groups": enhanced,
                   "scope": "Only supplied assignments; unassigned does not mean achiral or racemic. Tags use RDKit graph-neighbor order."},
        "charged_sites": [{"at": references[a["id"]], "formal_charge": a["formal_charge"]}
                          for a in atoms.values() if a["formal_charge"]],
        "charge_scope": "Charges are those encoded in the source; an empty list means all recorded atoms have zero formal charge, not a pH prediction."}


build_context = natural_structure_context

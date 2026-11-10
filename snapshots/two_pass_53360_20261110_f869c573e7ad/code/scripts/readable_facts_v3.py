"""Deterministic readable rendering of the frozen v2 chemical graph facts.

The renderer names distinct rings Ring A, Ring B, ... and spells out complete
paths, carbon counts, attachment sites, branches and relative benzene positions.
It performs no molecule-specific lookup and makes no model/API calls. Positions
remain local cycle positions, never IUPAC locants. Returned prose is evidence
for narration, not a guaranteed fluent final molecular description.
"""

from collections import Counter, deque
from itertools import combinations


_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
          6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
          12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
          16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}
_ELEMENTS = {"C": "carbon", "N": "nitrogen", "O": "oxygen", "F": "fluorine",
             "Cl": "chlorine", "Br": "bromine", "I": "iodine", "H": "hydrogen",
             "P": "phosphorus", "S": "sulfur", "B": "boron", "Si": "silicon"}


def _word(number):
    return _WORDS.get(number, str(number))


def _letter(index):
    result = ""
    while True:
        index, remainder = divmod(index, 26)
        result = chr(65 + remainder) + result
        if index == 0:
            return result
        index -= 1


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else "none"


def readable_facts_v3(facts):
    """Return readable, graph-complete evidence from structural_facts_v2 output."""
    if facts.get("schema_version") != "molecular_graph_facts/2":
        raise ValueError("readable_facts_v3 requires molecular_graph_facts/2")
    atoms = {a["id"]: a for a in facts["atoms"]}
    bonds = {b["id"]: b for b in facts["bonds"]}
    rings = {r["id"]: r for r in facts["rings"]}
    labels = {ring_id: "Ring " + _letter(i) for i, ring_id in enumerate(rings)}
    graph = {atom_id: [] for atom_id in atoms}
    for bond in bonds.values():
        a, b = bond["begin_atom_id"], bond["end_atom_id"]
        graph[a].append((b, bond))
        graph[b].append((a, bond))

    def atom_text(atom_id, hydrogen=True):
        atom = atoms[atom_id]
        text = f"{_ELEMENTS.get(atom['element'], atom['element'])} {atom_id}"
        if hydrogen:
            text += f" (H{atom['total_hydrogens']})"
        if atom["formal_charge"]:
            text += f" (formal charge {atom['formal_charge']:+d})"
        if atom["isotope"]:
            text += f" (isotope {atom['isotope']})"
        if atom["radical_electrons"]:
            text += f" ({_word(atom['radical_electrons'])} radical electrons)"
        return text

    def bond_text(bond):
        return bond["bond_type"].lower().replace("_", " ") + " bond"

    def sites(atom_id):
        return _join(f"{labels[m['ring_id']]} local position {m['position']}"
                     for m in atoms[atom_id]["ring_memberships"])

    def fragment_label(fragment):
        name = fragment.get("named_group")
        return f"{name} group {fragment['id']}" if name else f"complete attached branch {fragment['id']}"

    def counts(atom_ids):
        count = Counter(atoms[a]["element"] for a in atom_ids)
        return _join(f"{_word(n)} {_ELEMENTS.get(e, e)} atom{'s' if n != 1 else ''}"
                     for e, n in sorted(count.items()))

    def path(start, end, included):
        todo, visited = deque([(start, [start])]), {start}
        while todo:
            current, traversed = todo.popleft()
            if current == end:
                return traversed
            for neighbor, _ in graph[current]:
                if neighbor in included and neighbor not in visited:
                    visited.add(neighbor)
                    todo.append((neighbor, traversed + [neighbor]))
        raise ValueError("Disconnected component in structural facts")

    def edge(a, b):
        return next(bond for neighbor, bond in graph[a] if neighbor == b)

    ring_facts, attachment_facts, multiplicity_facts, relative_facts = [], [], [], []
    type_counts = Counter(r["description"] for r in rings.values())
    for description, count in type_counts.items():
        selected = [labels[r["id"]] for r in rings.values() if r["description"] == description]
        ring_facts.append(f"There {'is' if count == 1 else 'are'} {_word(count)} distinct {description}"
                          f"{'s' if count != 1 else ''}: {_join(selected)}.")
    if not rings:
        ring_facts.append("The supplied graph has no rings.")
    for ring_id, ring in rings.items():
        label = labels[ring_id]
        cycle = ring["ordered_atom_ids"]
        sequence = "; ".join(f"position {i}: {atom_text(a)}" for i, a in enumerate(cycle, 1))
        orders = [bonds[b]["bond_type"] for b in ring["ordered_bond_ids"]]
        if len(set(orders)) == 1:
            bond_sentence = f"Every consecutive cycle bond, including last-to-first closure, is {orders[0].lower()}."
        else:
            bond_sentence = "Cycle bonds are " + "; ".join(
                f"positions {i + 1} to {(i + 1) % len(cycle) + 1}: {order.lower()}"
                for i, order in enumerate(orders)) + "."
        ring_facts.append(f"{label} ({ring_id}) is a {ring['description']}. Ordered local cycle: {sequence}. {bond_sentence}")
        named = {}
        for fragment in ring["external_fragments"]:
            full_label = fragment_label(fragment)
            descriptions = []
            for attachment in fragment["attachments"]:
                descriptions.append(f"{label} position {attachment['ring_position']} "
                                    f"({atom_text(attachment['ring_atom_id'], False)}) connects by "
                                    f"{bond_text(bonds[attachment['bond_id']])} to "
                                    f"{atom_text(attachment['fragment_atom_id'])}")
            attachment_facts.append(f"{full_label}: {'; '.join(descriptions)}. "
                                    f"Its complete external atom set is {_join(fragment['atom_ids'])}. "
                                    f"Connection type: {fragment['kind'].replace('_', ' ')}.")
            if fragment.get("named_group"):
                named.setdefault(fragment["named_group"], []).append(fragment)
                if fragment["named_group"] == "isopropylamino":
                    root = fragment["attachments"][0]["fragment_atom_id"]
                    included = set(fragment["atom_ids"])
                    carbons = [a for a, b in graph[root] if a in included and atoms[a]["element"] == "C"]
                    if len(carbons) == 1:
                        central = carbons[0]
                        methyl = [a for a, b in graph[central] if a in included and atoms[a]["element"] == "C"
                                  and atoms[a]["total_hydrogens"] == 3]
                        attachment_facts.append(f"In {full_label}, the ring is attached through nitrogen {root}. "
                                                f"That nitrogen bonds to carbon {central}, which branches to "
                                                f"{_word(len(methyl))} distinct methyl carbons {_join(methyl)}. "
                                                "This is an N-isopropyl attachment.")
        for name, fragments in named.items():
            locations = [f"position {f['attachments'][0]['ring_position']} ({f['id']})" for f in fragments]
            multiplicity_facts.append(f"{label} bears exactly {_word(len(fragments))} separate {name} "
                                      f"group{'s' if len(fragments) != 1 else ''}, at {_join(locations)}.")
        fragments_by_id = {f["id"]: f for f in ring["external_fragments"]}
        for relation in ring["substituent_relations"]:
            left, right = (fragments_by_id[relation[key]] for key in ("left_fragment_id", "right_fragment_id"))
            left_pos, right_pos = relation["left_position"], relation["right_position"]
            separation = min(abs(left_pos - right_pos), len(cycle) - abs(left_pos - right_pos))
            relative_facts.append(f"On {label}, {fragment_label(left)} at local position {left_pos} is "
                                  f"{relation['relationship']} to {fragment_label(right)} at local position {right_pos}. "
                                  f"Their shortest separation around this ring is {_word(separation)} ring bond"
                                  f"{'s' if separation != 1 else ''}.")
    for relationship in facts["ring_relationships"]:
        left, right = [labels[r] for r in relationship["ring_ids"]]
        shared = _join(f"{a} ({sites(a)})" for a in relationship["shared_atom_ids"])
        ring_facts.append(f"{left} and {right} have a {relationship['relationship'].replace('_', ' ')} relationship; "
                          f"shared atoms: {shared}; shared bonds: {_join(relationship['shared_bond_ids'])}.")

    component_facts, linker_facts = [], []
    for component in facts["nonring_components"]:
        included = set(component["atom_ids"])
        sentence = f"Acyclic component {component['id']} contains exactly {counts(component['atom_ids'])}. "
        if component.get("named_group"):
            sentence += f"Its complete rooted group is {component['named_group']}. "
        if component["bond_ids"]:
            sentence += "Complete internal connectivity: " + "; ".join(
                f"{atom_text(bonds[b]['begin_atom_id'])} to {atom_text(bonds[b]['end_atom_id'])} by {bond_text(bonds[b])}"
                for b in component["bond_ids"]) + ". "
        else:
            sentence += "Its sole atom is " + atom_text(component["atom_ids"][0]) + ". "
        sentence += "All ring attachments: " + _join(
            f"{atom_text(a['fragment_atom_id'], False)} by {bond_text(bonds[a['bond_id']])} "
            f"to {sites(a['ring_atom_id'])} ({atom_text(a['ring_atom_id'], False)})"
            for a in component["attachments"]) + "."
        component_facts.append(sentence)
        for left, right in combinations(component["attachments"], 2):
            ordered = path(left["fragment_atom_id"], right["fragment_atom_id"], included)
            ordered_text = ""
            for i, atom_id in enumerate(ordered, 1):
                if i > 1:
                    ordered_text += " then " + bond_text(edge(ordered[i - 2], atom_id)) + " to "
                ordered_text += f"linker position {i}: {atom_text(atom_id)}"
            sentence = (f"The connector from {sites(left['ring_atom_id'])} to {sites(right['ring_atom_id'])} "
                        f"enters by {bond_text(bonds[left['bond_id']])}; its ordered internal path is {ordered_text}; "
                        f"it exits by {bond_text(bonds[right['bond_id']])}. "
                        f"This path contains exactly {counts(ordered)}")
            if all(atoms[a]["element"] == "C" for a in ordered):
                sentence += f", making it a {_word(len(ordered))}-carbon connecting path"
            sentence += ". "
            unsaturated = [(i + 1, i + 2, edge(a, b)) for i, (a, b) in enumerate(zip(ordered, ordered[1:]))
                           if edge(a, b)["bond_type"] != "SINGLE"]
            for start, end, bond in unsaturated:
                sentence += f"The {bond_text(bond)} lies between linker positions {start} and {end}. "
            off_path = included - set(ordered)
            if off_path:
                sentence += "Additional branches off this path contain " + _join(sorted(off_path, key=lambda a: int(a[1:]))) + "; their exact bonds are in the component connectivity."
            linker_facts.append(sentence.strip())
    for bond in facts["direct_ring_bonds"]:
        linker_facts.append(f"{sites(bond['begin_atom_id'])} ({atom_text(bond['begin_atom_id'], False)}) connects directly "
                            f"to {sites(bond['end_atom_id'])} ({atom_text(bond['end_atom_id'], False)}) by "
                            f"{bond_text(bonds[bond['bond_id']])}; there are no intervening linker atoms.")

    stereo_facts = []
    for atom in atoms.values():
        if atom["chiral_tag"] != "CHI_UNSPECIFIED":
            sentence = f"Atom {atom['id']} has supplied chirality {atom['chiral_tag']}, relative to ordered neighbors {_join(atom['neighbor_ids'])}."
            if atom.get("cip_code"):
                sentence += f" Its RDKit CIP label is {atom['cip_code']}."
            if atom.get("chiral_permutation") is not None:
                sentence += f" Its non-tetrahedral chiral permutation is {atom['chiral_permutation']}."
            stereo_facts.append(sentence)
    for bond in bonds.values():
        if bond["stereo"] != "STEREONONE":
            stereo_facts.append(f"Bond {bond['id']} between {bond['begin_atom_id']} and {bond['end_atom_id']} has supplied "
                                f"stereochemistry {bond['stereo']}; reference atoms are {_join(bond['stereo_atom_ids'])}.")
    for center in facts["stereo"]["potential_centers"]:
        if center["specified"] != "Specified":
            stereo_facts.append(f"Potential stereo center {center['center_id']} ({center['type']}) is {center['specified'].lower()}; "
                                "do not invent an absolute configuration.")
    for group in facts["stereo"]["groups"]:
        stereo_facts.append(f"Enhanced stereo group {group['read_id']} has type {group['type']} and includes atoms "
                            f"{_join(group['atom_ids'])} and bonds {_join(group['bond_ids'])}.")
    if not stereo_facts:
        stereo_facts.append("The supplied graph specifies no stereochemical assignment; this is not a claim of achirality.")
    return {"schema_version": "readable_molecular_facts/3",
            "canonical_isomeric_smiles": facts["canonical_isomeric_smiles"],
            "conventions": "Ring A, Ring B, etc. identify distinct rings. Ring and linker positions are local ordered positions, not IUPAC locants. Atom aN is a graph identity, never a chemical locant. Hn is the total number of attached hydrogen atoms. Shared external branches seen from different rings are the same atoms, not extra substituents. Aromatic bonds are delocalized, not fixed alternating single/double bonds.",
            "ring_facts": ring_facts, "attachment_facts": attachment_facts,
            "multiplicity_facts": multiplicity_facts, "relative_position_facts": relative_facts,
            "component_facts": component_facts, "linker_facts": linker_facts,
            "stereo_facts": stereo_facts}


readable_structural_facts_v3 = readable_facts_v3

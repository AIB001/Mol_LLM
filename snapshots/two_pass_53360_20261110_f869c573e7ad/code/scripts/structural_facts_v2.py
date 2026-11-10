"""Lossless molecular graph facts and a compact, graph-backed narration view.

``structural_facts_v2(mol, original_smiles=None)`` accepts a sanitized RDKit
molecule and returns JSON-compatible data (schema ``molecular_graph_facts/2``).
Atoms a1..aN and bonds b1..bM retain the supplied RDKit index order. Ring IDs
r1..rK identify *individual* symmetrized-SSSR cycles, ordered deterministically
by atom indices; equal ring descriptions never imply equal ring identities.
IDs are stable for the same indexed graph, not across arbitrary atom renumbering.
Ring positions are one-based positions along each displayed cycle, NOT IUPAC
locants. The symmetrized SSSR is a cycle set, not an enumeration of all cycles.

``atoms`` and ``bonds`` are authoritative, including hydrogen representation,
isotopes, formal charges, bond orders, neighbor order and stereo metadata.
``mol_from_structural_facts_v2`` reconstructs that chemical graph. Conformers,
arbitrary RDKit properties, query atoms/bonds and substance groups are outside
this schema. Original SMILES text cannot be recovered from a Mol: supply it to
preserve it verbatim. No standardization, neutralization or tautomerization is
performed. Aromatic order 1.5 is a delocalized aromatic bond, not an inferred
single/double Kekule assignment. Non-tetrahedral chiral permutations and CXSMILES
are retained; absence of assigned stereo does not assert achirality.

``nonring_components`` are the components after removing *all* ring atoms.
Each ring's ``external_fragments`` instead removes only that particular ring,
so a full side chain includes any attached rings, every branch and every bond.
Fragment atom/bond lists refer to the authoritative graph. Mapped fragment
SMILES use map number n for atom aN (replacing user maps in this display only);
they exclude attachment bonds and may be open aromatic fragments. They are
graph displays, not separately capped or independently sanitized molecules.
Explicit attachment records contain every boundary bond and its local ring
position. Paths describe connectivity, not inferred conformations.

``compact_facts_v2`` (alias ``compact_llm_facts``) is the prompt-facing view.
It keeps the complete atom/bond graph, ordered rings, whole substituent atom
sets, relative benzene attachment positions and specified/unspecified stereo,
while dropping duplicate fragment strings and redundant derived metadata.
"""

from collections import Counter, deque
from itertools import combinations

from rdkit import Chem, rdBase


SCHEMA_VERSION = "molecular_graph_facts/2"


def _aid(index):
    return f"a{index + 1}"


def _bid(index):
    return f"b{index + 1}"


def _index(identifier):
    return int(identifier[1:]) - 1


def _canonical_cycle(cycle):
    cycle = tuple(cycle)
    pos = cycle.index(min(cycle))
    forward = cycle[pos:] + cycle[:pos]
    reverse = (forward[0],) + tuple(reversed(forward[1:]))
    return min(forward, reverse)


def _components(mol, atom_indices):
    remaining = set(atom_indices)
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component, stack = {seed}, [seed]
        while stack:
            for neighbor in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                idx = neighbor.GetIdx()
                if idx in remaining:
                    remaining.remove(idx)
                    component.add(idx)
                    stack.append(idx)
        result.append(component)
    return result


def _internal_bonds(mol, atoms):
    return [b.GetIdx() for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]


def _path_in_component(mol, atoms, start, end):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        idx, path = queue.popleft()
        if idx == end:
            return path
        for neighbor in sorted(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()):
            if neighbor in atoms and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    raise ValueError("Disconnected fragment")


def _path_record(mol, path):
    bonds = [mol.GetBondBetweenAtoms(a, b) for a, b in zip(path, path[1:])]
    return {"atom_ids": [_aid(i) for i in path],
            "elements": [mol.GetAtomWithIdx(i).GetSymbol() for i in path],
            "bond_ids": [_bid(b.GetIdx()) for b in bonds],
            "bond_types": [str(b.GetBondType()) for b in bonds],
            "bond_orders": [b.GetBondTypeAsDouble() for b in bonds]}


# A name is emitted only when the complete, singly attached component matches
# the corresponding rooted query. This intentionally misses many valid names.
_GROUP_QUERIES = [
    ("methoxy", Chem.MolFromSmarts("[O;X2;H0;+0]-[C;X4;H3;+0]")),
    ("methylaminomethyl", Chem.MolFromSmarts("[C;X4;H2;+0]-[N;X3;H1;+0]-[C;X4;H3;+0]")),
    ("isopropylamino", Chem.MolFromSmarts("[N;X3;H1;+0]-[C;X4;H1;+0](-[C;X4;H3;+0])-[C;X4;H3;+0]")),
    ("methyl", Chem.MolFromSmarts("[C;X4;H3;+0]")),
    ("hydroxy", Chem.MolFromSmarts("[O;X2;H1;+0]")),
    ("fluoro", Chem.MolFromSmarts("[F;X1;+0]")),
    ("chloro", Chem.MolFromSmarts("[Cl;X1;+0]")),
    ("bromo", Chem.MolFromSmarts("[Br;X1;+0]")),
    ("iodo", Chem.MolFromSmarts("[I;X1;+0]")),
]


def _named_group(mol, component, attachments):
    if len(attachments) != 1 or attachments[0][2].GetBondType() != Chem.BondType.SINGLE:
        return None
    root = attachments[0][1]
    # Graph H atoms may be retained explicitly by the parser; both forms mean
    # the same group, while isotope/charge details stay in the atom records.
    heavy = {i for i in component if mol.GetAtomWithIdx(i).GetAtomicNum() != 1}
    for name, query in _GROUP_QUERIES:
        if query.GetNumAtoms() != len(heavy):
            continue
        for match in mol.GetSubstructMatches(query, uniquify=True):
            if match[0] == root and set(match) == heavy:
                return name
    return None


def _ring_description(mol, cycle, bonds):
    elements = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in cycle)
    aromatic = all(b.GetIsAromatic() for b in bonds)
    saturated = all(b.GetBondType() == Chem.BondType.SINGLE for b in bonds)
    name = None
    if len(cycle) == 6 and elements == {"C": 6} and aromatic:
        name = "benzene"
    elif len(cycle) == 6 and elements == {"C": 4, "N": 1, "O": 1} and saturated:
        npos = next(i for i, idx in enumerate(cycle) if mol.GetAtomWithIdx(idx).GetSymbol() == "N")
        opos = next(i for i, idx in enumerate(cycle) if mol.GetAtomWithIdx(idx).GetSymbol() == "O")
        if (npos - opos) % 6 == 3:
            name = "morpholine"
    state = "aromatic" if aromatic else "saturated" if saturated else "nonaromatic unsaturated"
    description = f"{name} ring" if name else f"{len(cycle)}-membered {state} ring"
    return description, name, state, dict(sorted(elements.items()))


def _fragment_record(mol, mapped_mol, component, boundaries, memberships, fragment_id, ring=None):
    internal = _internal_bonds(mol, component)
    attachment_records = []
    for ring_atom, fragment_atom, bond in boundaries:
        record = {"ring_atom_id": _aid(ring_atom), "fragment_atom_id": _aid(fragment_atom),
                  "bond_id": _bid(bond.GetIdx()), "bond_type": str(bond.GetBondType()),
                  "order": bond.GetBondTypeAsDouble(),
                  "ring_memberships": memberships[ring_atom]}
        if ring is not None:
            record["ring_position"] = ring["ordered_atom_ids"].index(_aid(ring_atom)) + 1
        attachment_records.append(record)
    paths = []
    roots = sorted({fragment_atom for _, fragment_atom, _ in boundaries})
    leaves = sorted(i for i in component
                    if sum(n.GetIdx() in component for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 1)
    seen_paths = set()
    for start in roots:
        for end in sorted(set(roots + leaves) - {start}):
            path = _path_in_component(mol, component, start, end)
            identity = min(tuple(path), tuple(reversed(path)))
            if identity not in seen_paths:
                seen_paths.add(identity)
                paths.append(_path_record(mol, path))
    branches = []
    for idx in sorted(component):
        neighbors = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                           if n.GetAtomicNum() != 1)
        if len(neighbors) >= 3:
            branches.append({"atom_id": _aid(idx), "neighbor_ids": [_aid(i) for i in neighbors]})
    kind = "unattached_component" if not boundaries else "substituent" if len(boundaries) == 1 else "linker_or_multisite_component"
    if ring is not None and len(boundaries) > 1:
        own = ring["id"]
        overlapping_ring_ids = {membership["ring_id"] for idx, _, _ in boundaries
                                for membership in memberships[idx] if membership["ring_id"] != own}
        if any(membership["ring_id"] in overlapping_ring_ids for idx in component
               for membership in memberships.get(idx, [])):
            kind = "ring_system_extension"
    return {"id": fragment_id, "kind": kind,
            "atom_ids": [_aid(i) for i in sorted(component)],
            "bond_ids": [_bid(i) for i in internal],
            "mapped_fragment_smiles": Chem.MolFragmentToSmiles(
                mapped_mol, atomsToUse=sorted(component), bondsToUse=internal,
                isomericSmiles=True, canonical=True, allBondsExplicit=True, allHsExplicit=True),
            "attachments": attachment_records,
            "named_group": _named_group(mol, component, boundaries),
            "paths_from_attachments": paths, "branch_points": branches}


def structural_facts_v2(mol, original_smiles=None):
    """Serialize a sanitized Mol without mutating it or inferring missing stereo."""
    if mol is None or not isinstance(mol, Chem.Mol):
        raise TypeError("mol must be a sanitized RDKit Mol")
    if original_smiles is not None and not isinstance(original_smiles, str):
        raise TypeError("original_smiles must be a string or None")
    if any(a.HasQuery() for a in mol.GetAtoms()) or any(b.HasQuery() for b in mol.GetBonds()):
        raise ValueError("Query molecules are outside molecular_graph_facts/2")
    work = Chem.Mol(mol)
    work.UpdatePropertyCache(strict=True)
    Chem.AssignStereochemistry(work, cleanIt=False, force=True)
    cycles = sorted({_canonical_cycle(tuple(r)) for r in Chem.GetSymmSSSR(work)})
    mapped = Chem.Mol(work)
    for atom in mapped.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    memberships = {i: [] for i in range(work.GetNumAtoms())}
    rings = []
    for ring_index, cycle in enumerate(cycles):
        ring_id = f"r{ring_index + 1}"
        bonds = [work.GetBondBetweenAtoms(cycle[i], cycle[(i + 1) % len(cycle)])
                 for i in range(len(cycle))]
        description, name, state, element_counts = _ring_description(work, cycle, bonds)
        ring = {"id": ring_id, "description": description, "conservative_name": name,
                "state": state, "size": len(cycle), "element_counts": element_counts,
                "ordered_atom_ids": [_aid(i) for i in cycle],
                "ordered_bond_ids": [_bid(b.GetIdx()) for b in bonds],
                "positions": [{"position": pos, "atom_id": _aid(idx),
                               "element": work.GetAtomWithIdx(idx).GetSymbol()}
                              for pos, idx in enumerate(cycle, 1)]}
        for pos, idx in enumerate(cycle, 1):
            memberships[idx].append({"ring_id": ring_id, "position": pos})
        rings.append(ring)
    all_atoms = set(range(work.GetNumAtoms()))
    ring_atoms = {i for i, member in memberships.items() if member}

    def boundaries(component, excluded):
        result = []
        for bond in work.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in excluded and b in component:
                result.append((a, b, bond))
            elif b in excluded and a in component:
                result.append((b, a, bond))
        return sorted(result, key=lambda item: (item[0], item[1], item[2].GetIdx()))

    for ring, cycle in zip(rings, cycles):
        ring_set = set(cycle)
        fragments = []
        for component in _components(work, all_atoms - ring_set):
            attached = boundaries(component, ring_set)
            if attached:
                fragments.append(_fragment_record(work, mapped, component, attached, memberships,
                                                   f"{ring['id']}.s{len(fragments) + 1}", ring))
        ring["external_fragments"] = fragments
        relations = []
        if ring["conservative_name"] == "benzene":
            simple = [f for f in fragments if f["kind"] == "substituent"]
            for left, right in combinations(simple, 2):
                lpos = left["attachments"][0]["ring_position"]
                rpos = right["attachments"][0]["ring_position"]
                distance = min(abs(lpos - rpos), 6 - abs(lpos - rpos))
                if distance:
                    relations.append({"left_fragment_id": left["id"], "right_fragment_id": right["id"],
                                      "left_position": lpos, "right_position": rpos,
                                      "relationship": {1: "ortho", 2: "meta", 3: "para"}[distance]})
        ring["substituent_relations"] = relations

    ring_relationships = []
    for left, right in combinations(rings, 2):
        shared_atoms = set(left["ordered_atom_ids"]) & set(right["ordered_atom_ids"])
        if not shared_atoms:
            continue
        shared_bonds = set(left["ordered_bond_ids"]) & set(right["ordered_bond_ids"])
        kind = "spiro" if len(shared_atoms) == 1 else "fused" if len(shared_atoms) == 2 and len(shared_bonds) == 1 else "overlapping_cycles"
        ring_relationships.append({"ring_ids": [left["id"], right["id"]], "relationship": kind,
                                   "shared_atom_ids": sorted(shared_atoms, key=_index),
                                   "shared_bond_ids": sorted(shared_bonds, key=_index)})
    nonring = [_fragment_record(work, mapped, component, boundaries(component, ring_atoms), memberships,
                               f"c{i + 1}")
               for i, component in enumerate(_components(work, all_atoms - ring_atoms))]
    atoms = []
    for atom in work.GetAtoms():
        idx = atom.GetIdx()
        atoms.append({"id": _aid(idx), "index": idx, "element": atom.GetSymbol(),
                      "atomic_number": atom.GetAtomicNum(), "isotope": atom.GetIsotope(),
                      "formal_charge": atom.GetFormalCharge(), "radical_electrons": atom.GetNumRadicalElectrons(),
                      "aromatic": atom.GetIsAromatic(), "hybridization": str(atom.GetHybridization()),
                      "explicit_hydrogens": atom.GetNumExplicitHs(),
                      "implicit_hydrogens": atom.GetNumImplicitHs(),
                      "total_hydrogens": atom.GetTotalNumHs(includeNeighbors=True),
                      "no_implicit": atom.GetNoImplicit(), "atom_map_number": atom.GetAtomMapNum(),
                      "chiral_tag": str(atom.GetChiralTag()),
                      "chiral_permutation": atom.GetIntProp("_chiralPermutation") if atom.HasProp("_chiralPermutation") else None,
                      "cip_code": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
                      "neighbor_ids": [_aid(n.GetIdx()) for n in atom.GetNeighbors()],
                      "ring_memberships": memberships[idx]})
    bonds = [{"id": _bid(b.GetIdx()), "index": b.GetIdx(),
              "begin_atom_id": _aid(b.GetBeginAtomIdx()), "end_atom_id": _aid(b.GetEndAtomIdx()),
              "bond_type": str(b.GetBondType()), "order": b.GetBondTypeAsDouble(),
              "aromatic": b.GetIsAromatic(), "conjugated": b.GetIsConjugated(),
              "direction": str(b.GetBondDir()), "stereo": str(b.GetStereo()),
              "stereo_atom_ids": [_aid(i) for i in b.GetStereoAtoms()]}
             for b in work.GetBonds()]
    stereo_groups = [{"type": str(g.GetGroupType()), "read_id": g.GetReadId(), "write_id": g.GetWriteId(),
                      "atom_ids": [_aid(a.GetIdx()) for a in g.GetAtoms()],
                      "bond_ids": [_bid(b.GetIdx()) for b in g.GetBonds()]}
                     for g in work.GetStereoGroups()]
    potential = []
    for info in Chem.FindPotentialStereo(Chem.Mol(work), cleanIt=False, flagPossible=True):
        is_bond = str(info.type).startswith("Bond_")
        potential.append({"type": str(info.type), "center_id": _bid(info.centeredOn) if is_bond else _aid(info.centeredOn),
                          "specified": str(info.specified), "descriptor": str(info.descriptor),
                          "controlling_atom_ids": [_aid(i) if i < work.GetNumAtoms() else None
                                                   for i in info.controllingAtoms]})
    direct_ring_bonds = []
    for bond in work.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left = {m["ring_id"] for m in memberships[a]}
        right = {m["ring_id"] for m in memberships[b]}
        if left and right and not left & right:
            direct_ring_bonds.append({"bond_id": _bid(bond.GetIdx()), "begin_atom_id": _aid(a),
                                      "end_atom_id": _aid(b), "begin_rings": memberships[a],
                                      "end_rings": memberships[b], "bond_type": str(bond.GetBondType()),
                                      "order": bond.GetBondTypeAsDouble()})
    return {"schema_version": SCHEMA_VERSION, "rdkit_version": rdBase.rdkitVersion,
            "original_smiles": original_smiles,
            "canonical_isomeric_smiles": Chem.MolToSmiles(work, canonical=True, isomericSmiles=True),
            "canonical_cxsmiles": Chem.MolToCXSmiles(work),
            "indexed_mapped_smiles": Chem.MolToSmiles(mapped, canonical=False, isomericSmiles=True),
            "conventions": {"ids": "aN/bN use original RDKit index plus one; rN identify distinct cycles",
                            "ring_positions": "one-based local cycle positions, not IUPAC locants",
                            "ring_basis": "RDKit symmetrized SSSR; not all possible cycles",
                            "aromatic_order": "1.5 means aromatic delocalization, not alternating fixed bond orders",
                            "fragment_maps": "map N identifies aN; attachment bonds are recorded separately",
                            "stereo": "Only supplied stereo is specified; unspecified does not mean achiral"},
            "atoms": atoms, "bonds": bonds, "rings": rings,
            "ring_relationships": ring_relationships, "direct_ring_bonds": direct_ring_bonds,
            "nonring_components": nonring,
            "connected_components": [[_aid(i) for i in sorted(c)] for c in _components(work, all_atoms)],
            "stereo": {"groups": stereo_groups, "potential_centers": potential}}


def mol_from_structural_facts_v2(facts):
    """Rebuild the serialized chemical graph, retaining indexed stereo context."""
    if facts.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported structural-facts schema")
    rw = Chem.RWMol()
    for record in facts["atoms"]:
        if record["index"] != rw.GetNumAtoms():
            raise ValueError("Atoms must remain in serialized index order")
        atom = Chem.Atom(record["atomic_number"])
        atom.SetIsotope(record["isotope"])
        atom.SetFormalCharge(record["formal_charge"])
        atom.SetNumRadicalElectrons(record["radical_electrons"])
        atom.SetIsAromatic(record["aromatic"])
        atom.SetNumExplicitHs(record["explicit_hydrogens"])
        atom.SetNoImplicit(record["no_implicit"])
        atom.SetAtomMapNum(record["atom_map_number"])
        atom.SetChiralTag(getattr(Chem.ChiralType, record["chiral_tag"]))
        if record["chiral_permutation"] is not None:
            atom.SetIntProp("_chiralPermutation", record["chiral_permutation"])
        rw.AddAtom(atom)
    for record in facts["bonds"]:
        if record["index"] != rw.GetNumBonds():
            raise ValueError("Bonds must remain in serialized index order")
        rw.AddBond(_index(record["begin_atom_id"]), _index(record["end_atom_id"]),
                   getattr(Chem.BondType, record["bond_type"]))
        bond = rw.GetBondWithIdx(record["index"])
        bond.SetIsAromatic(record["aromatic"])
        bond.SetIsConjugated(record["conjugated"])
        bond.SetBondDir(getattr(Chem.BondDir, record["direction"]))
    # Stereo neighbors can be bonds added later, so set these only once all
    # bonds exist. Bond insertion order also preserves tetrahedral neighbor order.
    for record in facts["bonds"]:
        bond = rw.GetBondWithIdx(record["index"])
        if record["stereo_atom_ids"]:
            bond.SetStereoAtoms(*[_index(i) for i in record["stereo_atom_ids"]])
        bond.SetStereo(getattr(Chem.BondStereo, record["stereo"]))
    groups = []
    for record in facts["stereo"]["groups"]:
        if record.get("bond_ids"):
            raise ValueError("This RDKit version cannot rebuild bond-centered enhanced stereo groups")
        group = Chem.CreateStereoGroup(getattr(Chem.StereoGroupType, record["type"]), rw,
                                      [_index(i) for i in record["atom_ids"]], record["read_id"])
        group.SetWriteId(record["write_id"])
        groups.append(group)
    rw.SetStereoGroups(groups)
    rebuilt = rw.GetMol()
    Chem.SanitizeMol(rebuilt)
    # Recompute RDKit's ring-stereo caches as well as CIP labels. Without
    # cleanIt, relative stereo in symmetric substituted rings can produce a
    # different canonical traversal even with identical indexed chiral tags.
    Chem.AssignStereochemistry(rebuilt, cleanIt=True, force=True)
    return rebuilt


def compact_facts_v2(facts):
    """Return an indexed prompt view with every atom, bond and full side chain."""
    def number(identifier):
        return _index(identifier) + 1

    def compact_ring(ring):
        fragments = []
        for fragment in ring["external_fragments"]:
            fragments.append({"id": fragment["id"], "kind": fragment["kind"],
                              "atoms": [number(i) for i in fragment["atom_ids"]],
                              "attach": [[a["ring_position"], number(a["ring_atom_id"]),
                                          number(a["fragment_atom_id"]), number(a["bond_id"])]
                                         for a in fragment["attachments"]],
                              **({"name": fragment["named_group"]} if fragment["named_group"] else {})})
        return {"id": ring["id"], "description": ring["description"],
                "cycle": [number(i) for i in ring["ordered_atom_ids"]], "fragments": fragments,
                **({"relative_positions": [[r["left_fragment_id"], r["right_fragment_id"], r["relationship"]]
                                           for r in ring["substituent_relations"]]}
                   if ring["substituent_relations"] else {})}

    atom_rows = []
    atom_extras = []
    for atom in facts["atoms"]:
        atom_rows.append([atom["element"], atom["total_hydrogens"], atom["formal_charge"]])
        extras = {key: atom[key] for key in ("isotope", "radical_electrons", "atom_map_number") if atom[key]}
        if atom["aromatic"]:
            extras["aromatic"] = True
        if atom["chiral_tag"] != "CHI_UNSPECIFIED":
            extras.update({"chiral_tag": atom["chiral_tag"], "cip_code": atom["cip_code"],
                           "ordered_neighbors": [number(i) for i in atom["neighbor_ids"]]})
        if atom["chiral_permutation"] is not None:
            extras["chiral_permutation"] = atom["chiral_permutation"]
        if extras:
            atom_extras.append({"atom": number(atom["id"]), **extras})
    bond_rows, bond_stereo = [], []
    ordinary_types = {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}
    for bond in facts["bonds"]:
        order = bond["order"] if bond["bond_type"] in ordinary_types else bond["bond_type"]
        bond_rows.append([number(bond["begin_atom_id"]), number(bond["end_atom_id"]), order])
        if bond["stereo"] != "STEREONONE" or bond["direction"] != "NONE":
            bond_stereo.append({"bond": number(bond["id"]), "stereo": bond["stereo"],
                                "stereo_atoms": [number(i) for i in bond["stereo_atom_ids"]],
                                "direction": bond["direction"]})
    result = {"schema_version": "molecular_graph_prompt/2",
              "canonical_isomeric_smiles": facts["canonical_isomeric_smiles"],
              "legend": {"atoms": "one-based row N is atom N (aN): [element, total attached H, formal charge]",
                         "bonds": "one-based row N is bond N (bN): [begin atom, end atom, order/type]; 1.5 is aromatic",
                         "cycle": "ordered atom IDs; position = row index + 1, local not IUPAC",
                         "fragments": "entire external component after removing this ring, including branches and attached rings; overlapping views are not additional groups",
                         "attach": "[local ring position, ring atom, fragment atom, bond ID]",
                         "stereo": "retain supplied assignments; Unspecified does not assert achirality"},
              "atoms": atom_rows, "bonds": bond_rows, "rings": [compact_ring(r) for r in facts["rings"]],
              "ring_relationships": facts["ring_relationships"],
              "atom_details": atom_extras,
              "stereo": {**facts["stereo"], "bond_details": bond_stereo}}
    if facts["stereo"]["groups"]:
        result["canonical_cxsmiles"] = facts["canonical_cxsmiles"]
    if len(facts["connected_components"]) > 1:
        result["connected_components"] = [[number(i) for i in c] for c in facts["connected_components"]]
    return result


compact_llm_facts = compact_facts_v2

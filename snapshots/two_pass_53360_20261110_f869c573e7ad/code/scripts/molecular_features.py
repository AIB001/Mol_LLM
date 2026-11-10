"""Deterministic RDKit features and nonnumeric context for molecular annotation."""
from __future__ import annotations

import math
from collections import Counter
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Fragments, Lipinski, rdMolDescriptors as R

# This order is the feature-vector contract. Do not infer it from dictionary sorting.
FEATURES = [
    ("mol_wt", "g/mol", "Descriptors.MolWt"),
    ("exact_mol_wt", "Da", "Descriptors.ExactMolWt"),
    ("crippen_logp", "dimensionless", "Crippen.MolLogP"),
    ("tpsa", "angstrom^2", "CalcTPSA(includeSandP=False)"),
    ("hbd", "count", "CalcNumHBD"),
    ("hba", "count", "CalcNumHBA"),
    ("rotatable_bonds", "count", "CalcNumRotatableBonds(Strict)"),
    ("ring_count", "count", "CalcNumRings"),
    ("aromatic_rings", "count", "CalcNumAromaticRings"),
    ("aliphatic_rings", "count", "CalcNumAliphaticRings"),
    ("saturated_rings", "count", "CalcNumSaturatedRings"),
    ("heterocycles", "count", "CalcNumHeterocycles"),
    ("aromatic_heterocycles", "count", "CalcNumAromaticHeterocycles"),
    ("fraction_csp3", "fraction", "CalcFractionCSP3"),
    ("heavy_atom_count", "count", "Mol.GetNumHeavyAtoms"),
    ("heteroatom_count", "count", "CalcNumHeteroatoms"),
    ("formal_charge", "elementary_charge", "Chem.GetFormalCharge"),
    ("positive_charge_centers", "count", "atoms with formal charge > 0"),
    ("negative_charge_centers", "count", "atoms with formal charge < 0"),
    ("molar_refractivity", "cm^3/mol", "Crippen.MolMR"),
    ("amide_bonds", "count", "CalcNumAmideBonds"),
    ("spiro_atoms", "count", "CalcNumSpiroAtoms"),
    ("bridgehead_atoms", "count", "CalcNumBridgeheadAtoms"),
    ("chiral_centers", "count", "FindMolChiralCenters(includeUnassigned=True)"),
]
FEATURE_NAMES = [x[0] for x in FEATURES]
FRAGMENT_FUNCTIONS = {
    "amide": "fr_amide", "ester": "fr_ester", "ketone": "fr_ketone",
    "aldehyde": "fr_aldehyde", "carboxylic acid or carboxylate": "fr_COO",
    "alcohol": "fr_Al_OH", "phenol": "fr_phenol",
    "nitrile": "fr_nitrile", "nitro": "fr_nitro", "sulfonamide": "fr_sulfonamd",
    "sulfone": "fr_sulfone", "thioether": "fr_sulfide", "thiol": "fr_SH",
    "urea": "fr_urea", "carbamate": "fr_alkyl_carbamate", "imide": "fr_imide",
    "lactam": "fr_lactam", "lactone": "fr_lactone", "guanidine": "fr_guanido",
    "amidine": "fr_amidine", "benzene ring": "fr_benzene",
    "pyridine ring": "fr_pyridine", "imidazole ring": "fr_imidazole",
    "piperidine ring": "fr_piperdine", "piperazine ring": "fr_piperzine",
    "morpholine ring": "fr_morpholine", "furan ring": "fr_furan",
    "thiophene ring": "fr_thiophene", "thiazole ring": "fr_thiazole",
    "oxazole ring": "fr_oxazole", "tetrazole ring": "fr_tetrazole",
}
ETHER_PATTERN = Chem.MolFromSmarts("[#6;!$([#6]=[O,N,S])]-[OX2H0]-[#6;!$([#6]=[O,N,S])]")


def graph_context(mol, features):
    rings=list(mol.GetRingInfo().AtomRings())
    sizes={3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten",11:"eleven",12:"twelve"}
    ring_info=[]
    bonds=[]
    for i,ring in enumerate(rings):
        edges={tuple(sorted((ring[j],ring[(j+1)%len(ring)]))) for j in range(len(ring))}
        bonds.append(edges)
        aromatic=all(mol.GetBondBetweenAtoms(*e).GetIsAromatic() for e in edges)
        atoms=[]
        for idx in ring:
            atom=mol.GetAtomWithIdx(idx)
            label=atom.GetSymbol().lower() if atom.GetIsAromatic() else atom.GetSymbol()
            if atom.GetAtomicNum() in (7,8,16) and atom.GetTotalNumHs():label+="H"
            if atom.GetFormalCharge():label+="+" if atom.GetFormalCharge()>0 else "-"
            atoms.append(label)
        hetero=Counter(mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum()!=6)
        names={"N":"nitrogen","O":"oxygen","S":"sulfur","P":"phosphorus","B":"boron","Si":"silicon"}
        counts={1:"one",2:"two",3:"three",4:"four",5:"five",6:"six"}
        composition=", ".join(f"{counts.get(n,str(n))} {names.get(e,e)}" for e,n in hetero.items()) or "carbon only"
        ring_info.append(f"{chr(65+i)}: {sizes.get(len(ring),str(len(ring)))}-membered {'aromatic' if aromatic else 'nonaromatic'}; ring composition: {composition}; cyclic atom sequence {'-'.join(atoms)}")
    shared=[]
    for i in range(len(rings)):
        for j in range(i+1,len(rings)):
            overlap=set(rings[i])&set(rings[j])
            if not overlap:continue
            edges=bonds[i]&bonds[j]
            relation="spiro atom" if len(overlap)==1 else "fused bond" if len(edges)==1 and len(overlap)==2 else "shared ring path" if edges else "shared nonadjacent atoms"
            shared.append(f"{chr(65+i)}-{chr(65+j)}: {relation}")
    donors={match[0] for match in mol.GetSubstructMatches(Lipinski.HDonorSmarts)}
    acceptors={match[0] for match in mol.GetSubstructMatches(Lipinski.HAcceptorSmarts)}
    roles={}
    if len(donors)==features["hbd"] and len(acceptors)==features["hba"]:
        grouped={"donor_only":set(),"acceptor_only":set(),"donor_and_acceptor":set(),"neither":set()}
        for atom in mol.GetAtoms():
            idx=atom.GetIdx()
            if atom.GetAtomicNum() not in (7,8,16) and idx not in donors|acceptors:continue
            atoms={idx}|{a.GetIdx() for a in atom.GetNeighbors()}
            for n in list(atoms):atoms.update(a.GetIdx() for a in mol.GetAtomWithIdx(n).GetNeighbors())
            fragment=Chem.MolFragmentToSmiles(mol,sorted(atoms),rootedAtAtom=idx,canonical=False)
            kind="donor_and_acceptor" if idx in donors and idx in acceptors else "donor_only" if idx in donors else "acceptor_only" if idx in acceptors else "neither"
            grouped[kind].add(fragment)
        roles={k:sorted(v) for k,v in grouped.items() if v}
    else:
        roles={"unresolved":"Site assignments are omitted because the SMARTS and descriptor conventions disagreed; avoid exact site assignments."}
    ring_atoms=set().union(*(set(r) for r in rings)) if rings else set()
    outside=set(range(mol.GetNumAtoms()))-ring_atoms
    connections=[]
    group_number=0
    while outside and rings:
        seed=min(outside);component={seed};stack=[seed];outside.remove(seed)
        while stack:
            for neighbor in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                n=neighbor.GetIdx()
                if n in outside:outside.remove(n);component.add(n);stack.append(n)
        attachments={a.GetIdx() for i in component for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() in ring_atoms}
        group_number+=1
        environments=[]
        for idx in sorted(attachments):
            labels="/".join(chr(65+j) for j,r in enumerate(rings) if idx in r)
            fragment=Chem.MolFragmentToSmiles(mol,sorted(component|{idx}),rootedAtAtom=idx,canonical=False)
            environments.append(f"from ring {labels}: {fragment}")
        connections.append("same external group; "+"; ".join(environments))
    direct=[]
    for bond in mol.GetBonds():
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        left={i for i,r in enumerate(rings) if a in r};right={i for i,r in enumerate(rings) if b in r}
        if left and right and not left&right:
            direct.append(f"rings {'/'.join(chr(65+i) for i in sorted(left))}-{'/'.join(chr(65+i) for i in sorted(right))}: direct {str(bond.GetBondType()).lower()} bond")
    return {"rings":ring_info,"ring_intersections":shared,"external_groups":connections,
            "direct_ring_connections":direct,"hbond_roles":roles}


def calculate_features(mol):
    values = [
        Descriptors.MolWt(mol), Descriptors.ExactMolWt(mol), Crippen.MolLogP(mol),
        R.CalcTPSA(mol, includeSandP=False), R.CalcNumHBD(mol), R.CalcNumHBA(mol),
        R.CalcNumRotatableBonds(mol, R.NumRotatableBondsOptions.Strict), R.CalcNumRings(mol),
        R.CalcNumAromaticRings(mol), R.CalcNumAliphaticRings(mol), R.CalcNumSaturatedRings(mol),
        R.CalcNumHeterocycles(mol), R.CalcNumAromaticHeterocycles(mol), R.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(), R.CalcNumHeteroatoms(mol), Chem.GetFormalCharge(mol),
        sum(a.GetFormalCharge() > 0 for a in mol.GetAtoms()),
        sum(a.GetFormalCharge() < 0 for a in mol.GetAtoms()), Crippen.MolMR(mol),
        R.CalcNumAmideBonds(mol), R.CalcNumSpiroAtoms(mol), R.CalcNumBridgeheadAtoms(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    ]
    result = dict(zip(FEATURE_NAMES, values, strict=True))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite molecular descriptor")
    return result


def qualitative_context(mol, features=None):
    f = calculate_features(mol) if features is None else features
    logp = f["crippen_logp"]
    lipophilicity = ("water-favoring partition tendency" if logp < 0 else
                    "modest octanol-favoring partition tendency" if logp < 2 else
                    "moderate octanol-favoring partition tendency" if logp < 4 else
                    "strong octanol-favoring partition tendency" if logp < 6 else
                    "very strong octanol-favoring partition tendency")
    polarity = ("limited" if f["tpsa"] < 40 else "moderate" if f["tpsa"] < 90 else
                "substantial" if f["tpsa"] < 140 else "extensive")
    flexibility = ("few or no freely rotatable bonds" if f["rotatable_bonds"] <= 3 else
                   "some rotatable connections" if f["rotatable_bonds"] <= 8 else
                   "many rotatable connections")
    pos, neg = f["positive_charge_centers"], f["negative_charge_centers"]
    charge = ("net positive formal charge" if f["formal_charge"] > 0 else
              "net negative formal charge" if f["formal_charge"] < 0 else
              "net neutral with separated formal charges" if pos or neg else
              "no formal charges in the supplied structure")
    return {
        "detected_motifs": [label for label, name in FRAGMENT_FUNCTIONS.items() if getattr(Fragments, name)(mol)]
                           + (["ether"] if mol.HasSubstructMatch(ETHER_PATTERN) else []),
        "motif_caution": "Substructure matches may overlap and are not exhaustive; verify connectivity from SMILES.",
        "elements_present": sorted({a.GetSymbol() for a in mol.GetAtoms()}),
        "formal_charge_state": charge,
        "hydrogen_bond_donation": "present" if f["hbd"] else "absent",
        "hydrogen_bond_acceptance": "present" if f["hba"] else "absent",
        "calculated_partition_tendency": lipophilicity,
        "calculated_polar_surface": polarity,
        "rotatable_connections": flexibility,
        "interpretation_limit": "These are descriptive bins for the supplied structure, not measured solubility, permeability, pKa or activity.",
        "graph_facts": graph_context(mol,f),
    }


def schema():
    return {"schema_version": "rdkit_properties_v1", "rdkit_version": rdBase.rdkitVersion,
            "vector_length": len(FEATURES), "features": [
                {"index": i, "name": n, "unit": u, "implementation": impl}
                for i, (n, u, impl) in enumerate(FEATURES)],
            "representation": "Unscaled descriptors computed from the exact source SMILES; no neutralization, tautomerization or conformer generation.",
            "tpsa_definition": "RDKit default N/O TPSA; includeSandP=False.",
            "qualitative_bins": {"crippen_logp_boundaries": [0, 2, 4, 6], "tpsa_boundaries": [40, 90, 140],
                "rotatable_bonds_upper_bounds": [3, 8], "status": "Project heuristics, not clinical or universal thresholds"}}

"""Plain structural facts computed from the graph, for constrained LLM narration."""
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Lipinski
from molecular_features import FRAGMENT_FUNCTIONS, ETHER_PATTERN
from rdkit.Chem import Fragments

WORDS={1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten",11:"eleven",12:"twelve"}
ELEMENTS={6:"carbon",7:"nitrogen",8:"oxygen",9:"fluorine",15:"phosphorus",16:"sulfur",17:"chlorine",35:"bromine",53:"iodine"}


def atom_label(mol,idx):
    atom=mol.GetAtomWithIdx(idx)
    if atom.GetAtomicNum()==6 and any(b.GetBondType()==Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetAtomicNum()==8 for b in atom.GetBonds()):
        return "carbonyl carbon"
    return ELEMENTS.get(atom.GetAtomicNum(),atom.GetSymbol())


def structural_facts(mol,features):
    rings=list(mol.GetRingInfo().AtomRings());phrases=[];edges=[]
    for ring in rings:
        ring_edges={tuple(sorted((ring[i],ring[(i+1)%len(ring)]))) for i in range(len(ring))}
        edges.append(ring_edges)
        aromatic=all(mol.GetBondBetweenAtoms(*e).GetIsAromatic() for e in ring_edges)
        saturated=all(mol.GetBondBetweenAtoms(*e).GetBondType()==Chem.BondType.SINGLE for e in ring_edges)
        counts=Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring)
        if len(ring)==6 and counts=={6:6} and aromatic:
            phrase="benzene ring"
        else:
            state="aromatic" if aromatic else "saturated" if saturated else "nonaromatic unsaturated"
            size=WORDS.get(len(ring),"large")
            hetero=[f"{WORDS.get(n,'multiple')} {ELEMENTS.get(e,Chem.GetPeriodicTable().GetElementName(e))} atom{'s' if n>1 else ''}" for e,n in counts.items() if e!=6]
            phrase=f"{size}-membered {state} ring containing "+(", ".join(hetero) if hetero else "carbon ring atoms only")
        phrases.append(phrase)
    facts=[]
    if not rings:facts.append("The molecular framework is acyclic.")
    for phrase in dict.fromkeys(phrases):facts.append(f"The structure contains a {phrase}.")
    for i in range(len(rings)):
        for j in range(i+1,len(rings)):
            overlap=set(rings[i])&set(rings[j]);shared=edges[i]&edges[j]
            if len(overlap)==2 and len(shared)==1:
                facts.append(f"A {phrases[i]} and a {phrases[j]} share a fused bond.")
    if features["spiro_atoms"]:facts.append("The ring framework includes spiro connections.")
    if features["bridgehead_atoms"]:facts.append("The ring framework includes bridged connections; do not describe all of these as simple ring fusions.")
    ring_atoms=set().union(*(set(r) for r in rings)) if rings else set()
    def endpoint(idx):
        atom=mol.GetAtomWithIdx(idx)
        ring_names=list(dict.fromkeys(phrases[j] for j,r in enumerate(rings) if idx in r))
        return f"a {atom_label(mol,idx)} atom belonging to the "+" and the ".join(ring_names)
    outside=set(range(mol.GetNumAtoms()))-ring_atoms
    while outside and rings:
        seed=min(outside);outside.remove(seed);component={seed};stack=[seed]
        while stack:
            for n in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                idx=n.GetIdx()
                if idx in outside:outside.remove(idx);component.add(idx);stack.append(idx)
        attachments=sorted({n.GetIdx() for i in component for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ring_atoms})
        if len(attachments)==1:
            root=attachments[0]
            attached=next(i for i in component if mol.GetBondBetweenAtoms(root,i))
            heavy=[i for i in component if mol.GetAtomWithIdx(i).GetAtomicNum()!=1]
            elements=[mol.GetAtomWithIdx(i).GetAtomicNum() for i in heavy]
            bond=mol.GetBondBetweenAtoms(root,attached)
            if len(heavy)==1 and elements[0] in (9,17,35,53):
                group=f"a {ELEMENTS[elements[0]]} substituent"
            elif len(heavy)==1 and elements[0]==8 and bond.GetBondType()==Chem.BondType.DOUBLE:
                facts.append(f"An exocyclic oxygen is double-bonded to {endpoint(root)}.");continue
            elif elements and all(e==6 for e in elements) and all(b.GetBondType()==Chem.BondType.SINGLE for i in heavy for b in mol.GetAtomWithIdx(i).GetBonds() if b.GetOtherAtomIdx(i) in component):
                group="a methyl substituent" if len(heavy)==1 else "an acyclic saturated carbon substituent"
            elif sorted(elements)==[6,8] and mol.GetAtomWithIdx(attached).GetAtomicNum()==8:
                group="a methoxy substituent through its oxygen"
            else:
                group=f"an external acyclic group through {atom_label(mol,attached)}"
            facts.append(f"There is {group} attached to {endpoint(root)}.")
        elif len(attachments)==2:
            start,end=attachments;path=Chem.GetShortestPath(mol,start,end)
            if not set(path[1:-1])<=component:
                facts.append("An external acyclic group links ring units.");continue
            sequence=", followed by ".join(atom_label(mol,i) for i in path[1:-1])
            bonds=[mol.GetBondBetweenAtoms(path[j],path[j+1]).GetBondType() for j in range(len(path)-1)]
            if all(b==Chem.BondType.SINGLE for b in bonds):bond_text="These connecting bonds are single bonds."
            else:bond_text="The connecting path includes unsaturation."
            facts.append(f"From {endpoint(start)} to {endpoint(end)}, the ordered linker atoms are {sequence}. {bond_text}")
        elif attachments:
            facts.append("A shared branching acyclic group links multiple ring units.")
    for bond in mol.GetBonds():
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        left={j for j,r in enumerate(rings) if a in r};right={j for j,r in enumerate(rings) if b in r}
        if left and right and not left&right:
            facts.append(f"There is a direct {str(bond.GetBondType()).lower()} bond between {endpoint(a)} and {endpoint(b)}.")
    # RDKit's broad sulfide fragment also matches acyl-sulfur connectivity; avoid
    # turning that substructure count into an unsupported thioether classification.
    groups=[name for name,fn in FRAGMENT_FUNCTIONS.items() if not name.endswith("ring") and name!="thioether" and getattr(Fragments,fn)(mol)]
    if mol.HasSubstructMatch(ETHER_PATTERN):groups.append("ether")
    if groups:facts.append("Detected functional motifs include "+", ".join(groups)+". These motif matches can overlap.")
    donor_types=set();acceptor_types=set()
    donors={m[0] for m in mol.GetSubstructMatches(Lipinski.HDonorSmarts)}
    acceptors={m[0] for m in mol.GetSubstructMatches(Lipinski.HAcceptorSmarts)}
    for idx in donors:
        a=mol.GetAtomWithIdx(idx)
        donor_types.add("aromatic nitrogen bearing hydrogen" if a.GetIsAromatic() else f"{atom_label(mol,idx)} bearing hydrogen")
    for idx in acceptors:
        a=mol.GetAtomWithIdx(idx)
        # Exclude pyrrole-like substituted aromatic nitrogen from prose attribution,
        # while leaving the RDKit descriptor vector unchanged and versioned.
        if a.GetAtomicNum()==7 and a.GetIsAromatic():
            if a.GetDegree()==2 and a.GetTotalNumHs()==0:acceptor_types.add("aromatic ring nitrogen without a bonded hydrogen")
            continue
        if a.GetAtomicNum()==8:
            carbonyl=any(b.GetBondType()==Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetAtomicNum()==6 for b in a.GetBonds())
            acceptor_types.add("carbonyl oxygen" if carbonyl else "noncarbonyl oxygen in an acceptor-capable environment")
        elif a.GetAtomicNum()==7:
            acceptor_types.add("nonaromatic nitrogen in an acceptor-capable environment")
        elif a.GetAtomicNum()==16:
            acceptor_types.add("sulfur in an acceptor-capable environment")
    if donor_types:facts.append("Hydrogen-bond donation is supported by "+", ".join(sorted(donor_types))+". A neutral nitrogen bearing hydrogen is not necessarily protonated.")
    if acceptor_types:facts.append("Conservative hydrogen-bond acceptor assignments include "+", ".join(sorted(acceptor_types))+"; do not generalize this role to every atom of the same element.")
    if any(a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):facts.append("The supplied structure specifies atom stereochemistry; no new absolute assignment is inferred here.")
    return list(dict.fromkeys(facts))

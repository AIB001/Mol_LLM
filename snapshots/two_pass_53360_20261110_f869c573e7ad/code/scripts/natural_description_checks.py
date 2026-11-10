"""Offline screens for free natural molecular prose, never chemistry proof.

Only deterministic format/policy failures are retry candidates. Structural and
stereochemical heuristics produce review reasons, not a claim of verification.
Ordinary nomenclature locants, ring sizes and local group counts are allowed.
"""
from __future__ import annotations

import re
from annotation_checks_v2 import _property_number_errors


_ACTIVITY_ADMET = re.compile(
    r"\b(?:inhibit(?:s|or|ors|ory|ion|ing)?|agonis(?:t|ts|m)|antagonis(?:t|ts|m)|"
    r"(?:anti[ -]?)?(?:cancer|tumou?r|viral|bacterial|microbial|inflammatory)|"
    r"therapeutic(?:ally)?|pharmacolog(?:ical|ically|y)|bioactiv(?:e|ity)|"
    r"drug[ -]?lik(?:e|eness)|poten(?:t|cy)|target\s+(?:binding|selectivity)|"
    r"bind(?:s|ing)?\s+(?:to\s+)?(?:the\s+)?(?:receptor|enzyme|protein)|"
    r"(?:receptor|enzyme|kinase)\s+(?:binding|activity|modulation)|"
    r"ADMET|bioavailab(?:le|ility)|permeab(?:le|ility)|blood[ -]brain|BBB|"
    r"(?:water|aqueous)[ -]solub(?:le|ility)|solubility|"
    r"(?:metabolic(?:ally)?|plasma)\s+(?:stable|stability)|"
    r"(?:hepatic|renal)\s+(?:clearance|elimination)|"
    r"(?:oral|intestinal)\s+(?:absorption|exposure)|"
    r"(?:non[ -]?)?toxic(?:ity)?|cytotoxic(?:ity)?|carcinogen(?:ic|icity)?|"
    r"(?:safe|safety|efficacy|effective)\s+(?:drug|medicine|treatment)|"
    r"pharmacokinetics?|pharmacodynamics?)\b", re.I)
_IDENTIFIER = re.compile(r"\bCHEMBL\d+\b", re.I)
_CONTROLLED_LABEL = re.compile(r"\b(?:[A-Z]\d+|G\d+\.\d+)\s*=|\b(?:RDKit|atom[- ]map|SMILES|InChI)\b")
_R_S = re.compile(r"\((?:\d+\s*)?[RS](?:\s*,\s*\d+[RS])*\)|\b(?:R|S)[ -](?:configuration|configured|stereochemistry|enantiomer)\b|\b\d+[RS](?=[,)-])")
_E_Z = re.compile(r"\((?:\d+)?([EZ])\)|\b([EZ])[ -](?:configured|configuration|geometry|alkene|double\s+bond|isomer)\b|\b(?:geometry|configuration)\s+(?:is\s+)?([EZ])\b")
_UNSUPPORTED_STEREO_STATE = re.compile(r"\b(?:achiral|racem(?:ic|ate)|enantiopure|enantiomerically\s+pure|optically\s+(?:active|inactive))\b", re.I)
_CONFORMATION = re.compile(r"\b(?:fully\s+planar|entirely\s+planar|coplanar|molecule\s+is\s+planar|adopts?\s+(?:a\s+)?(?:chair|boat)|fixed\s+(?:chair|boat))\b", re.I)
_ROLE_ACCEPTOR = re.compile(r"(?:amide\s+(?:N|nitrogen)|pyrrolic\s+(?:N|nitrogen)|\[nH\])[^.;]{0,65}\bacceptor\b", re.I)
_AMIDE_ROLE_ACCEPTOR = re.compile(r"\bacceptor\b[^.;]{0,65}(?:amide\s+(?:N|nitrogen)|pyrrolic\s+(?:N|nitrogen)|\[nH\])", re.I)

# A small conservative dictionary is used only for positive-claim mismatch
# triage. Matching a motif never validates its locants, role or prose context.
_MOTIFS = {
    "nitrile": (r"\b(?:nitrile|cyano)[a-z-]*\b", "[C]#[N]"),
    "alkyne": (r"\b(?:alkyne|alkynyl|triple[ -]bond)[a-z-]*\b", "[C]#[C]"),
    "alkene": (r"\b(?:alkene|alkenyl|vinyl|ethenyl)[a-z-]*\b", "[C]=[C]"),
    "amide": (r"\b(?:amide|acetamide|acetanilide)\b", "[C](=[O])[N]"),
    "ester": (r"\b(?:ester|acetate|acetoxy)\b", "[C](=[O])[O][#6]"),
    "phenol": (r"\b(?:phenols?|phenolic\s+(?:OH|hydroxyl)|free\s+phenolic)\b", "[c][O;H1]"),
    "carboxylic_acid": (r"\bcarboxylic\s+acid\b", "[C](=[O])[O;H1]"),
    "methoxy": (r"\b(?:di|tri|tetra)?methoxy[a-z-]*\b", "[O][CH3]"),
    "nitro": (r"\bnitro(?!gen|so|x)[a-z-]*\b", "[N+](=[O])[O-]"),
    "sulfonamide": (r"\bsul(?:f|ph)onamide[a-z-]*\b", "[S](=[O])(=[O])[N]"),
    "morpholine": (r"\bmorpholin[a-z-]*\b", "[O]1[C][C][N][C][C]1"),
    "piperazine": (r"\bpiperazin[a-z-]*\b", "[N]1[C][C][N][C][C]1"),
    "piperidine": (r"\bpiperidin[a-z-]*\b", "[N]1[C][C][C][C][C]1"),
    "pyridine": (r"\bpyridin[a-z-]*\b", "n1ccccc1"),
    "pyrimidine": (r"\bpyrimidin[a-z-]*\b", "n1cnccc1"),
    "pyridazine": (r"\bpyridazin[a-z-]*\b", "n1ncccc1"),
    "pyrazine": (r"\bpyrazin[a-z-]*\b", "n1ccncc1"),
    "pyrazole": (r"\bpyrazol(?:e|ic|yl|o)[a-z0-9,\[\].-]*\b", "[n]1[n][c][c][c]1"),
    "pyrrole": (r"\bpyrrol(?:e|ic|yl|o)[a-z0-9,\[\].-]*\b", "[n]1[c][c][c][c]1"),
    "cinnoline": (r"\bcinnolin[a-z-]*\b", "c1ccc2nnccc2c1"),
    "phthalazine": (r"\bphthalazin[a-z-]*\b", "c1ccc2cnncc2c1"),
    "quinazoline": (r"\bquinazolin[a-z-]*\b", "c1ccc2ncncc2c1"),
}
_COMPILED_MOTIFS = None


def _positive_matches(pattern, text):
    """Simple nearby negation filter; remaining matches still need human review."""
    matches = pattern.finditer(text) if hasattr(pattern, "finditer") else re.finditer(pattern, text, re.I)
    for match in matches:
        prefix = re.split(r"[.;]", text[max(0, match.start() - 55):match.start()])[-1]
        if re.search(r"\b(?:no|not|neither|without|lacks?|absent|rather\s+than)\b", prefix + " " + match.group(), re.I):
            continue
        yield match


def _source_molecule(facts):
    from rdkit import Chem
    smiles = facts.get("original_smiles") or facts.get("canonical_isomeric_smiles")
    return Chem.MolFromSmiles(smiles) if smiles else None


def _structural_review(description, facts):
    from rdkit import Chem
    global _COMPILED_MOTIFS
    reasons = []
    if _COMPILED_MOTIFS is None:
        _COMPILED_MOTIFS = {name: (pattern, Chem.MolFromSmarts(smarts)) for name, (pattern, smarts) in _MOTIFS.items()}
        if any(query is None for _, query in _COMPILED_MOTIFS.values()):
            raise RuntimeError("Invalid offline motif query")
    mol = _source_molecule(facts)
    if mol is None:
        return ["source_structure_unavailable_for_motif_screen"]
    for name, (pattern, query) in _COMPILED_MOTIFS.items():
        if any(_positive_matches(pattern, description)) and not mol.HasSubstructMatch(query):
            reasons.append("claimed_motif_not_found:" + name)
    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    for symbol, name in (("F", "fluor"), ("Cl", "chlor"), ("Br", "brom"), ("I", "iod")):
        if symbol not in elements and any(_positive_matches(r"\b(?:di|tri|tetra|penta|hexa)?" + name + r"[a-z-]*\b", description)):
            reasons.append("claimed_element_not_found:" + symbol)

    atoms, bonds = facts.get("atoms", []), facts.get("bonds", [])
    atom_assignments = [a for a in atoms if a.get("chiral_tag", "CHI_UNSPECIFIED") != "CHI_UNSPECIFIED"]
    if _R_S.search(description) and not atom_assignments:
        reasons.append("absolute_stereo_claim_without_source_assignment")
    source_geometry = {b.get("stereo") for b in bonds if b.get("stereo") in ("STEREOE", "STEREOZ")}
    claimed_geometry = {next(group for group in m.groups() if group) for m in _E_Z.finditer(description)}
    for geometry in claimed_geometry:
        if "STEREO" + geometry not in source_geometry:
            reasons.append("bond_geometry_claim_not_present_in_source:" + geometry)
    for geometry in source_geometry:
        letter = geometry[-1]
        if letter not in claimed_geometry:
            reasons.append("specified_bond_geometry_not_explicitly_retained:" + letter)
    if any(_positive_matches(_UNSUPPORTED_STEREO_STATE, description)):
        reasons.append("unsupported_stereochemical_state_claim")
    if re.search(r"\b(?:cis|trans)\b", description, re.I):
        # Cis/trans in a name may refer to ring relationships rather than the
        # source double bond; the free prose does not identify atoms reliably.
        reasons.append("cis_trans_assignment_requires_atom_specific_review")
    if atom_assignments and not (_R_S.search(description) or re.search(r"stereo|configur|chiral", description, re.I)):
        reasons.append("specified_atom_stereo_not_explicitly_discussed")
    unspecified = any(p.get("specified") != "Specified" for p in facts.get("stereo", {}).get("potential_centers", []))
    if unspecified and re.search(r"\b(?:all|fully|completely)\s+(?:stereo\w*\s+)?(?:specified|defined|assigned)\b", description, re.I):
        reasons.append("claims_complete_stereo_with_unspecified_source_sites")
    for pattern in (_ROLE_ACCEPTOR, _AMIDE_ROLE_ACCEPTOR):
        if any(_positive_matches(pattern, description)):
            reasons.append("amide_or_pyrrolic_n_acceptor_claim_requires_review")
            break
    if _CONFORMATION.search(description):
        reasons.append("specific_conformation_not_encoded_by_source_graph")
    return reasons


def check_description(description, facts=None):
    """Return schema/number/policy screens plus explicitly unverified chemistry.

    The word target is 100–150; 95–160 is accepted as the user's approximate
    boundary. Near-target counts are editorial hints, not chemistry warnings.
    """
    errors, review, hints = [], [], []
    if not isinstance(description, str) or not description.strip():
        errors.append("missing_description")
        description = "" if not isinstance(description, str) else description
    count = len(description.split())
    in_target = 100 <= count <= 150
    if description:
        if not 95 <= count <= 160:
            errors.append("word_count_outside_95_160")
        elif not in_target:
            hints.append("word_count_near_target")
        if "\n" in description or "\r" in description:
            errors.append("description_not_single_paragraph")
        if "```" in description or re.match(r"\s*(?:#{1,6}\s|[-*]\s|\d+[.)]\s)", description):
            errors.append("markdown_or_list_in_description")
        if _IDENTIFIER.search(description):
            errors.append("identifier_in_description")
        if _CONTROLLED_LABEL.search(description):
            errors.append("implementation_notation_in_description")
        errors.extend(_property_number_errors(description))
        if _ACTIVITY_ADMET.search(description):
            errors.append("activity_or_admet_claim_or_discussion")
        if facts is not None:
            review.extend(_structural_review(description, facts))
        elif _R_S.search(description) or _E_Z.search(description) or _UNSUPPORTED_STEREO_STATE.search(description):
            review.append("stereo_claim_without_source_evidence")
    errors, review, hints = sorted(set(errors)), sorted(set(review)), sorted(set(hints))
    return {"format_errors": errors, "review_reasons": review, "editorial_hints": hints,
            "word_count": count, "word_count_in_target": in_target,
            "automated_checks_passed": not errors, "chemistry_verified": False,
            "chemical_review_status": "needs_review" if review else "not_independently_reviewed",
            "validation_scope": "Format/policy screens and conservative source-motif/stereo triage only; no proof of chemical correctness, locants, connectivity coverage, or unique interpretation."}


def validate_response(result, expected_id, facts=None):
    """Single-record adapter; opaque batch keys remain the runner's contract."""
    if not isinstance(result, dict):
        checked = check_description(None, facts)
        checked["format_errors"].append("response_is_not_object")
    else:
        checked = check_description(result.get("description"), facts)
        if set(result) != {"chembl_id", "description"}:
            checked["format_errors"].append("unexpected_json_fields")
        if result.get("chembl_id") != expected_id:
            checked["format_errors"].append("mismatched_chembl_id")
    checked["format_errors"] = sorted(set(checked["format_errors"]))
    checked["automated_checks_passed"] = not checked["format_errors"]
    return checked

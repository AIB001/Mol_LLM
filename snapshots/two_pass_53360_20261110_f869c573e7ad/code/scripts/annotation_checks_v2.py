"""V2 schema and numeric-property checks, not a chemistry correctness proof.

Structural numerals (locants, ring sizes, multiplicities, isotope/stereo labels,
and atom charges) are permitted. Property numerals are screened using their
grammatical context rather than a blanket digit ban. Free-form language cannot
be exhaustively classified by regular expressions: independent structure
reconstruction and targeted chemical review remain necessary.
"""
from __future__ import annotations

import re
import unicodedata


NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion|dozen|single|half|quarter|third|fourth|fifth)"
)
NUMBER = (
    r"(?<![\w])(?:[+−-]?(?:\d+(?:[.,]\d+)*|\.\d+)(?:e[+−-]?\d+)?"
    r"(?:\s*/\s*\d+)?|(?:(?:minus|plus|negative|positive)\s+)?" + NUMBER_WORD + r")(?![\w])"
)
# Actual descriptor names, rather than generic ring/branch/atom counts. Local
# structure counts and even local charge magnitudes must remain legal.
VALUE_PROPERTY = (
    r"(?:molecular\s+(?:weight|mass)|exact\s+(?:molecular\s+)?mass|"
    r"(?:average\s+)?mol(?:ecular)?[ _]?(?:wt|weight)|m\.?w\.?|"
    r"(?:crippen[ _-]?)?(?:c\s*)?log[ _-]?[pd]|"
    r"tpsa|(?:topological\s+)?polar\s+surface\s+area|"
    r"(?:molar\s+)?refractivity|(?:aqueous\s+)?solubility|"
    r"lipophilicity|(?:octanol[ /-]water\s+)?partition\s+coefficient|"
    r"(?:molecular\s+)?volume|density|(?:melting|boiling)\s+point|"
    r"permeability|bioavailability|binding\s+affinity|half[ -]life|"
    r"pka|pki|pkd|pic50|pec50|ic50|ec50|ki|kd|"
    r"fraction[ _-]?(?:of[ _-]?)?(?:csp3|sp3(?:[ _-]?(?:carbon|carbons))?)|"
    r"f[ _-]?csp3|formal[ _-]?charge[ _-]?(?:total|sum)|"
    r"(?:ring|aromatic[ _-]rings?|aliphatic[ _-]rings?|saturated[ _-]rings?|"
    r"heterocycle|heavy[ _-]atom|heteroatom|amide[ _-]bond|spiro[ _-]atom|"
    r"bridgehead[ _-]atom|chiral[ _-]center|positive[ _-]charge[ _-]center|"
    r"negative[ _-]charge[ _-]center)[ _-]?(?:count|total)s?)"
)
COUNT_PROPERTY = (
    r"(?:(?:hydrogen[ -]bond(?:ing)?\s+|h[ -]bond\s+)?"
    r"(?:donors?|acceptors?)(?:\s+(?:sites?|groups?|centers?|atoms?))?|"
    r"hba|hbd|(?:freely\s+)?rotatable[ _-]bonds?|heavy[ _-]atoms?)"
)
PROPERTY_SUFFIX = r"(?:\s+(?:value|count|number|total))?"
MODIFIERS = r"(?:(?:about|approximately|roughly|around|nearly|exactly|only|just|estimated|calculated|predicted|at\s+least|at\s+most|up\s+to|more\s+than|less\s+than|greater\s+than|lower\s+than|between|from|equal\s+to|a\s+total\s+of|a)\s+)*"
CONNECTOR = (
    r"\s*(?:(?:is|are|was|were|equals?|totals?|numbers?|numbering|reaches?|"
    r"of|at|to|has|have|being|is\s+reported\s+as|is\s+calculated\s+as|"
    r"is\s+estimated\s+at|is\s+predicted\s+to\s+be)\s+)?"
    r"(?:[:=~≈<>≤≥(]\s*)?"
)
_FLAGS = re.IGNORECASE
_VALUE_AFTER = re.compile(
    r"\b" + VALUE_PROPERTY + PROPERTY_SUFFIX + CONNECTOR + MODIFIERS + NUMBER, _FLAGS
)
_COUNT_AFTER = re.compile(
    r"\b" + COUNT_PROPERTY + PROPERTY_SUFFIX + CONNECTOR + MODIFIERS + NUMBER, _FLAGS
)
_COUNT_BEFORE = re.compile(
    NUMBER + r"\s+(?:(?:potential|possible|strong|weak|available|distinct|"
    r"neutral|charged|amide|amine|nitrogen|oxygen|NH|NH2|OH)\s+){0,3}"
    + COUNT_PROPERTY + r"\b", _FLAGS
)
_VALUE_BEFORE = re.compile(
    NUMBER + r"\s+(?:(?:is|was|for|as|in|of)\s+)?(?:(?:the|its|a)\s+)?"
    + VALUE_PROPERTY + r"\b", _FLAGS
)
_TOTAL_COUNT = re.compile(
    r"\b(?:total\s+(?:number\s+of\s+)?(?:heavy\s+atoms|heteroatoms)|"
    r"number\s+of\s+(?:hydrogen[ -]bond\s+)?(?:donors?|acceptors?|"
    r"rotatable\s+bonds|heavy\s+atoms|heteroatoms))"
    + CONNECTOR + MODIFIERS + NUMBER, _FLAGS
)
_COUNTED_ROLE = re.compile(
    NUMBER + r"\s+(?:nitrogen|oxygen|sulfur|heteroatom)(?:\s+atoms?)?"
    r"\s+(?:act|serve|function)\s+as\s+(?:(?:hydrogen[ -]bond|h[ -]bond)\s+)?"
    r"(?:donors?|acceptors?)\b", _FLAGS
)
_PROPERTY_UNIT = re.compile(
    r"(?:\b(?:g\s*/\s*mol|g\s+mol\s*[-−]?1|daltons?|kDa|Da|"
    r"cm\s*(?:\^?3|³)\s*/\s*mol|nanomolar|micromolar|millimolar|"
    r"nM|uM|μM|µM|mM|angstroms?|percent|per\s+cent)\b|Å|%)", _FLAGS
)
# Mask only clearly structural notation. Do not mask bare numbers or rely on
# proximity windows, which confuse a property adjective with the next ring size.
_STRUCTURAL_NOTATION = re.compile(
    r"(?<!\w)(?:\d+[RSEZ](?:\s*,\s*\d+[RSEZ])*|"
    r"\d+(?:\s*,\s*\d+)+(?:[′']+)?(?=[-‐‑–][A-Za-z])|"
    r"\d+(?=[-‐‑–]membered\b)|\d+(?=[-‐‑–]carbon\b))"
)


def _property_number_errors(description: str) -> list[str]:
    text = unicodedata.normalize("NFKC", description)
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-")
    text = re.sub(r"\((hba|hbd|tpsa|mw|log[ _-]?[pd])\)", r"\1", text, flags=_FLAGS)
    # Hyphenated number words are ordinary number expressions, not locants.
    text = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", text)
    text = _STRUCTURAL_NOTATION.sub("STRUCTURAL_LABEL", text)
    errors = []
    if any(pattern.search(text) for pattern in (_VALUE_AFTER, _VALUE_BEFORE)):
        errors.append("numerical_property_value")
    if any(pattern.search(text) for pattern in (
        _COUNT_AFTER, _COUNT_BEFORE, _TOTAL_COUNT, _COUNTED_ROLE
    )):
        errors.append("numerical_property_count")
    if _PROPERTY_UNIT.search(text):
        errors.append("property_unit_or_percentage")
    return errors


def validate_annotation_v2(result, expected_id, facts=None):
    """Return stable failure codes for one generated annotation.

    ``facts`` is accepted for callers that retain the complete structural graph.
    It is deliberately not treated as proof that the prose covers that graph.
    Successful validation means only schema/numeric-property checks passed.
    Length is adaptive: this function does not require filler or truncate a
    complex molecule to an arbitrary word count.
    """
    errors = []
    if not isinstance(result, dict):
        return ["response_is_not_object"]
    if set(result) != {"chembl_id", "description", "status", "issues"}:
        errors.append("unexpected_json_fields")
    if result.get("chembl_id") != expected_id:
        errors.append("mismatched_chembl_id")
    description = result.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("missing_description")
        description = ""
    if description:
        errors.extend(_property_number_errors(description))
        if "\n" in description or "\r" in description or "```" in description:
            errors.append("description_not_single_paragraph")
        if re.search(r"\bCHEMBL\d+\b", description, re.IGNORECASE):
            errors.append("identifier_in_description")
    status = result.get("status")
    if status not in ("ok", "needs_review"):
        errors.append("invalid_status")
    issues = result.get("issues")
    if not isinstance(issues, list) or not all(isinstance(x, str) and x.strip() for x in issues):
        errors.append("invalid_issues")
    elif status == "ok" and issues:
        errors.append("ok_with_issues")
    elif status == "needs_review" and not issues:
        errors.append("review_without_issue")
    return sorted(set(errors))

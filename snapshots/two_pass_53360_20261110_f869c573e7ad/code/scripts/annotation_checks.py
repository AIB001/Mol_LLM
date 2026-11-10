"""Conservative format/numerical-property checks; not a chemistry correctness proof."""

import re

NUMBER_WORDS = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|dozen|hundred|thousand|million|single)"

COUNTED_PROPERTIES = r"(?:(?:hydrogen[ -]bond(?:ing)?\s+)?(?:donors?|acceptors?)(?:\s+(?:sites?|groups?|centers?))?|rotatable\s+bonds?|heavy\s+atoms?|heteroatoms?)"

def validate_annotation(result, expected_id):
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
    if re.search(r"\d", description):
        errors.append("digit_in_description")
    if re.search(r"\b" + NUMBER_WORDS + r"(?:\s+[a-z-]+){0,2}\s+" + COUNTED_PROPERTIES + r"\b", description, re.I):
        errors.append("spelled_out_property_count")
    if re.search(r"\b(?:molecular\s+(?:weight|mass)|log\s?p|tpsa|polar\s+surface\s+area)\b.{0,45}\b" + NUMBER_WORDS + r"\b", description, re.I):
        errors.append("spelled_out_property_value")
    if re.search(r"\b(?:donors?|acceptors?|rotatable\s+bonds?)\b.{0,16}\b" + NUMBER_WORDS + r"\b", description, re.I):
        errors.append("spelled_out_property_count")
    if re.search(r"\b(?:plus|minus|positive|negative)\s+" + NUMBER_WORDS + r"\b|\b(?:single|double|triple)\s+(?:positive|negative|formal)\s+charge\b", description, re.I):
        errors.append("numerical_charge_magnitude")
    if re.search(r"\b(?:g/mol|daltons?|nanomolar|micromolar|angstroms?|pIC|pKi|pKd|IC50|EC50)\b|Å|%", description, re.I):
        errors.append("property_unit_or_activity_value")
    if result.get("status") not in {"ok", "needs_review"}:
        errors.append("invalid_status")
    issues = result.get("issues")
    if not isinstance(issues, list) or not all(isinstance(x, str) for x in issues):
        errors.append("invalid_issues")
    elif result.get("status") == "ok" and issues:
        errors.append("ok_with_issues")
    elif result.get("status") == "needs_review" and not issues:
        errors.append("review_without_issue")
    if description and not 50 <= len(description.split()) <= 240:
        errors.append("description_length")
    if "\n" in description or "```" in description:
        errors.append("description_not_single_paragraph")
    return sorted(set(errors))

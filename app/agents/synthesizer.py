"""Synthesizer: rolls up per-topic evaluator results into a final review."""
import json
from typing import Any, Dict, List
import anthropic
from . import HAIKU_MODEL

REASONING_SYSTEM = """Write 2-3 sentences explaining an HOA ARC decision based on the provided per-topic outcomes. Be specific about which items pass and which need information or violate rules. No preamble, no closing. Plain prose only."""


def _dedupe_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dedupe by (source, section). Keep first occurrence."""
    seen = set()
    out = []
    for c in citations:
        key = (c.get("source", ""), c.get("section", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _rollup_decision(outcomes: List[str]) -> str:
    if "violates" in outcomes:
        return "Denial"
    if "needs_info" in outcomes:
        return "Conditional Approval"
    if outcomes and all(o == "compliant" for o in outcomes):
        return "Approval"
    return "Conditional Approval"


def _coverage_notes(
    parsed: Dict[str, Any],
    evaluator_results: List[Dict[str, Any]],
    retrieval_failures: List[str],
    evaluator_failures: List[str],
) -> List[str]:
    notes: List[str] = []
    for topic in retrieval_failures:
        notes.append(f"No guideline excerpt was retrieved for '{topic}'. Manual review required.")
    for msg in evaluator_failures:
        notes.append(msg)

    results_by_topic = {r.get("topic", ""): r for r in evaluator_results}
    for item in parsed.get("items", []):
        r = results_by_topic.get(item)
        if r is None:
            continue
        if not r.get("citations"):
            notes.append(
                f"No specific rule was cited for requested item '{item}' (evaluator outcome: {r.get('outcome', 'unknown')}). Manual review required."
            )

    for r in evaluator_results:
        if r.get("citations") and all(
            c.get("source") == "Covenants.txt" for c in r["citations"]
        ):
            notes.append(
                f"Only Covenants citations available for '{r.get('topic', '')}'; no Design Guidelines section covers this topic."
            )

    return notes


def _generate_reasoning(
    client: anthropic.Anthropic,
    decision: str,
    evaluator_results: List[Dict[str, Any]],
    failure_notes: List[str],
) -> str:
    summary = json.dumps({
        "decision": decision,
        "topics": [
            {"topic": r.get("topic"), "outcome": r.get("outcome")}
            for r in evaluator_results
        ],
        "failure_notes": failure_notes,
    })
    try:
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=256,
            system=REASONING_SYSTEM,
            messages=[{"role": "user", "content": summary}],
        )
        return response.content[0].text.strip()
    except Exception:
        outcomes = [r.get("outcome", "unknown") for r in evaluator_results]
        return (
            f"Decision: {decision}. Evaluated {len(evaluator_results)} topics — "
            f"{outcomes.count('compliant')} compliant, "
            f"{outcomes.count('needs_info')} need info, "
            f"{outcomes.count('violates')} violations, "
            f"{outcomes.count('not_addressed')} not addressed."
        )


def synthesize(
    client: anthropic.Anthropic,
    parsed: Dict[str, Any],
    evaluator_results: List[Dict[str, Any]],
    retrieval_failures: List[str],
    evaluator_failures: List[str],
    request_text: str,
) -> Dict[str, Any]:
    """Combine per-topic evaluator results into the final review object."""
    item_topics = set(parsed.get("items", []))

    # Order: items first (in request order), then global concerns, then anything else.
    ordered_results = []
    for item in parsed.get("items", []):
        for r in evaluator_results:
            if r.get("topic") == item:
                ordered_results.append(r)
                break
    for concern in parsed.get("global_concerns", []):
        for r in evaluator_results:
            if r.get("topic") == concern:
                ordered_results.append(r)
                break

    flat_citations = [c for r in ordered_results for c in r.get("citations", [])]
    citations = _dedupe_citations(flat_citations)

    outcomes = [r.get("outcome", "not_addressed") for r in ordered_results]
    decision = _rollup_decision(outcomes)

    failure_notes = _coverage_notes(parsed, ordered_results, retrieval_failures, evaluator_failures)
    reasoning = _generate_reasoning(client, decision, ordered_results, failure_notes)

    return {
        "decision": decision,
        "reasoning": reasoning,
        "citations": citations,
        "hitl_required": True,
        "failure_notes": failure_notes,
        "_request": request_text,
        "homeowner_name": parsed.get("homeowner_name"),
        "_parsed": parsed,
        "_evaluator_results": ordered_results,
    }

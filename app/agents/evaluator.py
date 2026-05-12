"""Evaluator: per-topic citation selection and outcome decision. One Sonnet call per topic."""
import json
from typing import Any, Dict, List
import anthropic
from langchain_core.documents import Document
from . import SONNET_MODEL, parse_json_response
from .retriever import format_chunks_for_prompt

EVALUATOR_SYSTEM = """You are evaluating one specific aspect of an HOA architectural request against the provided guideline excerpts.

Your job:
1. Determine if the topic is actually addressed by any excerpt. If not, return citations: [] and outcome: "not_addressed".
2. For each rule that applies to this topic, produce a citation. Be selective — cite the MOST SPECIFIC rule(s), not generic procedural sections ("must obtain ARC approval"). If a section names the improvement type or gives concrete requirements (dimensions, colors, setbacks, materials, neighbor approval steps), cite it. If only a generic procedural section exists, return citations: [] — do not cite it.
3. For each citation, write a one-sentence "ask" describing exactly what the homeowner must provide or confirm to satisfy the rule. Asks must be concrete (a number, a document, a confirmation), never vague ("provide additional details").
4. Decide a per-topic outcome:
   - "compliant" — request as described clearly satisfies the rule.
   - "violates" — request as described clearly violates the rule.
   - "needs_info" — rule applies but the homeowner hasn't provided enough information.
   - "not_addressed" — no specific rule in the provided excerpts covers this topic.

Return JSON only:
{
  "topic": string,
  "outcome": "compliant" | "violates" | "needs_info" | "not_addressed",
  "citations": [
    {
      "source": string,
      "article": string,
      "section": string,
      "quote": string,
      "topic_label": string,
      "ask": string
    }
  ],
  "notes": string | null
}

The "source", "article", and "section" fields must be copied verbatim from the excerpt metadata. The "quote" must be a 1-2 sentence verbatim excerpt from the chunk CONTENT. The "topic_label" is a short CAPS label for the email (e.g. "BASKETBALL COURT", "DRAINAGE", "IMPERVIOUS SURFACE")."""


def evaluate(
    client: anthropic.Anthropic,
    topic: str,
    request_text: str,
    chunks: List[Document],
) -> Dict[str, Any]:
    """Run the evaluator with one retry on JSON parse failure."""
    user_msg = (
        f"TOPIC: {topic}\n\n"
        f"FULL REQUEST (for context only — evaluate only the topic above):\n{request_text}\n\n"
        f"GUIDELINE EXCERPTS:\n{format_chunks_for_prompt(chunks)}"
    )

    last_error: Exception | None = None
    for _ in range(2):
        response = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=2048,
            system=EVALUATOR_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        try:
            result = parse_json_response(response.content[0].text)
            result.setdefault("topic", topic)
            result.setdefault("outcome", "not_addressed")
            result.setdefault("citations", [])
            result.setdefault("notes", None)
            return result
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            last_error = e
    raise RuntimeError(f"Evaluator failed to return valid JSON for topic '{topic}': {last_error}")

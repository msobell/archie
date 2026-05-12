"""Parser agent: decomposes the homeowner's request into items, global concerns, and a name."""
from typing import Any, Dict
import anthropic
from . import HAIKU_MODEL, parse_json_response

PARSER_SYSTEM = """You are parsing an HOA architectural review request. Extract three things:

1. items — distinct physical improvements the homeowner is requesting. Use short noun phrases naming the improvement type (e.g. "basketball court", "concrete walkway", "wood pergola"). One entry per distinct improvement. Do not split a single improvement into sub-parts (a "basketball court" is one item, not "court surface" + "hoop" + "lines").

2. global_concerns — request-wide regulatory dimensions that are likely to apply across the items, given what the homeowner described. Examples (not exhaustive): drainage, easements, impervious surface limits, setbacks, neighbor approval, materials, color, height, visibility from street, submittal requirements. Include every concern that could plausibly apply — err on the side of inclusion. Do NOT include concerns that clearly do not apply (e.g. "fence height" if no fence is mentioned).

3. homeowner_name — the homeowner's name if it appears in the request with HIGH confidence. Look for signatures, "from X", "my name is X", "we, X and Y, ...". If you are not confident, return null. Do not guess.

Return JSON only, this exact schema:
{
  "items": [string, ...],
  "global_concerns": [string, ...],
  "homeowner_name": string | null
}"""


def parse_request(client: anthropic.Anthropic, request_text: str) -> Dict[str, Any]:
    response = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=1024,
        system=PARSER_SYSTEM,
        messages=[{"role": "user", "content": request_text}],
    )
    parsed = parse_json_response(response.content[0].text)
    return {
        "items": parsed.get("items") or [],
        "global_concerns": parsed.get("global_concerns") or [],
        "homeowner_name": parsed.get("homeowner_name"),
    }

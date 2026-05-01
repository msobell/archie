import json
import os
from typing import Dict, Any
import anthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

REVIEW_MODEL = "claude-sonnet-4-6"

REVIEW_SYSTEM = """You are Archie, an HOA Design Review Bot.
Review the resident's design request against the provided community guidelines.

DECISION CRITERIA:
- APPROVAL: Request explicitly meets all relevant criteria and violates none.
- DENIAL: Request explicitly violates a specific, unambiguous rule.
- CONDITIONAL APPROVAL: Default to this if rules are ambiguous or subjective, there is a conflict between rules, or you are unsure for any reason.

CITATION GUIDELINES:
- Provide a citation for every claim.
- Use the SOURCE, ARTICLE, and SECTION from the context.
- For each citation, copy a short verbatim excerpt (1-2 sentences) from the CONTENT that directly supports your decision.

Respond with valid JSON only, using this exact schema:
{
  "decision": "Approval" | "Denial" | "Conditional Approval",
  "reasoning": "string",
  "citations": [{"source": "string", "article": "string", "section": "string", "quote": "string"}],
  "hitl_required": true | false
}"""

EMAIL_SYSTEM = """You are drafting a short email from an HOA Architectural Review Committee to a homeowner.

Tone and style:
- Open with "Hi [first name],"
- One sentence: acknowledge the request and state the decision
- If Conditional Approval: bullet the specific items they need to clarify or provide — be concrete, not generic
- If Denial: one or two plain sentences explaining why, referencing the relevant rule directly
- If Approval: one sentence confirming, noting any conditions
- Close with "Please let me know if you have any questions." then "Thanks," then "The ARC"
- Do not explain the guidelines or quote rule text in the body — use plain language
- No filler phrases like "We appreciate your submission" or "We look forward to hearing from you"
- Keep it under 150 words

Output only the email body — no subject line, no extra commentary."""


class ARCHIEReviewer:
    def __init__(self, persist_dir: str = ".chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        self.client = anthropic.Anthropic()

    def review_request(self, request_text: str) -> Dict[str, Any]:
        docs = self.vectorstore.similarity_search(request_text, k=5)
        context = "\n\n".join([
            f"SOURCE: {d.metadata['source']} | {d.metadata['article']} | {d.metadata['section']}\nCONTENT: {d.page_content}"
            for d in docs
        ])

        response = self.client.messages.create(
            model=REVIEW_MODEL,
            max_tokens=2048,
            system=REVIEW_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"CONTEXT FROM GUIDELINES:\n{context}\n\nRESIDENT REQUEST: {request_text}"
            }]
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    def draft_email(self, result: Dict[str, Any], homeowner_name: str) -> str:
        citations_text = "\n".join(
            f"- {c.get('source', '')}: {c.get('article', '')} {c.get('section', '')} | \"{c.get('quote', '')}\""
            for c in result.get('citations', [])
        )

        response = self.client.messages.create(
            model=REVIEW_MODEL,
            max_tokens=512,
            system=EMAIL_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"Homeowner name: {homeowner_name}\n"
                    f"Decision: {result.get('decision', '')}\n"
                    f"Reasoning: {result.get('reasoning', '')}\n"
                    f"Citations:\n{citations_text}"
                )
            }]
        )
        return response.content[0].text

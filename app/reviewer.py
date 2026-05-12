import re
from typing import Any, Dict
import anthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.agents import SONNET_MODEL
from app.agents.parser import parse_request
from app.agents.retriever import retrieve
from app.agents.evaluator import evaluate
from app.agents.synthesizer import synthesize

# Re-exported for the MCP server's status tool.
REVIEW_MODEL = SONNET_MODEL


SOURCE_LABELS = {
    "Rules_Regulations_Policies_Guidelines.txt": "Design Guidelines",
    "Bylaws.txt": "Bylaws",
    "Covenants.txt": "Covenants",
}

# Matches "2.39 Play Equipment" at the start of a section metadata string.
_SECTION_RE = re.compile(r'^(\d+[\.\d]*)\s+(.+?)(?:\s{2,}|$)')


def _parse_section(raw: str) -> tuple[str, str]:
    m = _SECTION_RE.match(raw.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return "", raw.strip()


def _email_system(hoa_name: str) -> str:
    return _EMAIL_SYSTEM_TEMPLATE.format(hoa_name=hoa_name)


_EMAIL_SYSTEM_TEMPLATE = """You are rendering an email from an HOA Architectural Review Committee to a homeowner.
You will be given a list of pre-selected citations, each with a topic, a section reference, a verbatim quote, and a concrete ask. Your job is ONLY to format them into the email below — do not add new citations, do not drop any, do not reorder topics across distinct items.

FORMAT — output exactly this structure, nothing else:

Subject: ARC Decision – [brief description of request]

Hi [homeowner name(s)],

[One sentence acknowledging the request and stating the decision.]

[One numbered item per provided citation, in the order given. Each item must follow this exact pattern:]

N. [TOPIC] – [Short issue title derived from the ask]
Per [Section X.XX (Section Name)] of the [Source document name]: "[the provided QUOTE, verbatim, in double quotes]" [The provided ask, verbatim or lightly cleaned up.]

[Closing paragraph. For Conditional Approval: ask them to resubmit with the above details; note the ARC strives to respond promptly and may take up to 45 days after a complete submittal is received. For Approval: confirm they may proceed. For Denial: state they may not proceed and cite the rule.]

Thank you for going through the proper approval process before beginning work — we appreciate it!

Warm regards,
The Architectural Review Committee
{hoa_name}

RULES:
- Use ONLY the citations provided. Do not invent sections or quotes.
- The rule text must be quoted VERBATIM from the provided QUOTE field, inside double quotes. Do not paraphrase, summarize, or alter wording. If the quote is awkwardly long, you may trim to the most relevant sentence — but only by truncating, not by rewording.
- Produce one numbered item per citation. Do not merge or drop citations.
- Section references must use the format: Per Section X.XX (Name) of the [Document].
- Do not add any text outside the format above."""


class ARCHIEReviewer:
    def __init__(self, persist_dir: str = ".chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        self.client = anthropic.Anthropic()
        self.hoa_name = self._detect_hoa_name()

    def _detect_hoa_name(self) -> str:
        docs = self.vectorstore.similarity_search("homeowners association name", k=3)
        for doc in docs:
            m = re.search(r'([A-Z][A-Za-z0-9 ]+ Homeowners Association)', doc.page_content)
            if m:
                return m.group(1).strip()
        return "the Homeowners Association"

    def review_request(self, request_text: str) -> Dict[str, Any]:
        """Multi-agent pipeline: parser → retriever → evaluator (per topic) → synthesizer."""
        parsed = parse_request(self.client, request_text)
        topics = list(parsed["items"]) + list(parsed["global_concerns"])

        evaluator_results = []
        retrieval_failures = []
        evaluator_failures = []

        for topic in topics:
            retrieval = retrieve(self.vectorstore, topic)
            chunks = retrieval["primary_chunks"] or retrieval["fallback_chunks"]
            if not chunks:
                retrieval_failures.append(topic)
                continue
            try:
                result = evaluate(self.client, topic, request_text, chunks)
                evaluator_results.append(result)
            except Exception as e:
                evaluator_failures.append(f"Evaluator failed for '{topic}': {e}")

        return synthesize(
            self.client,
            parsed,
            evaluator_results,
            retrieval_failures,
            evaluator_failures,
            request_text,
        )

    def _format_citation_for_email(self, c: Dict[str, Any]) -> str:
        source_label = SOURCE_LABELS.get(c.get("source", ""), c.get("source", ""))
        section_raw = c.get("section", "")
        num, name = _parse_section(section_raw)
        if num and name:
            ref = f"Section {num} ({name}) of the {source_label}"
        elif section_raw:
            ref = f"{section_raw} of the {source_label}"
        else:
            ref = source_label
        return (
            f"- TOPIC: {c.get('topic_label', '').strip()}\n"
            f"  REFERENCE: {ref}\n"
            f"  QUOTE: \"{c.get('quote', '').strip()}\"\n"
            f"  ASK: {c.get('ask', '').strip()}"
        )

    def draft_email(self, result: Dict[str, Any], homeowner_name: str) -> str:
        citations = result.get("citations", [])
        citation_block = "\n".join(self._format_citation_for_email(c) for c in citations) or "(none)"

        response = self.client.messages.create(
            model=SONNET_MODEL,
            max_tokens=2000,
            system=_email_system(self.hoa_name),
            messages=[{
                "role": "user",
                "content": (
                    f"Homeowner name: {homeowner_name}\n"
                    f"Decision: {result.get('decision', '')}\n"
                    f"Resident request: {result.get('_request', '')}\n\n"
                    f"CITATIONS (render one numbered item per citation, in order):\n{citation_block}"
                )
            }]
        )
        return response.content[0].text

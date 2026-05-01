import os
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class ReviewResult(BaseModel):
    decision: str = Field(description="One of: 'Approval', 'Denial', or 'Conditional Approval'")
    reasoning: str = Field(description="Detailed explanation for the decision")
    citations: List[Dict[str, str]] = Field(description="List of specific sections/articles cited")
    hitl_required: bool = Field(description="True if human intervention is needed due to ambiguity or conflict")

class ARCHIEReviewer:
    def __init__(self, persist_dir: str = ".chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        self.llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=ReviewResult)

    def review_request(self, request_text: str) -> Dict[str, Any]:
        # 1. Retrieve relevant rules
        docs = self.vectorstore.similarity_search(request_text, k=5)
        context = "\n\n".join([
            f"SOURCE: {d.metadata['source']} | {d.metadata['article']} | {d.metadata['section']}\nCONTENT: {d.page_content}"
            for d in docs
        ])

        # 2. Evaluation Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Archie, an HOA Design Review Bot. 
Your task is to review a resident's design request against the provided community guidelines.

DECISION CRITERIA:
- APPROVAL: Request explicitly meets all relevant criteria and violates none.
- DENIAL: Request explicitly violates a specific, unambiguous rule.
- CONDITIONAL APPROVAL (HITL): Default to this if:
    - Rules are ambiguous or subjective (e.g., "harmonious", "aesthetic").
    - There is a conflict between different rules.
    - You are unsure for any reason.

CITATION GUIDELINES:
- For every claim, you must provide a citation.
- Use the 'SOURCE', 'ARTICLE', and 'SECTION' provided in the context for each guideline.
- If 'ARTICLE' is 'Unknown', just use the 'SECTION' and 'SOURCE'.
- For each citation, copy a short verbatim excerpt (1–2 sentences) from the CONTENT of the matched guideline chunk that directly supports your decision.
- The citations in your JSON output must contain the keys: "source", "article", "section", and "quote".

CONTEXT FROM GUIDELINES:
{context}"""),
            ("human", "RESIDENT REQUEST: {request_text}\n\n{format_instructions}")
        ])

        chain = prompt | self.llm | self.parser
        
        result = chain.invoke({
            "context": context,
            "request_text": request_text,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result

    def draft_email(self, result: Dict[str, Any], homeowner_name: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are drafting a short email from an HOA Architectural Review Committee to a homeowner.

Tone and style (follow this closely):
- Open with "Hi [first name],"
- One sentence: acknowledge the request and state the decision
- If Conditional Approval: bullet the specific items they need to clarify or provide — be concrete, not generic
- If Denial: one or two plain sentences explaining why, referencing the relevant rule directly
- If Approval: one sentence confirming, noting any conditions
- Close with "Please let me know if you have any questions." then "Thanks," then "The ARC"
- Do not explain the guidelines or quote rule text in the body — use plain language
- No filler phrases like "We appreciate your submission" or "We look forward to hearing from you"
- Keep it under 150 words

Output only the email body — no subject line, no extra commentary."""),
            ("human", """Homeowner name: {homeowner_name}
Decision: {decision}
Reasoning: {reasoning}
Citations:
{citations}""")
        ])

        citations_text = "\n".join(
            f"- {c.get('source', '')}: {c.get('article', '')} {c.get('section', '')} | \"{c.get('quote', '')}\""
            for c in result.get('citations', [])
        )

        chain = prompt | self.llm
        response = chain.invoke({
            "homeowner_name": homeowner_name,
            "decision": result.get('decision', ''),
            "reasoning": result.get('reasoning', ''),
            "citations": citations_text,
        })
        return response.content

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    reviewer = ARCHIEReviewer()

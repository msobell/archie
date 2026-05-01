import argparse
import sys
import os

from app.ingest import ingest_documents
from app.ocr import convert_pdfs
from app.reviewer import ARCHIEReviewer

def main():

    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in environment. Please ensure it is set in your .envrc file.")
        sys.exit(1)

    # Ensure GOOGLE_API_KEY is set for LangChain compatibility
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

    parser = argparse.ArgumentParser(description="Archie: HOA Design Review Bot")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="OCR source PDFs from docs/source/ and ingest into the vector database")

    # Review command
    review_parser = subparsers.add_parser("review", help="Review a design request")
    review_parser.add_argument("request", type=str, help="The resident's design request text")
    review_parser.add_argument("--draft-email", metavar="HOMEOWNER_NAME", help="Also draft a decision email for the named homeowner")

    args = parser.parse_args()

    if args.command == "ingest":
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY not found in environment. Please ensure it is set in your .envrc file.")
            sys.exit(1)
        print("Step 1: Converting source PDFs to text...")
        convert_pdfs("docs/source", "docs")
        print("\nStep 2: Ingesting text documents into vector database...")
        ingest_documents("docs", ".chroma_db")
    elif args.command == "review":
        if not os.path.exists(".chroma_db"):
            print("Error: Vector database not found. Please run 'ingest' first.")
            sys.exit(1)
            
        reviewer = ARCHIEReviewer()
        print(f"Reviewing request: {args.request}\n")
        try:
            result = reviewer.review_request(args.request)
            print(f"DECISION: {result['decision']}")
            print(f"HITL REQUIRED: {result['hitl_required']}")
            print(f"\nREASONING:\n{result['reasoning']}")
            print("\nCITATIONS:")
            for cite in result['citations']:
                source = cite.get('source', 'Unknown')
                article = cite.get('article', '')
                if article == 'Unknown': article = ''
                section = cite.get('section', '')
                quote = cite.get('quote', '')
                print(f"- {source}: {article} {section}".strip().replace("  ", " "))
                if quote:
                    print(f'  "{quote}"')

            if args.draft_email:
                print("\n--- DRAFT EMAIL ---")
                draft = reviewer.draft_email(result, args.draft_email)
                print(draft)
                print("-------------------")
        except Exception as e:
            print(f"Error during review: {e}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

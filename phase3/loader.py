from pathlib import Path
import json
import re
from datetime import date

import yaml


HEADING = re.compile(r"^(#+)\s+(\d+(?:\.\d+)*)\.?\s+(.*)$")


def load_corpus(corpus_dir) -> dict:
    """
    Load all Markdown documents from corpus_dir.

    Returns:
        {
            "returns-refunds": {
                "metadata": {...},
                "sections": {
                    "1": {
                        "title": "Standard Return Window",
                        "text": "..."
                    },
                    "2.2": {
                        "title": "Store #42 Clearance Override",
                        "text": "..."
                    }
                }
            }
        }
    """

    corpus_dir = Path(corpus_dir)
    corpus = {}

    for path in sorted(corpus_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")

        # Frontmatter must be enclosed by ---
        if not content.startswith("---"):
            raise ValueError(
                f"{path.name}: missing YAML frontmatter"
            )

        parts = content.split("---", 2)

        if len(parts) != 3:
            raise ValueError(
                f"{path.name}: invalid frontmatter format"
            )

        _, frontmatter_text, body = parts

        metadata = yaml.safe_load(frontmatter_text)

        if not isinstance(metadata, dict):
            raise ValueError(
                f"{path.name}: frontmatter must be a YAML mapping"
            )

        if isinstance(metadata.get("effective_date"), date):
            metadata["effective_date"] = metadata["effective_date"].isoformat()

        doc_id = metadata.get("doc_id")

        if not doc_id:
            raise ValueError(
                f"{path.name}: missing doc_id in frontmatter"
            )

        sections = {}

        current_section_number = None
        current_section_title = None
        current_lines = []

        for line in body.splitlines():
            match = HEADING.match(line)

            if match:
                # Save the previous section before starting a new one.
                if current_section_number is not None:
                    text = "\n".join(current_lines).strip()

                    sections[current_section_number] = {
                        "title": current_section_title,
                        "text": text,
                    }

                current_section_number = match.group(2)
                current_section_title = match.group(3)
                current_lines = []

            elif current_section_number is not None:
                current_lines.append(line)

        # Save the final section.
        if current_section_number is not None:
            text = "\n".join(current_lines).strip()

            sections[current_section_number] = {
                "title": current_section_title,
                "text": text,
            }

        corpus[doc_id] = {
            "metadata": metadata,
            "sections": sections,
        }

    return corpus


def load_golden(path) -> list[dict]:
    """
    Load a JSONL golden set.

    Each non-empty line must contain one JSON object.
    """

    path = Path(path)
    golden = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                golden.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {e}"
                ) from e

    return golden


def validate_golden(golden, corpus) -> list[str]:
    """
    Validate that every source referenced by the golden set
    exists in the loaded corpus.

    Unanswerable questions have an empty sources list and
    therefore require no source validation.
    """

    errors = []

    for question in golden:
        question_id = question.get("id", "<missing id>")

        for source in question.get("sources", []):
            doc_id = source.get("doc")
            section_id = source.get("section")

            if doc_id not in corpus:
                errors.append(
                    f"{question_id}: source document not found: {doc_id}"
                )
                continue

            if section_id not in corpus[doc_id]["sections"]:
                errors.append(
                    f"{question_id}: source section not found: "
                    f"{doc_id} §{section_id}"
                )

    return errors


if __name__ == "__main__":
    base_dir = Path(__file__).parent

    corpus_dir = base_dir / "corpus"
    golden_path = base_dir / "golden" / "golden_set.jsonl"

    corpus = load_corpus(corpus_dir)
    golden = load_golden(golden_path)

    document_count = len(corpus)
    section_count = sum(
        len(document["sections"])
        for document in corpus.values()
    )
    question_count = len(golden)

    print(f"Documents: {document_count}")
    print(f"Sections:  {section_count}")
    print(f"Questions: {question_count}")

    errors = validate_golden(golden, corpus)

    if errors:
        print("\nValidation errors:")

        for error in errors:
            print(f"- {error}")
    else:
        print("✓ all golden sources resolve")
"""Create a local provenance manifest for the supplied MedQA English textbooks.

The MedQA paper describes the textbook collection as released under a
research-use-only license agreement. This manifest preserves that restriction;
it does not grant rights or authorize redistribution.
"""

import argparse
import json
from datetime import date
from pathlib import Path


SOURCE_DIR = Path("data/MedQA-USMLE/textbooks/en")
SOURCE_URL = "https://github.com/jind11/MedQA"
LICENSE_NOTE = (
    "Text extract supplied with MedQA-USMLE. The MedQA paper describes the "
    "textbook collection as released under a research-use-only license agreement. "
    "Use only under the terms applicable to this copy; local research processing; "
    "do not redistribute source text or derived passages."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=SOURCE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/medqa_sources_manifest.json"),
    )
    args = parser.parse_args()
    files = sorted(args.sources.glob("*.txt"))
    if not files:
        raise SystemExit(f"No English textbook text files found under {args.sources}")

    accessed_date = date.today().isoformat()
    records = []
    for path in files:
        stem = path.stem
        records.append(
            {
                "document_id": f"medqa-us-{stem.casefold().replace('_', '-')}",
                "title": f"{stem.replace('_', ' ')} (MedQA-USMLE supplied text extract)",
                "publisher": "MedQA-USMLE corpus; underlying publisher not identified in supplied files",
                "url": SOURCE_URL,
                "accessed_date": accessed_date,
                "license_note": LICENSE_NOTE,
                "path": f"../MedQA-USMLE/textbooks/en/{path.name}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"textbooks": len(records), "manifest": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

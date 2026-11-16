from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detector import DEFAULT_BACKENDS, DetectorEnsemble, create_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect AI-generated English/Russian text")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to analyze")
    source.add_argument("--file", type=Path, help="UTF-8 text file to analyze")
    parser.add_argument("--model", choices=["all", *DEFAULT_BACKENDS], default="all")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = args.file.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("provide --text, --file, or pipe text through stdin")

    detector = DetectorEnsemble() if args.model == "all" else create_detector(args.model)
    result = detector.detect(text, args.threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

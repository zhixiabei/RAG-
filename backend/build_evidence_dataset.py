from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata


DEFAULT_INPUT = Path("testsets/heishanliang_rag_eval_v1.0.0_chunk_candidates.jsonl")
DEFAULT_OUTPUT = Path("testsets/heishanliang_rag_eval_v1.0.0_evidence.jsonl")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")
FACT_SPLIT_PATTERN = re.compile(r"(?<=[\u3002\uff01\uff1f\uff1b;!?])|\n+")
ENUMERATION_PATTERN = re.compile(
    r"^\s*(?:[-*]+|\d+[.\u3001]|[\uff08(]\d+[\uff09)]|"
    r"[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\u3001.])\s*"
)
HEADING_PATTERN = re.compile(r"^.{0,24}[\uff1a:]$")
SENTENCE_PATTERN = re.compile(
    r"[^\u3002\uff01\uff1f\uff1b;!?\n]+[\u3002\uff01\uff1f\uff1b;!?]?"
)


def clean_text(value: object, *, preserve_lines: bool = False) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = CONTROL_PATTERN.sub("\n", text)
    text = text.replace("\r", "\n")
    if preserve_lines:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)
    return re.sub(r"\s+", " ", text).strip()


def split_facts(answer: object) -> list[str]:
    text = clean_text(answer, preserve_lines=True)
    facts = []
    for part in FACT_SPLIT_PATTERN.split(text):
        fact = ENUMERATION_PATTERN.sub("", part).strip()
        fact = fact.strip(" -")
        if not fact or HEADING_PATTERN.fullmatch(fact):
            continue
        if len(clean_text(fact)) < 8:
            continue
        facts.append(fact)
    return list(dict.fromkeys(facts))


def normalized_number(value: str) -> str:
    suffix = "%" if value.endswith("%") else ""
    raw = value[:-1] if suffix else value
    try:
        number = float(raw)
    except ValueError:
        return value
    normalized = str(int(number)) if number.is_integer() else f"{number:.8f}".rstrip("0")
    return normalized + suffix


def numbers(value: str) -> set[str]:
    return {normalized_number(match.group(0)) for match in NUMBER_PATTERN.finditer(value)}


def lexical_units(value: str) -> set[str]:
    text = "".join(character.casefold() for character in clean_text(value) if character.isalnum())
    if not text:
        return set()
    size = 2 if len(text) > 1 else 1
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def match_score(fact: str, passage: str) -> tuple[float, float, float]:
    fact_units = lexical_units(fact)
    passage_units = lexical_units(passage)
    if not fact_units or not passage_units:
        return 0.0, 0.0, 0.0
    overlap = len(fact_units & passage_units)
    lexical_recall = overlap / len(fact_units)
    lexical_precision = overlap / min(len(passage_units), max(len(fact_units) * 3, 1))
    lexical_score = lexical_recall * 0.8 + min(1.0, lexical_precision) * 0.2

    fact_numbers = numbers(fact)
    passage_numbers = numbers(passage)
    number_recall = (
        len(fact_numbers & passage_numbers) / len(fact_numbers)
        if fact_numbers
        else 1.0
    )
    score = (
        lexical_score * 0.45 + number_recall * 0.55
        if fact_numbers
        else lexical_score
    )
    return score, lexical_recall, number_recall


def source_passages(text: str) -> list[str]:
    cleaned = clean_text(text, preserve_lines=True)
    lines = cleaned.split("\n")
    passages = []

    for sentence in SENTENCE_PATTERN.findall(cleaned):
        sentence = clean_text(sentence)
        if len(sentence) >= 12:
            passages.append(sentence)

    for start in range(len(lines)):
        combined = []
        length = 0
        for line in lines[start:start + 12]:
            if not line:
                continue
            combined.append(line)
            length += len(line)
            if length >= 80:
                passages.append(" ".join(combined))
            if length >= 300:
                break

    flat = clean_text(cleaned)
    if flat:
        for start in range(0, len(flat), 120):
            passage = flat[start:start + 300]
            if len(passage) >= 40:
                passages.append(passage)

    unique = []
    seen = set()
    for passage in passages:
        normalized = clean_text(passage)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def trim_span(passage: str, fact: str, max_chars: int = 320) -> str:
    passage = clean_text(passage)
    if len(passage) <= max_chars:
        return passage

    anchors = []
    for value in sorted(numbers(fact), key=len, reverse=True):
        position = passage.find(value.rstrip("%"))
        if position >= 0:
            anchors.append(position)
    for unit in sorted(lexical_units(fact), key=len, reverse=True):
        position = passage.find(unit)
        if position >= 0:
            anchors.append(position)
        if len(anchors) >= 20:
            break
    center = sorted(anchors)[len(anchors) // 2] if anchors else len(passage) // 2
    start = max(0, min(len(passage) - max_chars, center - max_chars // 2))
    return passage[start:start + max_chars].strip()


def select_evidence(fact: str, source_items: list[dict]) -> dict | None:
    best = None
    for source_index, item in enumerate(source_items):
        for passage in source_passages(item.get("text_span") or ""):
            score, lexical_recall, number_recall = match_score(fact, passage)
            candidate = (
                score,
                number_recall,
                lexical_recall,
                -len(passage),
                -source_index,
                item,
                passage,
            )
            if best is None or candidate[:5] > best[:5]:
                best = candidate
    if best is None:
        return None

    score, number_recall, lexical_recall, _, _, item, passage = best
    fact_numbers = numbers(fact)
    sufficiently_supported = (
        score >= 0.34
        and lexical_recall >= 0.18
        and (not fact_numbers or number_recall >= 0.25)
    )
    if not sufficiently_supported:
        return None
    return {
        "document": str(item.get("document") or "").strip() or None,
        "page": item.get("page"),
        "section": str(item.get("section") or "").strip() or None,
        "text_span": trim_span(passage, fact),
    }


def convert_sample(sample: dict) -> dict:
    answer = sample.get("gold_answer")
    source_items = [
        item for item in sample.get("evidence") or []
        if isinstance(item, dict) and str(item.get("text_span") or "").strip()
    ]
    evidence = []
    for fact in split_facts(answer):
        source = select_evidence(fact, source_items)
        if source is None:
            continue
        evidence.append({
            "id": f"e{len(evidence) + 1}",
            "fact": fact,
            **source,
        })

    result = {
        "question_id": sample.get("question_id"),
        "query": sample.get("query") or sample.get("question"),
        "gold_answer": answer,
        "evidence": evidence,
    }
    for key in (
        "question_type",
        "difficulty",
        "should_refuse",
        "requires_multiple_chunks",
        "language",
        "refusal_reason",
        "status",
        "notes",
    ):
        if key in sample:
            result[key] = sample[key]
    result["notes"] = (
        "Evidence facts are evaluated by embedding similarity; text_span is provenance only."
    )
    return result


def build_dataset(input_path: Path, output_path: Path) -> tuple[int, int]:
    samples = []
    with input_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
            samples.append(convert_sample(sample))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    evidence_count = sum(len(sample["evidence"]) for sample in samples)
    return len(samples), evidence_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a fact-level Evidence dataset from reviewed chunk candidates."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sample_count, evidence_count = build_dataset(args.input, args.output)
    print(f"Wrote {sample_count} samples and {evidence_count} evidence facts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


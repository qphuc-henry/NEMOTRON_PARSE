"""Convert PDFs to Markdown with evidence-first Nemotron OCR.

The pipeline deliberately does not calculate, interpolate, permute, or restore
chart values from surrounding prose. Charts are emitted as Markdown tables or
raw OCR data, never as cropped image assets. Confirmed chart tables require
structurally valid direct OCR evidence, either position-bound visible labels or
agreeing independent reads.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path("/home/cqphuc/NVIDIA-Nemotron-Parse-2.0")
PROMPT_FILES = {
    "text": SCRIPT_DIR / "NEMOTRON_TEXT_PROMPT.txt",
    "chart": SCRIPT_DIR / "NEMOTRON_CHART_PROMPT.txt",
    "label": SCRIPT_DIR / "NEMOTRON_LABEL_PROMPT.txt",
    "ocr": SCRIPT_DIR / "NEMOTRON_OCR_PROMPT.txt",
    "plain": SCRIPT_DIR / "NEMOTRON_PLAIN_PROMPT.txt",
    "fast": SCRIPT_DIR / "NEMOTRON_FAST_PROMPT.txt",
}

END_SEGMENT = re.compile(r"<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>", re.DOTALL)
START_BBOX = re.compile(r"^\s*<x_([\d.]+)><y_([\d.]+)>")
CLASS_ONLY_END = re.compile(r"<class_([^>]+)>")
CONTROL_TOKENS = re.compile(
    r"<(?:predict_bbox|no_bbox|bbox|predict_classes|no_classes|"
    r"output_markdown|output_plain|output_ocr|output_no_text|"
    r"predict_text_in_pic|predict_no_text_in_pic)>"
)
SPECIAL_WRAPPERS = re.compile(r"</?s>|<tbc>|<\|unk\|>|\\unknown")
MODEL_ARTIFACT = re.compile(
    r"(?:the number of items used here is less than|for example,?\s+if any|"
    r"the color size of the background|the table is the table|"
    r"no change of the table|right-to-right order|previous box|bottom box)",
    re.IGNORECASE,
)
CHART_WORDS = re.compile(
    r"\b(?:biểu\s*đồ|đồ\s*thị|chart|graph|plot|diagram|histogram)\b",
    re.IGNORECASE,
)
CHART_SUBTITLE = re.compile(
    r"(?:^|\s)(?:%|thousand|million|billion|percent|axis|bar|line|"
    r"tỷ|triệu|nghìn|đồng|lần|điểm|cột|đường|trục)"
    r"(?=\s|[·,():]|$)",
    re.IGNORECASE,
)
NON_CHART_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+|(?:part|section|chapter|phan)\s+\d+)",
    re.IGNORECASE,
)
TABLE_SEPARATOR = re.compile(r":?-{3,}:?")
NUMBER_TOKEN = re.compile(r"(?<![\w])(?:[+\-−–]?\(?\d[\d., ]*\)?%?)")
VISIBLE_NUMBER_TOKEN = re.compile(r"(?<![\w])(?:[+\-−–]?\(?\d[\d.,]*\)?%?)")
OCR_CHART_TOKEN = re.compile(
    r"[1-4]Q\d{2,4}|Q[1-4](?:/)?\d{2,4}|(?:6T|9T|H[12]|S[12])\d{2,4}|"
    r"(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])[-/.]20\d{2}|"
    r"20\d{2}[-/.](?:0?[1-9]|1[0-2])|(?:0?[1-9]|1[0-2])[-/.]20\d{2}|"
    r"[+\-−–]?\(?\d[\d.,]*\)?(?:%|x)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChartObservation:
    """One chart cell with its semantic address and evidence provenance."""
    category: str
    series: str
    value: str
    source: str
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ChartIR:
    """Structured chart data retained until the final Markdown render."""
    category_name: str
    series: tuple[str, ...]
    categories: tuple[str, ...]
    observations: tuple[ChartObservation, ...]
    source: str


@dataclass(frozen=True)
class Segment:
    order: int
    cls: str
    bbox: tuple[float, float, float, float]
    text: str
    source: str = "pass1"
    chart_ir: ChartIR | None = None


@dataclass(frozen=True)
class PdfTextLine:
    bbox: tuple[float, float, float, float]
    text: str


@dataclass(frozen=True)
class PdfTextPage:
    width: float
    height: float
    lines: tuple[PdfTextLine, ...]
    render_width: int = 0
    render_height: int = 0


@dataclass(frozen=True)
class PageResult:
    page_no: int
    markdown: str
    chart_targets: int
    confirmed_charts: int
    uncertain_charts: int
    text_layer_replacements: int
    elapsed: float
    native_text_coverage: float = 1.0


def fold_text(text: str) -> str:
    value = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def is_source_label(text: str) -> bool:
    return fold_text(text).strip().startswith(("source:", "nguon:"))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", fold_text(text)).strip()


def load_task_prompt(prompt_file: str | Path) -> str:
    prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
    groups = (
        ("bbox", ("<predict_bbox>", "<no_bbox>")),
        ("class", ("<predict_classes>", "<no_classes>")),
        ("output", ("<output_markdown>", "<output_plain>", "<output_ocr>", "<output_no_text>")),
        ("picture text", ("<predict_text_in_pic>", "<predict_no_text_in_pic>")),
    )
    for label, tokens in groups:
        if sum(token in prompt for token in tokens) != 1:
            raise ValueError(f"Prompt must contain exactly one {label} control token")
    return prompt


def clean_generated_text(text: str) -> str:
    text = SPECIAL_WRAPPERS.sub("", text)
    text = CONTROL_TOKENS.sub("", text)
    lines = [line for line in text.splitlines() if not MODEL_ARTIFACT.search(line)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def is_degenerate_ocr(text: str) -> bool:
    """Detect continuation/repetition artifacts that are not source OCR data."""
    cleaned = clean_generated_text(text)
    if not cleaned:
        return bool(text.strip())
    if MODEL_ARTIFACT.search(cleaned):
        return True
    lines = [normalize_text(line) for line in cleaned.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    unique_ratio = len(set(lines)) / len(lines)
    most_common = Counter(lines).most_common(1)[0][1]
    return unique_ratio < 0.35 or most_common >= 5


def is_plausible_chart_transcription(text: str, context: str = "") -> bool:
    """Allow compact source transcriptions, reject structured hallucinations."""
    cleaned = clean_generated_text(text)
    if not cleaned or is_degenerate_ocr(cleaned):
        return False
    if len(cleaned) > 6000 or len(cleaned.splitlines()) > 100:
        return False
    if len(NUMBER_TOKEN.findall(cleaned)) > 80:
        return False
    if markdown_tables(cleaned) and chart_quality_issues(cleaned, context):
        return False
    return True


def parse_segments(raw: str, source: str = "pass1") -> list[Segment]:
    """Parse bbox/class output, including Nemotron's omitted first start bbox."""
    segments: list[Segment] = []
    cursor = 0
    for order, match in enumerate(END_SEGMENT.finditer(raw)):
        body = re.sub(r"^\s*</?s>\s*", "", raw[cursor:match.start()])
        body = CONTROL_TOKENS.sub("", body)
        start = START_BBOX.match(body)
        if start:
            x1, y1 = float(start.group(1)), float(start.group(2))
            body = body[start.end():]
        else:
            x1, y1 = 0.0, 0.0
        cls = match.group(3).strip()
        text = clean_generated_text(body)
        if text or cls.casefold() in {"chart", "picture"}:
            segments.append(
                Segment(
                    order,
                    cls,
                    (x1, y1, float(match.group(1)), float(match.group(2))),
                    text,
                    source,
                )
            )
        cursor = match.end()
    if segments:
        return segments

    cursor = 0
    for order, match in enumerate(CLASS_ONLY_END.finditer(raw)):
        text = clean_generated_text(raw[cursor:match.start()])
        cls = match.group(1).strip()
        if text or cls.casefold() in {"chart", "picture"}:
            y1 = float(order)
            segments.append(Segment(order, cls, (0.0, y1, 1.0, y1 + 1.0), text, source))
        cursor = match.end()
    return segments


def clean_unstructured_output(raw: str) -> str:
    text = clean_generated_text(raw)
    # Nemotron sometimes emits incomplete bbox fragments such as
    # ``<x_0.3105>`` around picture-only regions. They are model control
    # coordinates, never visible chart values.
    text = re.sub(r"<[xy]_[\d.]+>", "", text)
    return re.sub(r"<class_[^>]+>", "", text).strip()


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def markdown_row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def dedupe_consecutive_rows(rows: list[list[str]]) -> list[list[str]]:
    """Drop only byte-identical adjacent rows produced by duplicate PDF text."""
    result: list[list[str]] = []
    for row in rows:
        if not result or row != result[-1]:
            result.append(row)
    return result


def markdown_tables(text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        end = index
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1
        rows = [split_markdown_row(line) for line in lines[index:end]]
        if len(rows) >= 3:
            tables.append(rows)
        index = end
    return tables


def chart_quality_issues(text: str, context: str = "") -> list[str]:
    """Return evidence/structure failures without changing any chart value."""
    cleaned = clean_generated_text(text)
    if not cleaned:
        return ["empty OCR result"]
    if MODEL_ARTIFACT.search(text):
        return ["model continuation artifact"]
    tables = markdown_tables(cleaned)
    if not tables:
        return ["no Markdown data table"]

    issues: list[str] = []
    usable_table = False
    for rows in tables:
        separator = next(
            (
                i
                for i, row in enumerate(rows)
                if row and all(TABLE_SEPARATOR.fullmatch(cell) for cell in row)
            ),
            None,
        )
        if separator is None or separator == 0:
            issues.append("missing table separator")
            continue
        width = len(rows[separator - 1])
        data = rows[separator + 1:]
        header = [fold_text(cell.replace("**", "")) for cell in rows[separator - 1]]
        if width < 2 or len(data) < 2:
            issues.append("too little chart data")
            continue
        if len(data) > 40:
            issues.append("implausibly many chart rows")
            continue
        if any(len(row) != width for row in rows):
            issues.append("inconsistent table column count")
            continue
        numeric_cells = [
            cell
            for row in data
            for cell in row[1:]
            if NUMBER_TOKEN.search(cell)
        ]
        if len(numeric_cells) < 2:
            issues.append("too few printed numeric labels")
            continue

        first_cells = [row[0].strip() for row in data]
        numeric_first = sum(
            bool(re.fullmatch(r"[+\-−–]?\d[\d., ]*%?", cell))
            for cell in first_cells
        )
        period_first = sum(is_period_label(cell) for cell in first_cells)
        if (
            looks_like_time_chart_context(context)
            and len(data) >= 2
            and period_first / len(data) < 0.80
        ):
            issues.append("non-period categories in a time chart")
            continue
        scatter_context = "scatter" in fold_text(context)
        if (
            len(data) >= 3 and numeric_first / len(data) >= 0.80
            and period_first == 0 and not scatter_context
        ):
            issues.append("numeric values used as category labels")
            continue

        # Generated continuations often form long near-arithmetic sequences,
        # unlike visible labels in ordinary bar/line charts.
        if len(data) >= 8:
            for column in range(width):
                values: list[float] = []
                for row in data:
                    token = row[column].strip().replace("%", "").replace(" ", "")
                    if token.count(",") == 1 and "." not in token:
                        token = token.replace(",", ".")
                    try:
                        values.append(float(token))
                    except ValueError:
                        values = []
                        break
                if len(values) >= 8:
                    deltas = [round(values[i + 1] - values[i], 3) for i in range(len(values) - 1)]
                    if Counter(deltas).most_common(1)[0][1] / len(deltas) >= 0.70:
                        issues.append("synthetic arithmetic sequence")
                        break
            if "synthetic arithmetic sequence" in issues:
                continue
        usable_table = True

    if not usable_table and not issues:
        issues.append("no usable chart table")
    return list(dict.fromkeys(issues))


def canonical_number(token: str) -> str:
    value = token.strip().replace("−", "-").replace("–", "-").replace(" ", "")
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    return value


def numeric_signature(text: str) -> Counter[str]:
    """Numeric labels copied from OCR; used only for agreement, never synthesis."""
    table_text = "\n".join(
        " | ".join(cell for row in table for cell in row)
        for table in markdown_tables(text)
    )
    return Counter(canonical_number(match.group()) for match in NUMBER_TOKEN.finditer(table_text))


def chart_ir_from_markdown(
    text: str, source: str, min_data_rows: int = 2
) -> ChartIR | None:
    """Parse a chart table without losing row/series identity."""
    for rows in markdown_tables(clean_generated_text(text)):
        separator = next((
            index for index, row in enumerate(rows)
            if row and all(TABLE_SEPARATOR.fullmatch(cell) for cell in row)
        ), None)
        if separator is None or separator == 0:
            continue
        header = rows[separator - 1]
        data = rows[separator + 1:]
        if (
            len(header) < 2 or len(data) < min_data_rows
            or any(len(row) != len(header) for row in data)
        ):
            continue
        observations: list[ChartObservation] = []
        for row in data:
            category = row[0].strip()
            if not category:
                continue
            for column, cell in enumerate(row[1:], start=1):
                if cell.strip() in {"", "—", "-"} or not NUMBER_TOKEN.search(cell):
                    continue
                observations.append(ChartObservation(
                    category=category,
                    series=header[column].replace("**", "").strip(),
                    value=canonical_number(cell),
                    source=source,
                ))
        if observations:
            return ChartIR(
                category_name=header[0].strip(),
                series=tuple(cell.strip() for cell in header[1:]),
                categories=tuple(row[0].strip() for row in data if row[0].strip()),
                observations=tuple(observations),
                source=source,
            )
    return None


def chart_ir_from_rows(
    header: list[str], rows: list[list[str]], source: str
) -> ChartIR:
    observations = tuple(
        ChartObservation(row[0], header[column], cell, source)
        for row in rows
        for column, cell in enumerate(row[1:], start=1)
        if cell.strip() not in {"", "—", "-"}
    )
    return ChartIR(
        header[0], tuple(header[1:]),
        tuple(row[0] for row in rows), observations, source,
    )


def transposed_period_chart_ir(text: str, source: str) -> ChartIR | None:
    """Normalize direct OCR tables whose printed periods run across columns."""
    for rows in markdown_tables(text):
        separator = next((
            i for i, row in enumerate(rows)
            if row and all(TABLE_SEPARATOR.fullmatch(cell) for cell in row)
        ), None)
        if separator is None or separator == 0 or separator + 1 >= len(rows):
            continue
        header = rows[separator - 1]
        period_columns = [
            index for index, cell in enumerate(header) if is_period_label(cell)
        ]
        nonempty_headers = [cell for cell in header if cell.strip()]
        value_offset = 1 if period_columns and period_columns[0] == 0 else 0
        if (
            len(period_columns) < 2
            or len(period_columns) / max(1, len(nonempty_headers)) < 0.70
        ):
            continue
        series_rows = [
            row for row in rows[separator + 1:]
            if row and any(
                column < len(row) and compact_numeric_label(row[column]) is not None
                for column in (item + value_offset for item in period_columns)
            )
        ]
        if not series_rows:
            continue
        series = tuple(
            row[0].strip() or f"Chuỗi {index + 1}"
            for index, row in enumerate(series_rows)
        )
        if len(series_rows) == 1:
            direct_values = [
                series_rows[0][column + value_offset]
                for column in period_columns
                if column + value_offset < len(series_rows[0])
            ]
            if direct_values and all("%" in value for value in direct_values):
                # The row label is frequently shifted from the bar legend in a
                # transposed OCR table. Use a neutral unit-bearing name rather
                # than attach correctly read percentages to the wrong series.
                series = ("Giá trị (%)",)
        observations: list[ChartObservation] = []
        categories: list[str] = []
        for column in period_columns:
            category = header[column].strip()
            categories.append(category)
            for row, series_name in zip(series_rows, series):
                value_column = column + value_offset
                if value_column >= len(row):
                    continue
                value = compact_numeric_label(row[value_column])
                if value is not None:
                    observations.append(ChartObservation(
                        category, series_name, value, source
                    ))
        if len(observations) >= 2:
            return ChartIR(
                "Kỳ", series, tuple(categories), tuple(observations), source
            )
    return None


def combo_raw_chart_candidate(
    candidate: Segment | None, direct_line: ChartIR | None, context: str
) -> Segment | None:
    """Retain single-pass bars while accepting only direct line-label values."""
    folded = fold_text(context)
    if not (
        candidate is not None
        and markdown_tables(candidate.text)
        and any(token in folded for token in ("duong", "line", "truc phai"))
        and any(token in folded for token in ("cot", "bar"))
    ):
        return None
    chart = chart_ir_from_markdown(candidate.text, candidate.source)
    if chart is None or len(chart.series) < 2:
        return None
    line_series = chart.series[-1]
    observations = [
        replace(item, source="single_pass_bar")
        for item in chart.observations if item.series != line_series
    ]
    direct_values: dict[str, str] = {}
    if direct_line is not None:
        for item in direct_line.observations:
            direct_values[normalize_text(item.category)] = item.value
    observations.extend(
        ChartObservation(
            category, line_series, direct_values[normalize_text(category)],
            "direct_line_label",
        )
        for category in chart.categories
        if normalize_text(category) in direct_values
    )
    result = ChartIR(
        chart.category_name, chart.series, chart.categories,
        tuple(observations), "combo_raw_evidence",
    )
    if len(observations) < 2:
        return None
    return Segment(
        candidate.order, "Chart", candidate.bbox, render_chart_ir(result),
        "combo_raw_evidence", result,
    )


def render_chart_ir(chart: ChartIR) -> str:
    categories = list(dict.fromkeys(chart.categories))
    values = {
        (item.category, item.series): item.value for item in chart.observations
    }
    rows = [
        [category] + [values.get((category, series), "—") for series in chart.series]
        for category in categories
    ]
    return "\n".join(
        [markdown_row([chart.category_name, *chart.series]),
         markdown_row([":---"] + ["---:"] * len(chart.series))]
        + [markdown_row(row) for row in rows]
    )


def chart_ir_category_coverage(chart: ChartIR | None) -> float:
    """Fraction of categories containing at least one directly read value."""
    if chart is None or not chart.categories:
        return 0.0
    covered = {
        observation.category
        for observation in chart.observations
        if observation.value not in {"", "—", "-", "−", "–"}
    }
    return len(covered) / len(set(chart.categories))


def numeric_label_fingerprint(token: str) -> str:
    """Identity of a printed number, including every meaningful separator."""
    return canonical_number(token).casefold()


def mask_unverified_chart_values(structured: str, evidence: str) -> str:
    """Replace chart values absent from direct OCR evidence with an em dash."""
    evidence_counts = Counter(
        numeric_label_fingerprint(match.group())
        for match in VISIBLE_NUMBER_TOKEN.finditer(clean_generated_text(evidence))
    )
    if not evidence_counts:
        return structured
    available = evidence_counts.copy()
    output: list[str] = []
    inside_table = False
    separator_seen = False
    for line in structured.splitlines():
        if not line.lstrip().startswith("|"):
            inside_table = False
            separator_seen = False
            output.append(line)
            continue
        cells = split_markdown_row(line)
        if not inside_table:
            inside_table = True
            separator_seen = False
            output.append(line)  # Header is metadata, never evidence-masked.
            continue
        if cells and all(TABLE_SEPARATOR.fullmatch(cell) for cell in cells):
            separator_seen = True
            output.append(line)
            continue
        if not separator_seen:
            output.append(line)
            continue
        masked = list(cells)
        for index in range(1, len(masked)):
            matches = list(VISIBLE_NUMBER_TOKEN.finditer(masked[index]))
            if not matches:
                continue
            fingerprints = [numeric_label_fingerprint(match.group()) for match in matches]
            needed = Counter(fingerprints)
            if any(available[value] < count for value, count in needed.items()):
                masked[index] = "—"
            else:
                available.subtract(needed)
        output.append(markdown_row(masked))
    return "\n".join(output)


def mask_all_chart_values(structured: str) -> str:
    """Keep a chart skeleton while removing every unprinted data cell."""
    output: list[str] = []
    inside_table = False
    separator_seen = False
    for line in structured.splitlines():
        if not line.lstrip().startswith("|"):
            inside_table = False
            separator_seen = False
            output.append(line)
            continue
        cells = split_markdown_row(line)
        if not inside_table:
            inside_table = True
            output.append(line)
            continue
        if cells and all(TABLE_SEPARATOR.fullmatch(cell) for cell in cells):
            separator_seen = True
            output.append(line)
            continue
        if separator_seen:
            cells = [cells[0], *["—"] * (len(cells) - 1)]
            output.append(markdown_row(cells))
        else:
            output.append(line)
    return "\n".join(output)


def restore_unique_category_separators(structured: str, evidence: str) -> str:
    """Restore punctuation only for a unique same-category digit match."""
    evidence_by_category: dict[str, list[str]] = {}
    for rows in markdown_tables(evidence):
        separator = next((
            i for i, row in enumerate(rows)
            if row and all(TABLE_SEPARATOR.fullmatch(cell) for cell in row)
        ), None)
        if separator is None:
            continue
        for row in rows[separator + 1:]:
            if not row:
                continue
            evidence_by_category.setdefault(normalize_text(row[0]), []).extend(
                match.group() for cell in row[1:]
                for match in VISIBLE_NUMBER_TOKEN.finditer(cell)
            )

    def punctuationless(token: str) -> str:
        return re.sub(r"[.,]", "", canonical_number(token)).casefold()

    lines = structured.splitlines()
    for line_index, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            continue
        cells = split_markdown_row(line)
        if len(cells) < 2 or TABLE_SEPARATOR.fullmatch(cells[0]):
            continue
        local_evidence = evidence_by_category.get(normalize_text(cells[0]), [])
        if not local_evidence:
            continue
        generated = [
            (column, match.group())
            for column, cell in enumerate(cells[1:], start=1)
            for match in VISIBLE_NUMBER_TOKEN.finditer(cell)
        ]
        for column, token in generated:
            matches = [
                item for item in local_evidence
                if punctuationless(item) == punctuationless(token)
            ]
            same_row = [
                item for _, item in generated
                if punctuationless(item) == punctuationless(token)
            ]
            if len(matches) == 1 and len(same_row) == 1:
                cells[column] = cells[column].replace(token, matches[0], 1)
        lines[line_index] = markdown_row(cells)
    return "\n".join(lines)


def evidence_bound_chart_candidate(
    candidate: Segment | None, evidence: Segment | None, context: str
) -> Segment | None:
    """Keep only generated table cells reproduced by direct local-label OCR."""
    if candidate is None or evidence is None or not markdown_tables(candidate.text):
        return None
    restored = restore_unique_category_separators(candidate.text, evidence.text)
    masked = mask_unverified_chart_values(restored, evidence.text)
    masked = mask_combo_line_without_evidence(masked, context, evidence.text)
    chart = chart_ir_from_markdown(masked, "evidence_bound", min_data_rows=1)
    if chart is None or len(chart.observations) < 2:
        return None
    folded = fold_text(context)
    combo = (
        any(token in folded for token in ("duong", "line", "truc phai"))
        and any(token in folded for token in ("cot", "bar"))
    )
    # A collapsed one-series table cannot tell whether a visible endpoint is a
    # bar or line value, so exposing it would attach correct digits to the
    # wrong series (for example 23,3% as revenue).
    if combo and len(chart.series) < 2:
        return None
    if chart_quality_issues(masked, context):
        return None
    return replace(
        candidate, text=masked, source="evidence_bound_table", chart_ir=chart
    )


def mask_combo_line_without_evidence(
    structured: str, context: str, evidence: str = ""
) -> str:
    """Withhold combo-chart line cells lacking direct printed-label OCR."""
    folded_context = fold_text(context)
    if not (
        any(token in folded_context for token in ("duong", "line", "truc phai"))
        and any(token in folded_context for token in ("cot", "bar"))
    ):
        return structured
    lines = structured.splitlines()
    table_indexes = [i for i, line in enumerate(lines) if line.lstrip().startswith("|")]
    if len(table_indexes) < 3:
        return structured
    start = table_indexes[0]
    header = split_markdown_row(lines[start])
    line_columns = {
        index for index, cell in enumerate(header[1:], start=1)
        if "%" in cell
    }
    if not line_columns and len(header) >= 3:
        line_columns = {len(header) - 1}
    if not line_columns:
        return structured
    available = Counter(
        numeric_label_fingerprint(match.group())
        for match in VISIBLE_NUMBER_TOKEN.finditer(evidence)
    )
    for line_index in table_indexes[2:]:
        cells = split_markdown_row(lines[line_index])
        if len(cells) != len(header):
            continue
        for column in line_columns:
            matches = list(VISIBLE_NUMBER_TOKEN.finditer(cells[column]))
            needed = Counter(
                numeric_label_fingerprint(match.group()) for match in matches
            )
            if not matches or any(available[value] < count for value, count in needed.items()):
                cells[column] = "—"
            else:
                available.subtract(needed)
        lines[line_index] = markdown_row(cells)
    return "\n".join(lines)


def chart_reads_agree(first: str, second: str) -> bool:
    left_ir = chart_ir_from_markdown(first, "read_1")
    right_ir = chart_ir_from_markdown(second, "read_2")
    if left_ir is None or right_ir is None:
        return False
    if (
        chart_ir_category_coverage(left_ir) < 0.95
        or chart_ir_category_coverage(right_ir) < 0.95
    ):
        return False
    left = {
        (normalize_text(item.category), normalize_text(item.series)): canonical_number(item.value)
        for item in left_ir.observations
    }
    right = {
        (normalize_text(item.category), normalize_text(item.series)): canonical_number(item.value)
        for item in right_ir.observations
    }
    common_keys = set(left) & set(right)
    if len(common_keys) < 2:
        return False
    key_coverage = len(common_keys) / min(len(left), len(right))
    size_coverage = len(common_keys) / max(len(left), len(right))
    value_agreement = sum(left[key] == right[key] for key in common_keys) / len(common_keys)
    return key_coverage == 1.0 and size_coverage == 1.0 and value_agreement == 1.0


def keyed_numeric_cell_agreement(first: str, second: str) -> float:
    """Agreement of numeric cells at the same row/column address."""
    left_ir = chart_ir_from_markdown(first, "left", min_data_rows=1)
    right_ir = chart_ir_from_markdown(second, "right", min_data_rows=1)
    if left_ir is None or right_ir is None:
        return 0.0
    left = {
        (normalize_text(item.category), normalize_text(item.series)): canonical_number(item.value)
        for item in left_ir.observations
    }
    right = {
        (normalize_text(item.category), normalize_text(item.series)): canonical_number(item.value)
        for item in right_ir.observations
    }
    if not left or not right:
        return 0.0
    matched = sum(key in right and right[key] == value for key, value in left.items())
    return matched / min(len(left), len(right))


def bbox_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    denominator = min(bbox_area(first), bbox_area(second))
    return intersection / denominator if denominator else 0.0


def horizontal_overlap(first: Segment, second: Segment) -> float:
    intersection = max(0.0, min(first.bbox[2], second.bbox[2]) - max(first.bbox[0], second.bbox[0]))
    denominator = min(first.bbox[2] - first.bbox[0], second.bbox[2] - second.bbox[0])
    return intersection / denominator if denominator > 0 else 0.0


def looks_like_chart_title(text: str) -> bool:
    """Recognize short visual titles while rejecting report section headings."""
    value = re.sub(r"^[#■▪•\-\s]+", "", text.strip()).strip("*_ ")
    if not value or len(value) > 150 or NON_CHART_HEADING.search(fold_text(value)):
        return False
    first = next((line.strip() for line in value.splitlines() if line.strip()), "")
    if not first or len(first.split()) > 18:
        return False
    return bool(CHART_WORDS.search(first))


def looks_like_visual_caption_text(text: str) -> bool:
    """Generic caption shape check based on form, not subject vocabulary."""
    value = re.sub(r"^[#■▪•\-\s]+", "", text.strip()).strip("*_ ")
    folded = fold_text(value)
    if (
        not value or len(value) > 150 or len(value.split()) > 14
        or NON_CHART_HEADING.search(folded)
        or folded.startswith(("source:", "nguon:", "www."))
        or compact_numeric_label(value) is not None
        or bool(re.search(r"[.!?]\s*$", value))
    ):
        return False
    return bool(re.search(r"[^\W\d_]", value, re.UNICODE))


def is_period_label(text: str) -> bool:
    value = fold_text(text).strip().replace(" ", "")
    return bool(re.fullmatch(
        r"(?:[1-4]q\d{2,4}|q[1-4](?:/)?\d{2,4}|"
        r"(?:6t|9t|h[12]|s[12])\d{2,4}|20\d{2}|fy\d{2,4}|"
        r"20\d{2}[-/.](?:0?[1-9]|1[0-2])|"
        r"(?:0?[1-9]|1[0-2])[-/.]20\d{2}|"
        r"(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])[-/.]20\d{2}|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\d{2,4}|"
        r"(?:t|thang)(?:0?[1-9]|1[0-2])(?:/)?\d{2,4})",
        value, re.IGNORECASE,
    ))


def looks_like_table_header(cells: list[str]) -> bool:
    """Detect a header from cell form, without document- or domain-specific words."""
    if (
        len(cells) < 2
        or not re.search(r"[^\W\d_]", cells[0], re.UNICODE)
        or len(cells[0].split()) > 8
    ):
        return False
    tail = [cell.strip() for cell in cells[1:] if cell.strip()]
    if not tail:
        return False
    label_like = sum(
        is_period_label(cell)
        or (
            cell not in {"—", "-", "−", "–"}
            and not re.fullmatch(r"[+\-−–]?\(?\d[\d., ]*\)?%?", cell)
        )
        for cell in tail
    )
    return label_like / len(tail) >= 0.50


def compact_numeric_label(text: str) -> str | None:
    value = text.strip().replace("−", "-").replace("–", "-")
    return value if re.fullmatch(r"[+\-]?\(?\d[\d.,]*\)?(?:%|x)?", value, re.I) else None


def chart_series_names(count: int, kind: str) -> list[str]:
    """Return neutral names only; never infer domain semantics from a title."""
    if kind == "combo":
        return ["Giá trị", "Giá trị (%)"]
    if count == 2:
        return ["Chuỗi 1", "Chuỗi 2"]
    return ["Giá trị"]


def spatial_time_chart_ir(labels: list[Segment], context: str) -> ChartIR | None:
    """Build time-chart rows by binding visible labels to x-axis periods."""
    del context  # Titles never determine values or series semantics.
    periods = [
        label for label in labels
        if is_period_label(label.text) and label.bbox[2] - label.bbox[0] < 0.35
    ]
    period_rows: list[list[Segment]] = []
    for period in sorted(periods, key=lambda item: ((item.bbox[1] + item.bbox[3]) / 2, item.bbox[0])):
        cy = (period.bbox[1] + period.bbox[3]) / 2
        if not period_rows or abs(
            cy - sum((item.bbox[1] + item.bbox[3]) / 2 for item in period_rows[-1])
            / len(period_rows[-1])
        ) > 0.045:
            period_rows.append([period])
        else:
            period_rows[-1].append(period)
    if period_rows:
        # A y-axis tick such as 2000 or a year in the title is not an x-axis
        # category.  The real period row has the most aligned labels and is
        # normally the lowest such row in the plot.
        periods = max(
            period_rows,
            key=lambda row: (
                len(row),
                sum((item.bbox[1] + item.bbox[3]) / 2 for item in row) / len(row),
            ),
        )
    periods.sort(key=lambda item: (item.bbox[0] + item.bbox[2]) / 2)
    deduped: list[Segment] = []
    for period in periods:
        if not deduped or normalize_text(period.text) != normalize_text(deduped[-1].text):
            deduped.append(period)
    periods = deduped
    if len(periods) < 2:
        return None
    period_y = sum((item.bbox[1] + item.bbox[3]) / 2 for item in periods) / len(periods)
    centers = [(item.bbox[0] + item.bbox[2]) / 2 for item in periods]
    if centers[-1] - centers[0] < 0.22:
        return None
    spacing = min(
        (centers[i + 1] - centers[i] for i in range(len(centers) - 1)),
        default=0.2,
    )
    values: list[list[tuple[float, float, str]]] = [[] for _ in periods]
    for label in labels:
        token = compact_numeric_label(label.text)
        if token is None or is_period_label(label.text):
            continue
        cx = (label.bbox[0] + label.bbox[2]) / 2
        cy = (label.bbox[1] + label.bbox[3]) / 2
        if cy >= period_y + 0.02:
            continue
        nearest = min(range(len(centers)), key=lambda index: abs(centers[index] - cx))
        if abs(centers[nearest] - cx) <= max(0.08, spacing * 0.48):
            values[nearest].append((cx, cy, token))

    has_percent = any("%" in token for group in values for _, _, token in group)
    has_amount = any("%" not in token and not token.lower().endswith("x") for group in values for _, _, token in group)
    has_ratio = any(token.lower().endswith("x") for group in values for _, _, token in group)
    combo = has_percent and has_amount
    max_count = max((len(group) for group in values), default=0)
    series_count = 2 if combo or has_ratio or max_count >= 2 else 1
    names = chart_series_names(series_count, "combo" if combo else "regular")
    rows: list[list[str]] = []
    for period, group, center in zip(periods, values, centers):
        if combo:
            amount = next((token for _, _, token in group if "%" not in token), "—")
            line = next((token for _, _, token in group if "%" in token), "—")
            row = [period.text.strip(), amount, line]
        elif series_count == 2:
            # Ratio labels share the same x position. Keep a stable vertical
            # series order instead of letting tiny OCR x jitter swap columns.
            ordered = sorted(group, key=lambda item: item[1] if has_ratio else item[0])
            tokens = [item[2] for item in ordered[:2]]
            row = [period.text.strip()] + tokens + ["—"] * (2 - len(tokens))
        else:
            token = min(group, key=lambda item: abs(item[0] - center))[2] if group else "—"
            row = [period.text.strip(), token]
        rows.append(row)
    if sum(cell != "—" for row in rows for cell in row[1:]) < 2:
        return None
    header = ["Kỳ"] + names
    return chart_ir_from_rows(header, rows, "focused_chart_spatial")


def spatial_time_chart_table(labels: list[Segment], context: str) -> str | None:
    chart = spatial_time_chart_ir(labels, context)
    return render_chart_ir(chart) if chart is not None else None


def positioned_chart_ir(labels: list[Segment], context: str) -> ChartIR | None:
    """Build the strongest chart IR available from positioned OCR labels."""
    return spatial_time_chart_ir(labels, context) or spatial_categorical_chart_ir(
        labels, context
    )


def choose_stronger_spatial_labels(
    original: list[Segment], retry: list[Segment], context: str
) -> list[Segment]:
    """Prevent a sparse per-band retry from erasing a stronger bbox read."""
    def strongest_ir(labels: list[Segment]) -> ChartIR | None:
        candidates = [positioned_chart_ir(labels, context)]
        candidates.extend(
            chart_ir_from_markdown(label.text, label.source, min_data_rows=1)
            for label in labels if markdown_tables(label.text)
        )
        valid = (chart for chart in candidates if chart is not None)
        if looks_like_time_chart_context(context):
            # A fluent OCR table can still be semantically unrelated to the
            # time axis (for example legend colours interpreted as rows).
            # Do not let its larger cell count outrank sparse, directly read
            # endpoint labels that use the printed periods.
            valid = (
                chart for chart in valid
                if chart.categories
                and sum(is_period_label(item) for item in chart.categories)
                / len(chart.categories) >= 0.80
            )
        return max(
            valid,
            key=lambda chart: (
                len(chart.observations), chart_ir_category_coverage(chart)
            ),
            default=None,
        )

    original_ir = strongest_ir(original)
    retry_ir = strongest_ir(retry)

    def score(chart: ChartIR | None) -> tuple[int, float]:
        return (
            (len(chart.observations), chart_ir_category_coverage(chart))
            if chart is not None else (0, 0.0)
        )

    return retry if score(retry_ir) > score(original_ir) else original


def spatial_categorical_chart_ir(labels: list[Segment], context: str) -> ChartIR | None:
    """Build categorical bar-chart rows from spatial label/value positions."""
    del context  # Titles never determine values or series semantics.
    numeric = [
        label for label in labels
        if compact_numeric_label(label.text) is not None and not is_period_label(label.text)
    ]
    if len(numeric) < 2:
        return None
    numeric_y = sorted((item.bbox[1] + item.bbox[3]) / 2 for item in numeric)
    median_numeric_y = numeric_y[len(numeric_y) // 2]
    candidates: list[Segment] = []
    for label in labels:
        text = label.text.strip()
        folded = fold_text(text)
        cy = (label.bbox[1] + label.bbox[3]) / 2
        if (
            not text or compact_numeric_label(text) is not None or is_period_label(text)
            or cy <= median_numeric_y + 0.035
            or any(word in folded for word in ("nguon:", "source:", "legend"))
            or looks_like_chart_title(text)
        ):
            continue
        candidates.append(label)
    if len(candidates) < 2:
        return None
    first_band = min((item.bbox[1] + item.bbox[3]) / 2 for item in candidates)
    candidates = [
        item for item in candidates
        if (item.bbox[1] + item.bbox[3]) / 2 <= first_band + 0.16
    ]
    groups: list[list[Segment]] = []
    for item in sorted(candidates, key=lambda value: (value.bbox[0] + value.bbox[2]) / 2):
        cx = (item.bbox[0] + item.bbox[2]) / 2
        group = next((
            group for group in groups
            if abs(sum((part.bbox[0] + part.bbox[2]) / 2 for part in group) / len(group) - cx) < 0.07
        ), None)
        if group is None:
            groups.append([item])
        else:
            group.append(item)
    if len(groups) < 2:
        return None
    categories = [
        " ".join(part.text.strip() for part in sorted(group, key=lambda value: value.bbox[1]))
        for group in groups
    ]
    centers = [
        sum((part.bbox[0] + part.bbox[2]) / 2 for part in group) / len(group)
        for group in groups
    ]
    order = sorted(range(len(centers)), key=lambda index: centers[index])
    categories = [categories[index] for index in order]
    centers = [centers[index] for index in order]
    spacing = min(
        (centers[i + 1] - centers[i] for i in range(len(centers) - 1)),
        default=0.2,
    )
    assigned: list[list[tuple[float, str]]] = [[] for _ in centers]
    category_y = min((item.bbox[1] + item.bbox[3]) / 2 for item in candidates)
    for label in numeric:
        cx = (label.bbox[0] + label.bbox[2]) / 2
        cy = (label.bbox[1] + label.bbox[3]) / 2
        if cy >= category_y:
            continue
        nearest = min(range(len(centers)), key=lambda index: abs(centers[index] - cx))
        if abs(centers[nearest] - cx) <= max(0.09, spacing * 0.48):
            assigned[nearest].append((cx, label.text.strip()))
    series_count = 2 if max((len(group) for group in assigned), default=0) >= 2 else 1
    names = chart_series_names(series_count, "regular")
    rows: list[list[str]] = []
    for category, group in zip(categories, assigned):
        tokens = [token for _, token in sorted(group, key=lambda item: item[0])[:series_count]]
        rows.append([category] + tokens + ["—"] * (series_count - len(tokens)))
    if sum(cell != "—" for row in rows for cell in row[1:]) < 2:
        return None
    return chart_ir_from_rows(["Hạng mục"] + names, rows, "focused_chart_spatial")


def spatial_categorical_chart_table(labels: list[Segment], context: str) -> str | None:
    chart = spatial_categorical_chart_ir(labels, context)
    return render_chart_ir(chart) if chart is not None else None


def is_metric_card_table(segment: Segment) -> bool:
    return (
        segment.cls.casefold() == "table"
        and segment.source == "text_layer_metric_cards"
    )


def is_report_data_table(segment: Segment) -> bool:
    """Identify ordinary report tables that must not enter chart OCR."""
    tables = markdown_tables(segment.text)
    if not tables or not tables[0] or not tables[0][0]:
        return False
    return segment.source.startswith("text_layer")


def nearby_chart_caption(target: Segment, segments: Iterable[Segment]) -> str:
    """Return only local, short title-like text; never scan surrounding prose."""
    candidates: list[tuple[float, str]] = []
    for segment in segments:
        if segment is target or segment.cls.casefold() not in {
            "caption", "section-header", "text", "list-item"
        }:
            continue
        gap = target.bbox[1] - segment.bbox[3]
        if not -0.01 <= gap <= 0.065 or horizontal_overlap(target, segment) < 0.35:
            continue
        if (
            looks_like_chart_title(segment.text)
            or (
                segment.cls.casefold() == "caption"
                and looks_like_visual_caption_text(segment.text)
            )
        ):
            candidates.append((segment.bbox[1], segment.text))
    return max(candidates, default=(0.0, ""))[1]


def local_chart_context(target: Segment, segments: Iterable[Segment]) -> str:
    context = [target.text]
    target_width = target.bbox[2] - target.bbox[0]
    for segment in segments:
        if segment is target or segment.cls.casefold() in {"chart", "picture", "table"}:
            continue
        width = segment.bbox[2] - segment.bbox[0]
        if target_width < 0.75 and width > target_width * 1.4:
            continue
        gap = max(target.bbox[1] - segment.bbox[3], segment.bbox[1] - target.bbox[3], 0.0)
        if gap <= 0.10 and horizontal_overlap(target, segment) >= 0.35:
            context.append(segment.text)
    return "\n".join(part for part in context if part).strip()


def chart_target_indexes(segments: list[Segment]) -> tuple[set[int], str]:
    def native_geometry(segment: Segment) -> bool:
        return segment.source in {
            "native_geometry", "native_title_geometry",
            "native_profile_geometry",
        }

    direct = {
        i for i, segment in enumerate(segments)
        if segment.cls.casefold() == "chart"
        and (
            segment.bbox[1] >= 0.14
            or native_geometry(segment)
            or bool(markdown_tables(segment.text))
            or len(VISIBLE_NUMBER_TOKEN.findall(segment.text)) >= 2
            or looks_like_chart_title(segment.text)
        )
        and (
            native_geometry(segment)
            or bool(markdown_tables(segment.text))
            or len(VISIBLE_NUMBER_TOKEN.findall(segment.text)) >= 2
            or looks_like_chart_title(segment.text)
            or bool(nearby_chart_caption(segment, segments))
        )
    }
    contextual_pictures = {
        i
        for i, segment in enumerate(segments)
        if segment.cls.casefold() == "picture"
        and segment.bbox[1] >= 0.14
        and (looks_like_chart_title(segment.text) or nearby_chart_caption(segment, segments))
    }
    chart_like_tables = {
        i
        for i, segment in enumerate(segments)
        if segment.cls.casefold() == "table"
        and not is_metric_card_table(segment)
        and not is_report_data_table(segment)
        and (markdown_tables(segment.text) or "\\begin{tabular}" in segment.text)
        and (looks_like_chart_title(segment.text) or nearby_chart_caption(segment, segments))
    }
    targets = direct | contextual_pictures | chart_like_tables
    if direct:
        return targets, f"{len(direct)} class_Chart block(s)"
    if contextual_pictures:
        return targets, "Picture block(s) with local chart context"
    if chart_like_tables:
        return targets, "Table block(s) with local chart context"
    return set(), "no chart signal"


def match_pass2_charts(
    pass1: list[Segment], target_indexes: set[int], pass2: list[Segment]
) -> dict[int, Segment]:
    candidates = [
        segment
        for segment in pass2
        if segment.cls.casefold() in {"chart", "picture", "table"} and segment.text
    ]
    matches: dict[int, Segment] = {}
    used: set[int] = set()
    for target_index in sorted(target_indexes):
        ranked = sorted(
            (
                (bbox_overlap(pass1[target_index].bbox, candidate.bbox), index, candidate)
                for index, candidate in enumerate(candidates)
                if index not in used
            ),
            reverse=True,
            key=lambda item: item[0],
        )
        if ranked and ranked[0][0] >= 0.20:
            _, candidate_index, candidate = ranked[0]
            matches[target_index] = candidate
            used.add(candidate_index)

    remaining_targets = [index for index in sorted(target_indexes) if index not in matches]
    remaining_candidates = [candidate for i, candidate in enumerate(candidates) if i not in used]
    if remaining_targets and len(remaining_targets) == len(remaining_candidates):
        remaining_targets.sort(key=lambda index: (pass1[index].bbox[1], pass1[index].bbox[0]))
        remaining_candidates.sort(key=lambda segment: (segment.bbox[1], segment.bbox[0]))
        matches.update(zip(remaining_targets, remaining_candidates))
    return matches


def match_chart_ocr_output(
    pass1: list[Segment], target_indexes: set[int], raw: str, source: str,
) -> dict[int, Segment]:
    """Match structured chart blocks and retain a lone unstructured OCR read.

    Nemotron occasionally returns useful chart text without bbox/class closing
    tokens. When the page has exactly one chart target, that raw transcription
    is unambiguous and must not be discarded.
    """
    matches = match_pass2_charts(
        pass1, target_indexes, parse_segments(raw, source)
    )
    if matches or len(target_indexes) != 1:
        return matches
    text = clean_unstructured_output(raw)
    if not text:
        return matches
    target_index = next(iter(target_indexes))
    target = pass1[target_index]
    return {
        target_index: Segment(
            target.order, "Chart", target.bbox, text, f"{source}_raw"
        )
    }


def extract_pdf_text_page(pdf_path: str, page_no: int) -> PdfTextPage:
    xml = subprocess.check_output(
        ["pdftotext", "-bbox-layout", "-f", str(page_no + 1), "-l", str(page_no + 1), pdf_path, "-"],
        text=True,
        stderr=subprocess.PIPE,
    )
    root = ET.fromstring(xml)
    page = next((element for element in root.iter() if element.tag.endswith("page")), None)
    if page is None:
        return PdfTextPage(1.0, 1.0, ())
    width = float(page.attrib.get("width", 1.0))
    height = float(page.attrib.get("height", 1.0))
    lines: list[PdfTextLine] = []
    for element in page.iter():
        if not element.tag.endswith("line"):
            continue
        text = " ".join(
            "".join(child.itertext()).strip()
            for child in element
            if child.tag.endswith("word")
        ).strip()
        try:
            bbox = tuple(float(element.attrib[key]) for key in ("xMin", "yMin", "xMax", "yMax"))
        except (KeyError, ValueError):
            continue
        if text:
            lines.append(PdfTextLine(bbox, text))
    return PdfTextPage(width, height, tuple(sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))))


def pdf_bbox_to_model(
    bbox: tuple[float, float, float, float], page: PdfTextPage
) -> tuple[float, float, float, float]:
    image_width = float(page.render_width or page.width)
    image_height = float(page.render_height or page.height)
    resized_width, resized_height = image_width, image_height
    if resized_height > 2048:
        resized_width *= 2048 / resized_height
        resized_height = 2048.0
    if resized_width > 1664:
        resized_height *= 1664 / resized_width
        resized_width = 1664.0
    pad_x = (1664 - resized_width) / 2
    pad_y = (2048 - resized_height) / 2
    x1, y1, x2, y2 = bbox
    return (
        ((x1 / page.width) * resized_width + pad_x) / 1664,
        ((y1 / page.height) * resized_height + pad_y) / 2048,
        ((x2 / page.width) * resized_width + pad_x) / 1664,
        ((y2 / page.height) * resized_height + pad_y) / 2048,
    )


def native_text_for_segment(segment: Segment, page: PdfTextPage) -> str:
    selected: list[tuple[float, float, str]] = []
    x1, y1, x2, y2 = segment.bbox
    for line in page.lines:
        box = pdf_bbox_to_model(line.bbox, page)
        center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if x1 - 0.004 <= center_x <= x2 + 0.004 and y1 - 0.004 <= center_y <= y2 + 0.004:
            selected.append((box[1], box[0], line.text))
    rows: list[list[tuple[float, float, str]]] = []
    for item in sorted(selected):
        if not rows or abs(item[0] - rows[-1][0][0]) > 0.0015:
            rows.append([item])
        else:
            rows[-1].append(item)
    return "\n".join(
        " ".join(value[2] for value in sorted(row, key=lambda value: value[1])) for row in rows
    ).strip()


def native_chart_value_evidence(
    target: Segment, page: PdfTextPage
) -> tuple[str, str]:
    """Return all native chart text and likely interior data-label text.

    Axis ticks live at the left edge and x-axis categories at the bottom. They
    are useful to establish that a born-digital chart has selectable text, but
    they are not point values and must never verify an OCR-generated cell.
    """
    full: list[str] = []
    interior: list[str] = []
    x1, y1, x2, y2 = target.bbox
    width, height = max(0.001, x2 - x1), max(0.001, y2 - y1)
    for line in page.lines:
        box = pdf_bbox_to_model(line.bbox, page)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        full.append(line.text)
        if (
            x1 + width * 0.12 < cx < x2 - width * 0.03
            and y1 + height * 0.12 < cy < y2 - height * 0.18
            and VISIBLE_NUMBER_TOKEN.search(line.text)
        ):
            interior.append(line.text)
    return "\n".join(full), "\n".join(interior)


def native_endpoint_time_chart_ir(
    target: Segment, page: PdfTextPage
) -> ChartIR | None:
    """Build a time-series skeleton from printed dates and endpoint labels.

    Multi-line performance plots often print no intermediate point values, but
    do print ``series, value%`` beside the last point.  Those endpoint labels
    are direct evidence: serialize them only at the final printed period and
    leave every unlabelled point blank.
    """
    full, _ = native_chart_value_evidence(target, page)
    periods: list[str] = []
    for match in OCR_CHART_TOKEN.finditer(full):
        token = match.group().strip()
        if is_period_label(token) and token not in periods:
            periods.append(token)
    if len(periods) < 2:
        return None

    endpoints: list[tuple[str, str]] = []
    endpoint_pattern = re.compile(
        r"^\s*([^|]+?[A-Za-zÀ-ỹ][^|]*?)\s*[,;:]\s*"
        r"([+\-−–]?\d[\d.,]*%)\s*$",
        re.IGNORECASE,
    )
    for line in full.splitlines():
        match = endpoint_pattern.match(line)
        if not match:
            continue
        series = match.group(1).strip(" -*•\t")
        value = compact_numeric_label(match.group(2))
        if not series or value is None:
            continue
        key = normalize_text(series)
        if key not in {normalize_text(item[0]) for item in endpoints}:
            endpoints.append((series, value))
    if len(endpoints) < 2:
        return None

    final_period = periods[-1]
    return ChartIR(
        "Kỳ",
        tuple(series for series, _ in endpoints),
        tuple(periods),
        tuple(
            ChartObservation(
                final_period, series, value, "native_endpoint_label"
            )
            for series, value in endpoints
        ),
        "native_endpoint_labels",
    )


def native_positioned_time_chart_ir(
    target: Segment, page: PdfTextPage
) -> ChartIR | None:
    """Bind selectable printed values to selectable period labels by x-position."""
    x1, y1, x2, y2 = target.bbox
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None
    labels: list[Segment] = []
    for order, line in enumerate(page.lines):
        box = pdf_bbox_to_model(line.bbox, page)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        normalized_box = (
            (box[0] - x1) / width, (box[1] - y1) / height,
            (box[2] - x1) / width, (box[3] - y1) / height,
        )
        labels.append(Segment(
            order, "Text", normalized_box, line.text,
            "native_positioned_label",
        ))
    chart = spatial_time_chart_ir(labels, target.text)
    return replace(chart, source="native_positioned_time") if chart is not None else None


def native_chart_label_table(target: Segment, page: PdfTextPage) -> str | None:
    """Serialize selectable chart labels as raw fields without inventing mappings."""
    full, _ = native_chart_value_evidence(target, page)
    lines = [line.strip() for line in full.splitlines() if line.strip()]
    if not lines:
        return None
    title_keys = {normalize_text(line) for line in target.text.splitlines() if line.strip()}
    periods: list[str] = []
    numbers: list[str] = []
    labels: list[str] = []
    for line in lines:
        if normalize_text(line) in title_keys or is_source_label(line):
            continue
        tokens = [match.group().strip() for match in OCR_CHART_TOKEN.finditer(line)]
        period_tokens = [token for token in tokens if is_period_label(token)]
        numeric_tokens = [
            token for token in tokens
            if compact_numeric_label(token) is not None and not is_period_label(token)
        ]
        periods.extend(token for token in period_tokens if token not in periods)
        numbers.extend(token for token in numeric_tokens if token not in numbers)
        residual = OCR_CHART_TOKEN.sub(" ", line)
        residual = re.sub(r"\s+", " ", residual).strip(" ·,;|-")
        if re.search(r"[^\W\d_]", residual, re.UNICODE) and residual not in labels:
            labels.append(residual)
    rows = [["Tiêu đề", " / ".join(target.text.splitlines())]]
    if periods:
        rows.append(["Nhãn thời gian/kỳ", "; ".join(periods)])
    if numbers:
        rows.append(["Nhãn số in trên biểu đồ", "; ".join(numbers)])
    if labels:
        rows.append(["Nhãn chuỗi/hạng mục", "; ".join(labels)])
    if len(rows) < 2:
        return None
    return "\n".join([
        markdown_row(["Trường", "Dữ liệu OCR thô"]),
        markdown_row([":---", ":---"]),
        *(markdown_row(row) for row in rows),
    ])


def bind_chart_to_native_value_labels(
    chart: Segment, target: Segment, page: PdfTextPage, context: str
) -> Segment:
    """Mask OCR values that are not printed inside a born-digital plot."""
    if any(
        marker in chart.text for marker in (
            "chart_raw_source=native_endpoint_labels",
            "chart_raw_source=native_positioned_time",
            "chart_raw_source=native_label_table",
        )
    ):
        # This table was already constructed exclusively from the selectable
        # endpoint labels; repeated values belonging to different series are
        # valid and must not be collapsed by fingerprint-based masking.
        return chart
    if not markdown_tables(chart.text):
        return chart
    full, evidence = native_chart_value_evidence(target, page)
    # Compact profile plots are sometimes raster graphics embedded in an
    # otherwise born-digital page.  Their surrounding fiscal/performance rows
    # prove that the plot exists, but an empty selectable plot layer provides
    # no evidence for point values.  Keep its categories/skeleton and withhold
    # every generated value instead of accepting axis ticks as observations.
    if target.source == "native_profile_geometry" and not VISIBLE_NUMBER_TOKEN.search(evidence):
        masked = mask_all_chart_values(chart.text)
        if masked == chart.text:
            return chart
        return replace(
            chart,
            text=masked,
            chart_ir=chart_ir_from_markdown(masked, "native_value_labels", min_data_rows=1),
            source="uncertain:native plot has no printed value labels",
        )
    # Do not constrain scans or charts whose selectable text layer is absent.
    if len(VISIBLE_NUMBER_TOKEN.findall(full)) < 2:
        return chart
    masked = (
        mask_unverified_chart_values(chart.text, evidence)
        if VISIBLE_NUMBER_TOKEN.search(evidence)
        else mask_all_chart_values(chart.text)
    )
    if masked == chart.text:
        return chart
    ir = chart_ir_from_markdown(masked, "native_value_labels", min_data_rows=1)
    return replace(
        chart, text=masked, chart_ir=ir,
        source="uncertain:native plot-label evidence",
    )


def text_token_overlap(first: str, second: str) -> float:
    tokens = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?", re.IGNORECASE)
    left = set(tokens.findall(fold_text(first)))
    right = set(tokens.findall(fold_text(second)))
    return len(left & right) / min(len(left), len(right)) if left and right else 0.0


def reconcile_with_text_layer(
    segments: list[Segment], page: PdfTextPage
) -> tuple[list[Segment], int]:
    """Use exact native text for prose only; chart/table values remain OCR evidence."""
    eligible = {
        "text", "list-item", "title", "section-header", "caption", "footnote",
        "page-header", "page-footer",
    }
    result: list[Segment] = []
    replacements = 0
    for segment in segments:
        if segment.cls.casefold() not in eligible:
            result.append(segment)
            continue
        native = native_text_for_segment(segment, page)
        similarity = difflib.SequenceMatcher(None, normalize_text(segment.text), normalize_text(native)).ratio()
        if native and (text_token_overlap(segment.text, native) >= 0.55 or similarity >= 0.62):
            result.append(replace(segment, text=native, source="text_layer"))
            replacements += 1
        else:
            result.append(segment)
    return result, replacements


def recover_missing_native_text(
    segments: list[Segment], page: PdfTextPage
) -> tuple[list[Segment], int]:
    """Recover native prose omitted on mixed pages without entering visuals."""
    existing_text = normalize_text("\n".join(segment.text for segment in segments))
    visuals = [
        segment for segment in segments
        if segment.cls.casefold() in {"chart", "picture", "table"}
    ]
    missing: list[PdfTextLine] = []
    for line in page.lines:
        normalized = normalize_text(line.text)
        line_tokens = re.findall(r"[a-z0-9]+", normalized)
        shingles = [" ".join(line_tokens[index:index + 4]) for index in range(max(0, len(line_tokens) - 3))]
        fuzzy_present = (
            bool(shingles)
            and sum(shingle in existing_text for shingle in shingles) / len(shingles) >= 0.80
        )
        if len(normalized) < 2 or normalized in existing_text or fuzzy_present:
            continue
        box = pdf_bbox_to_model(line.bbox, page)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        covering_visuals = [
            visual for visual in visuals
            if visual.bbox[0] - 0.003 <= cx <= visual.bbox[2] + 0.003
            and visual.bbox[1] - 0.003 <= cy <= visual.bbox[3] + 0.003
        ]
        prose_like = (
            len(line_tokens) >= 8
            or bool(re.match(r"^\d+(?:\.\d+)+\.?\s+", normalized))
        )
        if covering_visuals and (
            any(
                visual.cls.casefold() == "table"
                and visual.source.startswith("text_layer")
                for visual in covering_visuals
            )
            or not prose_like
        ):
            continue
        covered = False
        probe = Segment(0, "Text", box, line.text)
        if not prose_like:
            for segment in segments:
                if segment.cls.casefold() in {"chart", "picture", "table"}:
                    continue
                if bbox_overlap(segment.bbox, probe.bbox) >= 0.50:
                    covered = True
                    break
        if not covered:
            missing.append(line)
    if not missing:
        return segments, 0

    groups: list[list[PdfTextLine]] = []
    for line in sorted(missing, key=lambda item: (item.bbox[1], item.bbox[0])):
        if not groups:
            groups.append([line])
            continue
        previous = groups[-1][-1]
        line_height = max(1.0, previous.bbox[3] - previous.bbox[1])
        vertical_gap = line.bbox[1] - previous.bbox[3]
        same_column = abs(line.bbox[0] - previous.bbox[0]) <= page.width * 0.08
        if same_column and -1.0 <= vertical_gap <= line_height * 1.4:
            groups[-1].append(line)
        else:
            groups.append([line])

    recovered: list[Segment] = []
    base_order = max((segment.order for segment in segments), default=0) + 1
    for offset, group in enumerate(groups):
        pdf_box = (
            min(line.bbox[0] for line in group), min(line.bbox[1] for line in group),
            max(line.bbox[2] for line in group), max(line.bbox[3] for line in group),
        )
        text = " ".join(line.text.strip() for line in group if line.text.strip())
        if text:
            recovered.append(
                Segment(
                    base_order + offset, "Text", pdf_bbox_to_model(pdf_box, page),
                    text, "text_layer_recovered",
                )
            )
    return segments + recovered, len(recovered)


def native_text_coverage(segments: list[Segment], page: PdfTextPage) -> float:
    """Token recall quality gate for the page's native PDF text layer."""
    visual_boxes = [
        segment.bbox for segment in segments
        if segment.cls.casefold() in {"chart", "picture"}
    ]
    native_lines: list[str] = []
    for line in page.lines:
        box = pdf_bbox_to_model(line.bbox, page)
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        if any(
            visual[0] <= center[0] <= visual[2]
            and visual[1] <= center[1] <= visual[3]
            for visual in visual_boxes
        ):
            continue
        native_lines.append(line.text)
    native = Counter(re.findall(
        r"[a-z0-9]+", fold_text(" ".join(native_lines))
    ))
    output = Counter(re.findall(
        r"[a-z0-9]+", fold_text(" ".join(segment.text for segment in segments))
    ))
    if not native:
        return 1.0
    return sum(min(count, output[token]) for token, count in native.items()) / sum(native.values())


def native_table_for_segment(segment: Segment, page: PdfTextPage) -> str | None:
    """Render an exact native-text table inside an OCR table bbox.

    The text layer supplies cell contents only; the OCR bbox supplies table
    ownership. No missing value is calculated or copied from another region.
    """
    selected: list[tuple[float, float, str]] = []
    x1, y1, x2, y2 = segment.bbox
    for line in page.lines:
        box = pdf_bbox_to_model(line.bbox, page)
        center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if x1 - 0.003 <= center_x <= x2 + 0.003 and y1 - 0.003 <= center_y <= y2 + 0.003:
            selected.append((box[1], box[0], line.text.strip()))
    rows: list[list[tuple[float, float, str]]] = []
    for item in sorted(selected):
        if not rows or abs(item[0] - rows[-1][0][0]) > 0.0018:
            rows.append([item])
        else:
            rows[-1].append(item)
    cell_rows = [
        [item[2] for item in sorted(row, key=lambda value: value[1]) if item[2]]
        for row in rows
    ]
    widths = Counter(len(row) for row in cell_rows if len(row) >= 2)
    if not widths:
        return None
    width = max(widths, key=lambda value: (widths[value], value))
    runs: list[list[list[str]]] = []
    current: list[list[str]] = []
    for row in cell_rows:
        if len(row) == width:
            current.append(row)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []
    if len(current) >= 2:
        runs.append(current)
    if not runs:
        return None
    run = dedupe_consecutive_rows(max(runs, key=len))
    if not looks_like_table_header(run[0]):
        return None
    rendered = [markdown_row(run[0]), markdown_row([":---"] + ["---:"] * (width - 1))]
    rendered.extend(markdown_row(row) for row in run[1:])
    return "\n".join(rendered)


def robust_native_report_table(markdown: str | None) -> bool:
    """Distinguish a real report table from two chart axes aligned by chance."""
    if not markdown:
        return False
    tables = markdown_tables(markdown)
    if not tables:
        return False
    rows = tables[0]
    separator = next((
        index for index, row in enumerate(rows)
        if row and all(TABLE_SEPARATOR.fullmatch(cell) for cell in row)
    ), None)
    if separator is None or separator == 0:
        return False
    width = len(rows[separator - 1])
    data_rows = len(rows) - separator - 1
    return width >= 3 or data_rows >= 2


def extract_native_table_candidates(page: PdfTextPage) -> list[str]:
    """Extract complete rectangular tables directly from a page text layer."""
    native_segments = native_wide_table_segments(page)
    rows: list[list[PdfTextLine]] = []
    for line in page.lines:
        if not rows or abs(line.bbox[1] - rows[-1][0].bbox[1]) > 2.5:
            rows.append([line])
        else:
            rows[-1].append(line)
    cells = [
        [line.text.strip() for line in sorted(row, key=lambda value: value.bbox[0]) if line.text.strip()]
        for row in rows
    ]
    candidates: list[str] = [segment.text for segment in native_segments]
    index = 0
    while index < len(cells):
        if len(cells[index]) < 3:
            index += 1
            continue
        width = len(cells[index])
        end = index
        while end < len(cells) and len(cells[end]) == width:
            end += 1
        run = dedupe_consecutive_rows(cells[index:end])
        if len(run) >= 2 and looks_like_table_header(run[0]):
            rendered = "\n".join(
                [markdown_row(run[0]), markdown_row([":---"] + ["---:"] * (width - 1))]
                + [markdown_row(row) for row in run[1:]]
            )
            if normalize_text(rendered) not in {normalize_text(item) for item in candidates}:
                candidates.append(rendered)
        index = max(end, index + 1)
    return candidates


def native_wide_table_segments(page: PdfTextPage) -> list[Segment]:
    """Recover complete wide tables using header-column anchors and geometry.

    Empty cells are retained. Collection stops at a visual title, source/footer,
    or a large vertical gap so a table cannot absorb the chart below it.
    """
    rows: list[list[PdfTextLine]] = []
    for line in sorted(page.lines, key=lambda item: (item.bbox[1], item.bbox[0])):
        if not rows or abs(line.bbox[1] - rows[-1][0].bbox[1]) > 2.5:
            rows.append([line])
        else:
            rows[-1].append(line)

    result: list[Segment] = []
    for header_index, row in enumerate(rows):
        header = sorted(row, key=lambda value: value.bbox[0])
        if len(header) < 3:
            continue
        header_cells = [line.text.strip() for line in header]
        if (
            not looks_like_table_header(header_cells)
            or header[-1].bbox[2] - header[0].bbox[0] < page.width * 0.35
        ):
            continue
        anchors = [(line.bbox[0] + line.bbox[2]) / 2 for line in header]
        later_gaps = [
            anchors[index + 1] - anchors[index]
            for index in range(1, len(anchors) - 1)
        ]
        if later_gaps:
            typical_later_gap = sorted(later_gaps)[len(later_gaps) // 2]
            if anchors[1] - anchors[0] < typical_later_gap * 1.15:
                continue
        width = len(anchors)
        rendered_rows = [header_cells]
        owned = list(header)
        previous_y = header[0].bbox[1]
        for raw_candidate in rows[header_index + 1:]:
            candidate = sorted(raw_candidate, key=lambda value: value.bbox[0])
            if not candidate:
                continue
            text = " ".join(line.text.strip() for line in candidate if line.text.strip())
            folded = fold_text(text)
            gap = candidate[0].bbox[1] - previous_y
            sparse_row = len(candidate) <= max(2, width // 2)
            sparse_nonnumeric = sparse_row and all(
                compact_numeric_label(line.text.strip()) is None
                for line in candidate
            )
            if (
                gap > 28
                or folded.startswith(("nguon:", "source:", "www."))
                or (sparse_row and looks_like_chart_title(text))
                or (sparse_row and CHART_SUBTITLE.search(text))
                or (len(rendered_rows) >= 3 and sparse_nonnumeric)
            ):
                break
            if len(candidate) < 2 or candidate[0].bbox[0] > page.width * 0.28:
                if gap <= 5:
                    continue
                break
            cells = [""] * width
            assigned = 0
            for line in candidate:
                center = (line.bbox[0] + line.bbox[2]) / 2
                column = min(range(width), key=lambda index: abs(anchors[index] - center))
                # Reject prose spanning several numeric columns.
                if column > 0 and abs(anchors[column] - center) > page.width * 0.08:
                    continue
                cells[column] = " ".join(
                    part for part in (cells[column], line.text.strip()) if part
                )
                assigned += 1
            if not cells[0] or assigned < 2:
                break
            if (
                len(rendered_rows) == 1
                and not re.search(r"[^\W\d_]", cells[0], re.UNICODE)
            ):
                break
            rendered_rows.append(cells)
            owned.extend(candidate)
            previous_y = candidate[0].bbox[1]
        if len(rendered_rows) < 3:
            continue
        markdown = "\n".join(
            [markdown_row(rendered_rows[0]), markdown_row([":---"] + ["---:"] * (width - 1))]
            + [markdown_row(row) for row in rendered_rows[1:]]
        )
        native_bbox = (
            min(line.bbox[0] for line in owned),
            min(line.bbox[1] for line in owned),
            max(line.bbox[2] for line in owned),
            max(line.bbox[3] for line in owned),
        )
        result.append(Segment(
            header_index, "Table", pdf_bbox_to_model(native_bbox, page),
            markdown, "text_layer_wide_table",
        ))
    return result


def coalesce_native_wide_tables(
    segments: list[Segment], page: PdfTextPage
) -> tuple[list[Segment], int]:
    """Replace fragmented OCR table output with complete native PDF tables."""
    tables = native_wide_table_segments(page)
    if not tables:
        return segments, 0
    removed: set[int] = set()
    accepted: list[Segment] = []
    eligible = {
        "text", "caption", "table", "list-item", "title", "section-header",
        "chart", "picture",
    }
    for table in tables:
        table_key = normalize_text(table.text)
        owners = {
            index for index, segment in enumerate(segments)
            if index not in removed
            and segment.cls.casefold() in eligible
            and (
                bbox_overlap(segment.bbox, table.bbox) >= 0.35
                or (
                    normalize_text(segment.text)
                    and normalize_text(segment.text) in table_key
                    and max(
                        table.bbox[1] - segment.bbox[3],
                        segment.bbox[1] - table.bbox[3],
                        0.0,
                    ) <= 0.01
                )
            )
        }
        # The native header is sufficient ownership evidence even when OCR
        # emitted the table as many tiny cells instead of a Table segment.
        if not owners and not any(
            normalize_text(split_markdown_row(table.text.splitlines()[0])[0])
            in normalize_text(segment.text)
            for segment in segments
        ):
            continue
        removed.update(owners)
        order = min((segments[index].order for index in owners), default=table.order)
        accepted.append(replace(table, order=order))
    if not accepted:
        return segments, 0
    result = [segment for index, segment in enumerate(segments) if index not in removed]
    result.extend(accepted)
    result.sort(key=lambda segment: (round(segment.bbox[1], 3), segment.bbox[0], segment.order))
    return result, len(accepted)


def reconcile_tables_with_text_layer(
    segments: list[Segment], page: PdfTextPage
) -> tuple[list[Segment], int]:
    """Replace OCR tables with exact, position-matched native table cells."""
    result: list[Segment] = []
    replacements = 0
    for segment in segments:
        if segment.cls.casefold() != "table":
            result.append(segment)
            continue
        native = native_table_for_segment(segment, page)
        if native is None:
            result.append(segment)
            continue
        ocr_markdown = simple_latex_table_to_markdown(segment.text)
        ocr_tables = markdown_tables(ocr_markdown)
        native_tables = markdown_tables(native)
        # Aligned prose columns can accidentally look rectangular in the PDF
        # text layer.  They cannot replace an OCR table with a different schema
        # even when a few numbers happen to overlap.
        if (
            not ocr_tables or not native_tables
            or not ocr_tables[0] or not native_tables[0]
            or len(ocr_tables[0][0]) != len(native_tables[0][0])
        ):
            result.append(segment)
            continue
        ocr_numbers = numeric_signature(ocr_markdown)
        native_numbers = numeric_signature(native)
        common = sum((ocr_numbers & native_numbers).values())
        denominator = max(1, min(sum(ocr_numbers.values()), sum(native_numbers.values())))
        if ocr_numbers and common / denominator < 0.65:
            result.append(segment)
            continue
        result.append(replace(segment, text=native, source="text_layer_table"))
        replacements += 1
    return result, replacements


def restore_richer_original_tables(
    segments: list[Segment], originals: list[Segment]
) -> tuple[list[Segment], int]:
    """Prevent a partial native/layout repair from shortening an OCR table."""
    result = list(segments)
    restored = 0

    def row_count(segment: Segment) -> int:
        markdown = simple_latex_table_to_markdown(segment.text)
        tables = markdown_tables(markdown)
        if not tables:
            return 0
        rows = tables[0]
        separator = next((
            index for index, row in enumerate(rows)
            if row and all(TABLE_SEPARATOR.fullmatch(cell) for cell in row)
        ), None)
        return max(0, len(rows) - separator - 1) if separator is not None else 0

    for original in originals:
        original_rows = row_count(original)
        if not 2 <= original_rows <= 50 or is_degenerate_ocr(original.text):
            continue
        matches = [
            (bbox_overlap(original.bbox, current.bbox), index, current)
            for index, current in enumerate(result)
            if current.cls.casefold() == "table"
        ]
        if not matches:
            continue
        overlap, index, current = max(matches, key=lambda item: item[0])
        if overlap >= 0.25 and row_count(current) < original_rows:
            result[index] = replace(
                original, order=current.order, bbox=current.bbox,
                source="restored_richer_ocr_table",
            )
            restored += 1
    return result, restored


def recover_misclassified_visuals(
    segments: list[Segment], page: PdfTextPage
) -> tuple[list[Segment], int]:
    """Recover tables/infographics that the model incorrectly labels Chart.

    Exact same-region PDF text is used. Real charts are retained whenever a
    local chart title exists or the region is primarily graphical labels.
    """
    result: list[Segment] = []
    replacements = 0
    for segment in segments:
        if segment.cls.casefold() != "chart":
            result.append(segment)
            continue
        as_table = native_table_for_segment(replace(segment, cls="Table"), page)
        if as_table is not None:
            ocr_numbers = numeric_signature(simple_latex_table_to_markdown(segment.text))
            native_numbers = numeric_signature(as_table)
            common = sum((ocr_numbers & native_numbers).values())
            denominator = max(1, min(sum(ocr_numbers.values()), sum(native_numbers.values())))
            if not ocr_numbers or common / denominator >= 0.65:
                result.append(
                    replace(segment, cls="Table", text=as_table, source="text_layer_table")
                )
                replacements += 1
                continue
        # A chart table that is not a native rectangular report table remains
        # a chart candidate even when one OCR read has quality issues. Later
        # evidence checks will mask or reject values; prose density must not
        # erase the visual itself.
        if markdown_tables(segment.text):
            result.append(segment)
            continue
        native = native_text_for_segment(segment, page)
        word_count = len(re.findall(r"[^\W\d_]{2,}", native, re.UNICODE))
        longest_line = max(
            (len(re.findall(r"[^\W\d_]{2,}", line, re.UNICODE)) for line in native.splitlines()),
            default=0,
        )
        if native and word_count >= 12 and longest_line >= 8:
            result.append(replace(segment, cls="Text", text=native, source="text_layer_infographic"))
            replacements += 1
            continue
        result.append(segment)
    return result, replacements


def infer_native_chart_segments(
    segments: list[Segment], page: PdfTextPage
) -> tuple[list[Segment], int]:
    """Recover omitted charts from title-to-source geometry in the PDF layer.

    Many documents place a short, larger title above a plot and a narrow source
    line below it. This identifies visual geometry only; it does not infer data.
    """
    inferred: list[Segment] = []
    midpoint = page.width / 2
    sources = [
        line for line in page.lines
        if is_source_label(line.text)
        and line.bbox[2] - line.bbox[0] <= page.width * 0.44
    ]
    for source in sources:
        left_column = (source.bbox[0] + source.bbox[2]) / 2 < midpoint
        column = (
            (page.width * 0.04, page.width * 0.49)
            if left_column else (page.width * 0.51, page.width * 0.96)
        )
        candidates = []
        for line in page.lines:
            height = line.bbox[3] - line.bbox[1]
            center = (line.bbox[0] + line.bbox[2]) / 2
            gap = source.bbox[1] - line.bbox[3]
            if not (
                column[0] <= center <= column[1]
                and 35 <= gap <= 165
                and height >= 11.0
                and len(line.text) <= 100
                and not is_source_label(line.text)
                and not NON_CHART_HEADING.search(fold_text(line.text))
                and looks_like_visual_caption_text(line.text)
            ):
                continue
            # Exclude prose spanning both chart columns.
            if line.bbox[0] < midpoint < line.bbox[2]:
                continue
            candidates.append(line)
        if not candidates:
            continue
        # Prefer the largest text line: this rejects a short prose tail that
        # happens to sit above the visual title in the same column.
        largest_height = max(line.bbox[3] - line.bbox[1] for line in candidates)
        title = min(
            (
                line for line in candidates
                if line.bbox[3] - line.bbox[1] >= largest_height - 0.5
            ),
            key=lambda line: line.bbox[1],
        )
        subtitle_candidates = [
            line for line in page.lines
            if title.bbox[3] < line.bbox[1] <= title.bbox[3] + 35
            and column[0] <= (line.bbox[0] + line.bbox[2]) / 2 <= column[1]
            and line.bbox[3] - line.bbox[1] < title.bbox[3] - title.bbox[1]
            and not is_source_label(line.text)
        ]
        caption = title.text.strip()
        if subtitle_candidates:
            caption += "\n" + min(subtitle_candidates, key=lambda line: line.bbox[1]).text.strip()
        native_bbox = (
            column[0],
            max(0.0, title.bbox[1] - 3.0),
            column[1],
            min(page.height, source.bbox[1] - 3.0),
        )
        model_bbox = pdf_bbox_to_model(native_bbox, page)
        candidate_segment = Segment(
            len(segments) + len(inferred), "Chart", model_bbox,
            title.text.strip(), "native_title_geometry",
        )
        if robust_native_report_table(
            native_table_for_segment(replace(candidate_segment, cls="Table"), page)
        ):
            continue
        if any(
            segment.cls.casefold() in {"chart", "picture", "table"}
            and bbox_overlap(segment.bbox, model_bbox) >= 0.25
            for segment in segments + inferred
        ):
            continue
        inferred.append(
            Segment(
                order=len(segments) + len(inferred),
                cls="Chart",
                bbox=model_bbox,
                text=caption,
                source="native_geometry",
            )
        )
    if not inferred:
        return segments, 0
    return segments + inferred, len(inferred)


def infer_native_chart_segments_from_titles(
    segments: list[Segment], page: PdfTextPage
) -> tuple[list[Segment], int]:
    """Recover plot rectangles from title rows followed by source rows.

    Report pages commonly place two charts side by side and one shared source
    below them.  Associating every short line with every later source promotes
    dates, table headers and prose fragments to charts.  Instead, source lines
    define a visual band; the strongest sparse title row defines its columns.
    """
    sources = sorted(
        (line for line in page.lines if is_source_label(line.text)),
        key=lambda line: (line.bbox[1], line.bbox[0]),
    )
    if not sources:
        return segments, 0

    source_rows: list[list[PdfTextLine]] = []
    for source in sources:
        if not source_rows or abs(source.bbox[1] - source_rows[-1][0].bbox[1]) > 15.0:
            source_rows.append([source])
        else:
            source_rows[-1].append(source)

    generic = {
        "chi tieu", "ket qua", "ghi chu", "nguon", "source", "so luong",
        "year", "date", "linh", "vuc", "ma", "san", "dtt", "lnst",
    }

    def title_candidate(line: PdfTextLine) -> bool:
        text = line.text.strip()
        folded = fold_text(text).strip()
        height = line.bbox[3] - line.bbox[1]
        words = re.findall(r"[^\W\d_]+", text, re.UNICODE)
        return bool(
            8.5 <= height <= 16.0
            and 2 <= len(words) <= 20
            and len(text) <= 110
            and folded not in generic
            and not is_source_label(text)
            and not is_period_label(text)
            and compact_numeric_label(text) is None
            and not text.startswith("(")
            and not re.match(r"^[+\-•▪]\s*|^\d+(?:[.,)]|\s)", text)
            and not re.search(r"[.!?;:]\s*$", text)
            and not NON_CHART_HEADING.search(folded)
            and looks_like_visual_caption_text(text)
        )

    inferred: list[Segment] = []
    for source_row in source_rows:
        source_y = min(line.bbox[1] for line in source_row)
        pool = [
            line for line in page.lines
            if 70.0 <= source_y - line.bbox[3] <= 220.0
            and title_candidate(line)
        ]
        # Do not treat the second line of a wrapped chart title as another
        # neighbouring chart.  Its first line owns the caption and column.
        pool = [
            line for line in pool
            if not any(
                0.0 < line.bbox[1] - previous.bbox[1] <= 16.0
                and max(
                    0.0,
                    min(line.bbox[2], previous.bbox[2])
                    - max(line.bbox[0], previous.bbox[0]),
                ) / max(1.0, min(
                    line.bbox[2] - line.bbox[0],
                    previous.bbox[2] - previous.bbox[0],
                )) >= 0.55
                for previous in pool if previous is not line
            )
        ]
        rows: list[list[PdfTextLine]] = []
        for line in sorted(pool, key=lambda item: (item.bbox[1], item.bbox[0])):
            if not rows or abs(line.bbox[1] - rows[-1][0].bbox[1]) > 8.0:
                rows.append([line])
            else:
                rows[-1].append(line)

        ranked: list[tuple[int, float, list[PdfTextLine]]] = []
        for row in rows:
            row_y = sum(line.bbox[1] for line in row) / len(row)
            all_on_row = [
                line for line in page.lines if abs(line.bbox[1] - row_y) <= 3.0
            ]
            # Four or more aligned cells are overwhelmingly a report table,
            # not a row of chart captions.
            if len(all_on_row) > 3 or len(row) > 2:
                continue
            distinct = sorted(row, key=lambda line: line.bbox[0])
            ranked.append((len(distinct), row_y, distinct))
        if not ranked:
            continue
        # Prefer paired/triple chart title rows, then the lowest plausible row
        # above the plot.  Prose tails are already rejected by punctuation.
        titles = max(ranked, key=lambda item: (item[1], item[0]))[2]

        centers = [(line.bbox[0] + line.bbox[2]) / 2 for line in titles]
        for title_index, title in enumerate(titles):
            if len(titles) == 1:
                center = centers[0] / page.width
                if 0.34 <= center <= 0.66:
                    left, right = page.width * 0.03, page.width * 0.97
                elif center < 0.5:
                    left, right = page.width * 0.03, page.width * 0.50
                else:
                    left, right = page.width * 0.50, page.width * 0.97
            else:
                left = (
                    page.width * 0.03 if title_index == 0
                    else (centers[title_index - 1] + centers[title_index]) / 2
                )
                right = (
                    page.width * 0.97 if title_index == len(titles) - 1
                    else (centers[title_index] + centers[title_index + 1]) / 2
                )
            continuation = [
                line for line in page.lines
                if title.bbox[3] < line.bbox[1] <= title.bbox[3] + 16.0
                and max(0.0, min(title.bbox[2], line.bbox[2]) - max(title.bbox[0], line.bbox[0]))
                / max(1.0, title.bbox[2] - title.bbox[0]) >= 0.45
                and (
                    title_candidate(line)
                    or (
                        len(line.text.strip()) <= 80
                        and not is_source_label(line.text)
                        and compact_numeric_label(line.text) is None
                        and len([
                            token for token in OCR_CHART_TOKEN.findall(line.text)
                            if is_period_label(token)
                        ]) >= 2
                    )
                )
            ]
            caption = title.text.strip()
            if continuation:
                caption += "\n" + min(continuation, key=lambda line: line.bbox[1]).text.strip()
            native_bbox = (
                max(0.0, left), max(0.0, title.bbox[1] - 4.0),
                min(page.width, right), min(page.height, source_y - 3.0),
            )
            model_bbox = pdf_bbox_to_model(native_bbox, page)
            candidate = Segment(
                len(segments) + len(inferred), "Chart", model_bbox,
                caption, "native_title_geometry",
            )
            native = native_text_for_segment(candidate, page)
            numeric = [
                token for token in VISIBLE_NUMBER_TOKEN.findall(native)
                if not is_period_label(token)
            ]
            periods = OCR_CHART_TOKEN.findall(native)
            if "→" in native or (
                len(titles) == 1 and len(numeric) < 2 and len(periods) < 3
            ):
                continue
            if robust_native_report_table(
                native_table_for_segment(replace(candidate, cls="Table"), page)
            ):
                continue
            inferred.append(candidate)

    if not inferred:
        return segments, 0

    # Source/title-derived geometry owns the visual rectangle.  Discard an
    # overlapping layout-model Chart/Picture placeholder; it may span one or
    # several neighbouring plots and often carries prose instead of a title.
    split_existing: set[int] = set()
    for index, item in enumerate(segments):
        if item.cls.casefold() not in {"chart", "picture"}:
            continue
        owned = [
            candidate for candidate in inferred
            if (
                bbox_overlap(item.bbox, candidate.bbox) >= 0.10
                or (
                    horizontal_overlap(item, candidate) >= 0.35
                    and max(
                        candidate.bbox[1] - item.bbox[3],
                        item.bbox[1] - candidate.bbox[3],
                        0.0,
                    ) <= 0.035
                )
            )
        ]
        if owned:
            split_existing.add(index)

    retained = [item for index, item in enumerate(segments) if index not in split_existing]
    accepted: list[Segment] = []
    for candidate in inferred:
        if any(
            item.cls.casefold() in {"chart", "picture"}
            and bbox_overlap(item.bbox, candidate.bbox) >= 0.15
            for item in [*retained, *accepted]
        ):
            continue
        accepted.append(candidate)
    if not accepted:
        return segments, 0
    return [*retained, *accepted], len(accepted)


def infer_native_profile_chart_segment(
    segments: list[Segment], page: PdfTextPage
) -> tuple[list[Segment], int]:
    """Recover the performance plot in a compact equity-profile layout."""
    rows: list[list[PdfTextLine]] = []
    for line in page.lines:
        if not rows or abs(line.bbox[1] - rows[-1][0].bbox[1]) > 1.5:
            rows.append([line])
        else:
            rows[-1].append(line)
    fy_row = next((
        row for row in rows
        if sum(bool(re.fullmatch(r"FY\s*20\d{2}(?:\s*\(F\))?", line.text, re.I))
               for line in row) >= 5
    ), None)
    performance_row = next((
        row for row in rows
        if {normalize_text(line.text) for line in row} >= {"1 m", "3m", "12m"}
    ), None)
    if fy_row is None or performance_row is None:
        return segments, 0
    fiscal_headers = [
        line for line in fy_row
        if re.fullmatch(r"FY\s*20\d{2}(?:\s*\(F\))?", line.text, re.I)
    ]
    first_fy_x = min(line.bbox[0] for line in fiscal_headers)
    y1 = min(line.bbox[1] for line in fy_row) - 8.0
    y2 = min(page.height * 0.965, max(
        (line.bbox[3] for line in page.lines if line.bbox[1] >= y1),
        default=page.height * 0.90,
    ))
    native_bbox = (
        page.width * 0.27, max(0.0, y1),
        min(first_fy_x - 12.0, page.width * 0.56), y2,
    )
    model_bbox = pdf_bbox_to_model(native_bbox, page)
    if any(
        item.cls.casefold() in {"chart", "picture", "table"}
        and bbox_overlap(item.bbox, model_bbox) >= 0.20
        for item in segments
    ):
        return segments, 0
    candidate = Segment(
        len(segments), "Chart", model_bbox, "Hiệu suất giá tương đối",
        "native_profile_geometry",
    )
    return [*segments, candidate], 1


def native_metric_card_segments(page: PdfTextPage) -> list[Segment]:
    """Find repeated horizontal KPI cards from exact positioned text."""
    rows: list[list[PdfTextLine]] = []
    for line in page.lines:
        if not rows or abs(line.bbox[1] - rows[-1][0].bbox[1]) > 1.2:
            rows.append([line])
        else:
            rows[-1].append(line)
    result: list[Segment] = []
    index = 0
    while index + 1 < len(rows):
        header = sorted(rows[index], key=lambda line: line.bbox[0])
        values = sorted(rows[index + 1], key=lambda line: line.bbox[0])
        width = len(header)
        if not (
            3 <= width <= 8
            and len(values) == width
            and all(re.search(r"[A-Za-zÀ-ỹ]", line.text) for line in header)
            and all(line.text == line.text.upper() for line in header)
            and all(re.search(r"\d", line.text) for line in values)
            and max(line.bbox[1] for line in values) - min(line.bbox[1] for line in header) <= 30
        ):
            index += 1
            continue
        details: list[PdfTextLine] = []
        if index + 2 < len(rows):
            possible = sorted(rows[index + 2], key=lambda line: line.bbox[0])
            if (
                len(possible) == width
                and max(line.bbox[1] for line in possible) - min(line.bbox[1] for line in values) <= 30
            ):
                details = possible
        detail_values = [line.text for line in details]
        consumed_rows = 3 if details else 2
        if details and index + 3 < len(rows):
            continuation = sorted(rows[index + 3], key=lambda line: line.bbox[0])
            vertical_gap = min(line.bbox[1] for line in continuation) - max(
                line.bbox[3] for line in details
            )
            column_centers = [(line.bbox[0] + line.bbox[2]) / 2 for line in header]
            assignments: list[tuple[int, PdfTextLine]] = []
            for line in continuation:
                center = (line.bbox[0] + line.bbox[2]) / 2
                column = min(range(width), key=lambda value: abs(column_centers[value] - center))
                column_span = page.width / width
                if abs(column_centers[column] - center) <= column_span * 0.45:
                    assignments.append((column, line))
            if 0 <= vertical_gap <= 8 and len(assignments) == len(continuation):
                for column, line in assignments:
                    detail_values[column] = f"{detail_values[column]} {line.text}".strip()
                consumed_rows += 1
                details.extend(continuation)
        table_rows = [
            [header[column].text, values[column].text]
            + ([detail_values[column]] if details else [])
            for column in range(width)
        ]
        columns = ["Chỉ tiêu", "Giá trị"] + (["Chi tiết"] if details else [])
        markdown = "\n".join(
            [markdown_row(columns), markdown_row([":---", "---:"] + ([":---"] if details else []))]
            + [markdown_row(row) for row in table_rows]
        )
        owned = header + values + details
        native_bbox = (
            min(line.bbox[0] for line in owned),
            min(line.bbox[1] for line in owned),
            max(line.bbox[2] for line in owned),
            max(line.bbox[3] for line in owned),
        )
        result.append(
            Segment(
                order=index,
                cls="Table",
                bbox=pdf_bbox_to_model(native_bbox, page),
                text=markdown,
                source="text_layer_metric_cards",
            )
        )
        index += consumed_rows
    if len(result) < 2:
        return result
    merged: list[Segment] = []
    for card in result:
        if not merged:
            merged.append(card)
            continue
        previous = merged[-1]
        x_overlap = max(
            0.0, min(previous.bbox[2], card.bbox[2])
            - max(previous.bbox[0], card.bbox[0])
        )
        x_denominator = min(
            previous.bbox[2] - previous.bbox[0], card.bbox[2] - card.bbox[0]
        )
        vertical_gap = card.bbox[1] - previous.bbox[3]
        previous_lines = previous.text.splitlines()
        card_lines = card.text.splitlines()
        same_schema = previous_lines[:2] == card_lines[:2]
        if (
            same_schema and x_denominator > 0 and x_overlap / x_denominator >= 0.85
            and 0 <= vertical_gap <= 0.025
        ):
            merged[-1] = replace(
                previous,
                bbox=(
                    min(previous.bbox[0], card.bbox[0]), previous.bbox[1],
                    max(previous.bbox[2], card.bbox[2]), card.bbox[3],
                ),
                text="\n".join(previous_lines + card_lines[2:]),
            )
        else:
            merged.append(card)
    return merged


def coalesce_metric_cards(
    segments: list[Segment], page: PdfTextPage
) -> tuple[list[Segment], int]:
    """Replace scattered OCR card fragments with exact native card tables."""
    cards = native_metric_card_segments(page)
    if not cards:
        return segments, 0
    removed: set[int] = set()
    accepted: list[Segment] = []
    # Models occasionally label a compact KPI card as Chart/Picture. Exact
    # native card geometry takes ownership of that small overlapping block.
    eligible = {
        "text", "caption", "table", "list-item", "chart", "picture",
        "title", "section-header",
    }
    for card in cards:
        card_key = normalize_text(card.text)
        owners = {
            index
            for index, segment in enumerate(segments)
            if segment.cls.casefold() in eligible
            and (
                bbox_overlap(segment.bbox, card.bbox) >= 0.35
                or (
                    0 < len(normalize_text(segment.text)) <= 50
                    and normalize_text(segment.text) in card_key
                    and max(
                        card.bbox[1] - segment.bbox[3],
                        segment.bbox[1] - card.bbox[3],
                        0.0,
                    ) <= 0.02
                )
            )
        }
        removed.update(owners)
        order = min((segments[index].order for index in owners), default=card.order)
        accepted.append(replace(card, order=order))
    if not accepted:
        return segments, 0
    result = [segment for index, segment in enumerate(segments) if index not in removed]
    result.extend(accepted)
    result.sort(key=lambda segment: (round(segment.bbox[1], 3), segment.bbox[0], segment.order))
    return result, len(accepted)


def use_full_native_text_if_ocr_truncated(
    segments: list[Segment], page: PdfTextPage
) -> tuple[list[Segment], bool]:
    """Recover text-only pages when generation ends early; never rebuild figures."""
    if any(segment.cls.casefold() in {"chart", "picture", "table"} for segment in segments):
        return segments, False
    native = "\n".join(line.text for line in page.lines).strip()
    ocr = "\n".join(segment.text for segment in segments).strip()
    if not native:
        return segments, False
    native_tokens = re.findall(r"[a-z0-9]+", fold_text(native))
    ocr_tokens = Counter(re.findall(r"[a-z0-9]+", fold_text(ocr)))
    coverage = sum(min(count, ocr_tokens[token]) for token, count in Counter(native_tokens).items()) / max(1, len(native_tokens))
    if coverage >= 0.78:
        return segments, False
    return [Segment(0, "Text", (0.0, 0.0, 1.0, 1.0), native, "native_page_fallback")], True


def fallback_page_segment(raw: str, native_page: PdfTextPage | None) -> Segment | None:
    native = "\n".join(line.text for line in native_page.lines).strip() if native_page else ""
    text = native or clean_unstructured_output(raw)
    if not text:
        return None
    source = "native_page_fallback" if native else "unstructured_fallback"
    return Segment(0, "Text", (0.0, 0.0, 1.0, 1.0), text, source)


def simple_latex_table_to_markdown(text: str) -> str:
    """Convert regular LaTeX tabular output without changing cell values."""
    match = re.search(
        r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        return text.strip()
    rows: list[list[str]] = []
    for raw_row in re.split(r"\\\\\s*(?:\n|$)", match.group(1).strip()):
        raw_row = re.sub(r"\\(?:hline|toprule|midrule|bottomrule)\b", "", raw_row).strip()
        if not raw_row:
            continue
        protected = raw_row.replace(r"\&", "__ESCAPED_AMPERSAND__")
        cells = [
            cell.strip()
            .replace("__ESCAPED_AMPERSAND__", "&")
            .replace(r"\%", "%")
            .replace(r"\_", "_")
            for cell in re.split(r"\s*&\s*", protected)
        ]
        if any(cell for cell in cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    return "\n".join(
        [markdown_row(rows[0]), markdown_row([":---"] + ["---:"] * (width - 1))]
        + [markdown_row(row) for row in rows[1:]]
    )


def aligned_plain_table_to_markdown(text: str) -> str | None:
    """Convert consistently tab/alignment-delimited OCR rows to Markdown.

    This is intentionally conservative: a header-like first row and at least
    one equally wide data row are required. Single-space prose is never split.
    """
    lines = [line.strip() for line in clean_generated_text(text).splitlines() if line.strip()]
    rows: list[list[str]] = []
    for line in lines:
        if "\t" in line:
            cells = [cell.strip() for cell in re.split(r"\t+", line)]
        else:
            cells = [cell.strip() for cell in re.split(r"\s{2,}", line)]
        cells = [cell for cell in cells if cell]
        if len(cells) >= 2:
            rows.append(cells)
    if len(rows) < 2:
        return None
    widths = Counter(len(row) for row in rows)
    width, count = max(widths.items(), key=lambda item: (item[1], item[0]))
    rectangular = [row for row in rows if len(row) == width]
    if count < 2 or not looks_like_table_header(rectangular[0]):
        return None
    return "\n".join(
        [markdown_row(rectangular[0]), markdown_row([":---"] + ["---:"] * (width - 1))]
        + [markdown_row(row) for row in rectangular[1:]]
    )


def convert_table(text: str, model_dir: str | Path) -> str:
    if "|" in text and re.search(r"(?m)^\s*\|", text):
        return text.strip()
    if "\\begin{tabular}" in text:
        return simple_latex_table_to_markdown(text)
    aligned = aligned_plain_table_to_markdown(text)
    if aligned is not None:
        return aligned
    try:
        sys.path.insert(0, str(model_dir))
        from postprocessing import postprocess_text  # type: ignore

        processed = postprocess_text(
            text, cls="Table", text_format="markdown", table_format="markdown"
        ).strip()
        converted = simple_latex_table_to_markdown(processed)
        return aligned_plain_table_to_markdown(converted) or converted
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError):
        return text.strip()


def uncertain_chart(
    target: Segment, reason: str, candidates: Iterable[Segment | None] = (),
    context: str = "",
) -> Segment:
    """Keep the best direct transcription as data without creating an image.

    Unconfirmed values remain useful for search and manual review, but are
    explicitly labelled as raw OCR. No value is synthesized or reconciled here.
    """
    def eligible_candidate(candidate: Segment | None) -> bool:
        if candidate is None:
            return False
        if candidate.source == "native_label_table":
            return bool(markdown_tables(candidate.text))
        if not is_plausible_chart_transcription(candidate.text, context):
            return False
        # Whole-chart generated tables are never exposed from one pass. A table
        # reconstructed in code from positioned labels may be retained as raw
        # data even when it is too incomplete to confirm.
        return (
            not markdown_tables(candidate.text)
            or candidate.source in {
                "combo_raw_evidence", "evidence_bound_table", "focused_chart_spatial",
                "focused_chart_spatial_consensus", "native_endpoint_labels",
                "native_positioned_time", "native_label_table",
            }
        )

    available = [
        candidate
        for candidate in candidates
        if eligible_candidate(candidate)
    ]
    if (
        eligible_candidate(target)
        and all(candidate is not target for candidate in available)
    ):
        available.append(target)

    def transcription_score(candidate: Segment) -> tuple[int, int, int, int, int]:
        cleaned = clean_generated_text(candidate.text)
        valid = not chart_quality_issues(cleaned, context)
        chart = candidate.chart_ir or chart_ir_from_markdown(
            cleaned, candidate.source, min_data_rows=1
        )
        # A consensus-shaped skeleton containing only em dashes is not data.
        # It must never outrank values read directly from selectable labels.
        consensus = (
            candidate.source == "focused_chart_spatial_consensus"
            and chart is not None
            and chart_ir_category_coverage(chart) >= 0.80
        )
        bound = candidate.source in {
            "combo_raw_evidence", "evidence_bound_table", "focused_chart_spatial",
            "focused_chart_spatial_consensus", "native_endpoint_labels",
            "native_positioned_time", "native_label_table",
        }
        observations = len(chart.observations) if chart is not None else 0
        numbers = len(NUMBER_TOKEN.findall(cleaned))
        return int(consensus), int(bound), int(valid), observations, numbers

    best = max(available, key=transcription_score, default=None)
    note = (
        "*[Dữ liệu OCR thô: chưa đủ nhãn có tọa độ hoặc các lượt OCR chưa khớp. "
        "Giá trị được giữ nguyên như OCR, không nội suy hoặc tự sửa; hãy đối chiếu PDF gốc.]*"
    )
    if best is not None:
        body = clean_generated_text(best.text)
        text = f"<!-- chart_raw_source={best.source} -->\n{note}\n\n{body}"
    else:
        text = (
            f"{note}\n\n"
            "| Trường | Dữ liệu OCR thô |\n"
            "| :--- | :--- |\n"
            "| Trạng thái | Không đọc được nhãn hoặc số liệu đủ rõ từ trang gốc |"
        )
    return Segment(target.order, "Chart", target.bbox, text, f"uncertain:{reason}")


def native_chart_caption(target: Segment, page: PdfTextPage | None) -> list[str]:
    """Read a chart title/subtitle from exact nearby PDF text geometry."""
    if target.source in {
        "native_geometry", "native_title_geometry", "native_profile_geometry",
    }:
        embedded = [
            line.strip() for line in target.text.splitlines()
            if looks_like_visual_caption_text(line)
        ]
        if embedded:
            # Geometry inference already bound this exact title to the plot.
            # Prefer it over nearby unit labels such as "Triệu USD"/"YoC %".
            return embedded[:2]
    if page is None:
        return []
    profile_candidates, _ = infer_native_profile_chart_segment([], page)
    profile = next((
        candidate for candidate in profile_candidates
        if candidate.source == "native_profile_geometry"
        and bbox_overlap(candidate.bbox, target.bbox) >= 0.20
    ), None)
    if profile is not None:
        return [profile.text]
    candidates: list[tuple[float, float, tuple[float, float, float, float], str]] = []
    for line in page.lines:
        box = pdf_bbox_to_model(line.bbox, page)
        line_segment = Segment(0, "Text", box, line.text)
        gap = target.bbox[1] - box[3]
        near_top_inside = target.bbox[1] <= box[1] <= target.bbox[1] + 0.055
        if (
            (-0.012 <= gap <= 0.075 or near_top_inside)
            and horizontal_overlap(target, line_segment) >= 0.30
            and looks_like_visual_caption_text(line.text)
        ):
            height = box[3] - box[1]
            candidates.append((height, box[1], box, line.text.strip()))
    if not candidates:
        return []
    _, _, title_box, title = max(
        candidates,
        key=lambda item: (
            int(len(re.findall(r"[^\W\d_]+", item[3], re.UNICODE)) >= 3),
            int(not CHART_SUBTITLE.search(item[3])),
            # The actual title is normally the lowest eligible line immediately
            # above (or just inside) the visual.  Ranking prose by word count
            # first could steal a sentence from a report table above it.
            item[1],
            min(8, len(re.findall(r"[^\W\d_]+", item[3], re.UNICODE))),
            item[0],
        ),
    )
    subtitles: list[tuple[float, str]] = []
    for line in page.lines:
        box = pdf_bbox_to_model(line.bbox, page)
        if not title_box[1] < box[1] <= title_box[3] + 0.035:
            continue
        line_segment = Segment(0, "Text", box, line.text)
        if horizontal_overlap(target, line_segment) >= 0.25 and CHART_SUBTITLE.search(line.text):
            subtitles.append((box[1], line.text.strip()))
    return [title] + ([min(subtitles)[1]] if subtitles else [])


def attach_chart_captions(
    segments: list[Segment], replacements: dict[int, Segment],
    native_page: PdfTextPage | None = None,
) -> tuple[list[Segment], dict[int, Segment]]:
    """Bind spatially local titles/subtitles to their chart placeholder."""
    consumed: set[int] = set()
    updated = dict(replacements)
    for target_index, replacement in replacements.items():
        target = segments[target_index]
        native = native_chart_caption(target, native_page)
        if native:
            prefix = f"### Biểu đồ: {native[0]}"
            if len(native) > 1:
                prefix += f"\n\n*{native[1]}*"
            body = replacement.text
            # Picture placeholders have no embedded title when their raw label
            # table is built.  Once geometry supplies the nearby title, retain
            # it in the structured payload as well as the Markdown heading.
            body = re.sub(
                r"(?m)^\|\s*Tiêu đề\s*\|\s*\|$",
                markdown_row(["Tiêu đề", native[0]]),
                body,
                count=1,
            )
            updated[target_index] = replace(
                replacement, text=f"{prefix}\n\n{body}"
            )
            native_keys = {normalize_text(value) for value in native}
            for index, candidate in enumerate(segments):
                if index != target_index and normalize_text(candidate.text) in native_keys:
                    consumed.add(index)
            continue
        candidates: list[tuple[float, int, str]] = []
        for index, candidate in enumerate(segments):
            if index == target_index or index in consumed:
                continue
            if candidate.cls.casefold() not in {
                "caption", "section-header", "text", "list-item"
            }:
                continue
            text = re.sub(r"^[#■▪•\-\s]+", "", candidate.text.strip()).strip("*_ ")
            if not text or len(text) > 150 or is_source_label(text):
                continue
            gap = target.bbox[1] - candidate.bbox[3]
            if not -0.008 <= gap <= 0.09:
                continue
            if horizontal_overlap(target, candidate) < 0.40:
                continue
            if looks_like_chart_title(text) or CHART_SUBTITLE.search(text):
                candidates.append((candidate.bbox[1], index, text))
        candidates.sort()
        titles = [item for item in candidates if looks_like_chart_title(item[2])]
        if not titles:
            continue
        title_item = max(titles)
        subtitles = [
            item for item in candidates
            if item[0] >= title_item[0] and item != title_item
            and CHART_SUBTITLE.search(item[2])
        ]
        chosen = [title_item] + ([min(subtitles)] if subtitles else [])
        existing = normalize_text(replacement.text)
        unique = [item for item in chosen if normalize_text(item[2]) not in existing]
        if not unique:
            continue
        title = unique[0][2]
        prefix = f"### Biểu đồ: {title}"
        if len(unique) > 1:
            prefix += f"\n\n*{unique[1][2]}*"
        updated[target_index] = replace(replacement, text=f"{prefix}\n\n{replacement.text}")
        consumed.update(item[1] for item in unique)
    output = [segment for index, segment in enumerate(segments) if index not in consumed]
    remapped: dict[int, Segment] = {}
    for old_index, replacement in updated.items():
        new_index = old_index - sum(index < old_index for index in consumed)
        remapped[new_index] = replacement
    return output, remapped


def remove_empty_markdown_tables(content: str) -> str:
    """Remove table shells whose data rows have no visible cells."""
    lines = content.splitlines()
    removals: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        end = index
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1
        rows = [split_markdown_row(line) for line in lines[index:end]]
        separator = next(
            (i for i, row in enumerate(rows) if row and all(TABLE_SEPARATOR.fullmatch(cell) for cell in row)),
            None,
        )
        if separator is not None:
            data = rows[separator + 1:]
            if not data or not any(any(cell.strip() for cell in row) for row in data):
                removals.append((index, end))
        index = end
    for start, end in reversed(removals):
        del lines[start:end]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def repair_fused_trailing_comparison(content: str) -> str:
    """Split a final ``amount +change`` cell when every table row has it."""
    lines = content.splitlines()
    replacements: list[tuple[int, int, list[str]]] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        end = index
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1
        rows = [split_markdown_row(line) for line in lines[index:end]]
        separator = next(
            (i for i, row in enumerate(rows) if row and all(TABLE_SEPARATOR.fullmatch(cell) for cell in row)),
            None,
        )
        if separator is None or separator == 0:
            index = end
            continue
        header = rows[separator - 1]
        data = rows[separator + 1:]
        last_header = fold_text(header[-1].replace("**", "")) if header else ""
        if not re.fullmatch(r"q[1-4]/20\d{2}", last_header) or len(data) < 2:
            index = end
            continue
        split_values: list[tuple[str, str]] = []
        for row in data:
            if len(row) != len(header):
                split_values = []
                break
            match = re.fullmatch(
                r"(.+?)\s+([+−–-]\d+(?:[.,]\d+)?|n/?a)",
                row[-1].strip(),
                flags=re.IGNORECASE,
            )
            if match is None or not NUMBER_TOKEN.search(match.group(1)):
                split_values = []
                break
            split_values.append((match.group(1), match.group(2)))
        if len(split_values) != len(data):
            index = end
            continue
        bold = any("**" in cell for cell in header)
        header.append("**%QoQ**" if bold else "%QoQ")
        rows[separator].append("---:")
        for row, (amount, comparison) in zip(data, split_values):
            row[-1] = amount
            row.append(comparison)
        replacements.append((index, end, [markdown_row(row) for row in rows]))
        index = end
    for start, end, replacement in reversed(replacements):
        lines[start:end] = replacement
    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")


def convert_latex_tables_in_document(content: str) -> str:
    pattern = re.compile(
        r"\\begin\{tabular\}\{[^}]*\}.*?\\end\{tabular\}", re.DOTALL
    )
    return pattern.sub(lambda match: simple_latex_table_to_markdown(match.group()), content)


def remove_degenerate_document_lines(content: str) -> str:
    """Drop model continuation artifacts while preserving source OCR lines."""
    lines = content.splitlines()
    normalized = [normalize_text(line) for line in lines]
    counts = Counter(value for value in normalized if value)
    kept: list[str] = []
    for line, value in zip(lines, normalized):
        artifact_phrase = MODEL_ARTIFACT.search(line) or re.match(
            r"^\s*[*_#>~-]*\s*(?:the color(?: size)?(?: of the background)?|"
            r"the table(?:\s+is\b| in the bottom box)?|for example\b|"
            r"please see this section\b|\*?<x_[\d.]+>)",
            line, re.IGNORECASE,
        )
        character_collapse = (
            len(line) > 500 and len(set(line.strip())) <= 12
        )
        repeated_prose = (
            value and counts[value] >= 5 and not line.lstrip().startswith("|")
            and not line.lstrip().startswith("<!--")
            and bool(re.search(r"[a-z]{4}", line, re.IGNORECASE))
        )
        if artifact_phrase or character_collapse or repeated_prose:
            continue
        kept.append(line)
    return "\n".join(kept) + ("\n" if content.endswith("\n") else "")


def postprocess_document(content: str) -> str:
    """Safe formatting only: no domain rules and no numeric synthesis."""
    content = remove_degenerate_document_lines(content)
    content = convert_latex_tables_in_document(content)
    content = repair_fused_trailing_comparison(content)
    return remove_empty_markdown_tables(content)


def reconcile_document_tables_with_pdf(content: str, pdf_path: str) -> tuple[str, int]:
    """Replace matching OCR tables with exact cells from the same PDF page."""
    parts = re.split(r"(?=<!-- page \d+ -->)", content)
    replacements = 0
    for part_index in range(1, len(parts)):
        marker = re.match(r"<!-- page (\d+) -->", parts[part_index])
        if marker is None:
            continue
        page_no = int(marker.group(1)) - 1
        candidates = extract_native_table_candidates(extract_pdf_text_page(pdf_path, page_no))
        if not candidates:
            continue
        used: set[int] = set()
        table_pattern = re.compile(r"(?m)(?:^\|.*\|\s*$\n?){3,}")

        def replace_table(match: re.Match[str]) -> str:
            nonlocal replacements
            prefix = parts[part_index][:match.start()]
            structural = list(re.finditer(
                r"(?m)^(?:<!--\s*(?:page\s+\d+|chart_region)\b[^\n]*-->|#{1,6}\s+[^\n]+)$",
                prefix,
            ))
            if structural:
                owner = structural[-1].group().casefold()
                if "chart_region" in owner or "biểu đồ:" in owner:
                    # Chart tables are already governed by label evidence and
                    # strict confirmation.  Native report-table reconciliation
                    # must not overwrite them with an axis-derived skeleton.
                    return match.group()
            original = match.group().rstrip("\n")
            original_tables = markdown_tables(original)
            if not original_tables:
                return match.group()
            rows = original_tables[0]
            ranked: list[tuple[float, int, int, str]] = []
            for candidate_index, candidate in enumerate(candidates):
                if candidate_index in used:
                    continue
                candidate_tables = markdown_tables(candidate)
                if not candidate_tables:
                    continue
                candidate_rows = candidate_tables[0]
                if len(rows[0]) != len(candidate_rows[0]):
                    continue
                candidate_separator = next((
                    index for index, row in enumerate(candidate_rows)
                    if row and all(TABLE_SEPARATOR.fullmatch(cell) for cell in row)
                ), None)
                if (
                    candidate_separator is None
                    or len(candidate_rows) - candidate_separator - 1 < 2
                ):
                    continue
                original_header = [normalize_text(cell) for cell in rows[0]]
                candidate_header = [normalize_text(cell) for cell in candidate_rows[0]]
                header_match = sum(
                    left == right or left in right or right in left
                    for left, right in zip(original_header, candidate_header)
                ) / max(1, len(original_header))
                # A longer native candidate may safely complete a table that
                # OCR stopped midway, but only with the same column headers.
                if len(candidate_rows) < len(rows) or header_match < 0.75:
                    continue
                cell_agreement = keyed_numeric_cell_agreement(original, candidate)
                ranked.append((
                    cell_agreement,
                    len(candidate_rows),
                    candidate_index,
                    candidate,
                ))
            if not ranked or max(ranked)[0] < 0.72:
                return match.group()
            _, _, candidate_index, candidate = max(ranked)
            used.add(candidate_index)
            replacements += 1
            # Keep the following caption/prose outside the Markdown table.
            return candidate + "\n\n"

        parts[part_index] = table_pattern.sub(replace_table, parts[part_index])
    return "".join(parts), replacements


def segment_to_markdown(segment: Segment, model_dir: str | Path) -> str:
    cls, text = segment.cls.casefold(), segment.text.strip()
    if cls == "title":
        return text if text.startswith("#") else f"# {text}"
    if cls == "section-header":
        return text if text.startswith("#") else f"## {text}"
    if cls == "list-item":
        return text if text.startswith(("-", "*", "•")) else f"- {text}"
    if cls == "caption":
        return f"*{text}*" if text else ""
    if cls == "table":
        return convert_table(text, model_dir)
    if cls == "chart":
        status = "uncertain" if segment.source.startswith("uncertain:") else "confirmed"
        source = segment.source.split(":", 1)[0]
        return f"<!-- chart_region source={source} status={status} -->\n{text}"
    if cls == "picture":
        return f"<!-- picture_text source={segment.source} -->\n{text}" if text else ""
    if cls == "page-number":
        return ""
    if cls in {"page-header", "page-footer"}:
        return f"*{text}*" if text else ""
    return text


def segments_to_markdown(segments: list[Segment], model_dir: str | Path) -> str:
    blocks = [segment_to_markdown(segment, model_dir) for segment in segments]
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(block for block in blocks if block)).strip()


def render_page_png_bytes(pdf_path: str, page_no: int, zoom: float) -> bytes:
    with tempfile.TemporaryDirectory(prefix="nemotron-render-") as directory:
        prefix = Path(directory) / "page"
        subprocess.run(
            [
                "pdftoppm", "-f", str(page_no + 1), "-l", str(page_no + 1),
                "-r", str(round(72 * zoom)), "-png", "-singlefile", pdf_path, str(prefix),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return prefix.with_suffix(".png").read_bytes()


def chart_region_png(
    page_png: bytes, bbox: tuple[float, float, float, float], padding: float,
    exact_geometry: bool = False,
) -> bytes:
    """Create an in-memory enlarged chart region for OCR; never persist it."""
    with Image.open(BytesIO(page_png)) as image:
        image = image.convert("RGB")
        width, height = image.size
        resized_width, resized_height = float(width), float(height)
        if resized_height > 2048:
            resized_width *= 2048 / resized_height
            resized_height = 2048.0
        if resized_width > 1664:
            resized_height *= 1664 / resized_width
            resized_width = 1664.0
        pad_x, pad_y = (1664 - resized_width) / 2, (2048 - resized_height) / 2
        x1, y1, x2, y2 = bbox
        left = round(((x1 * 1664 - pad_x) / resized_width) * width)
        top = round(((y1 * 2048 - pad_y) / resized_height) * height)
        right = round(((x2 * 1664 - pad_x) / resized_width) * width)
        bottom = round(((y2 * 2048 - pad_y) / resized_height) * height)
        left, right = max(0, left), min(width, right)
        top, bottom = max(0, top), min(height, bottom)
        region_width = right - left
        inner_x = round(region_width * (0.025 if exact_geometry else max(padding, 0.04)))
        outer_x = round(region_width * (0.025 if exact_geometry else max(padding, 0.25)))
        center_x = (left + right) / 2
        if center_x < width / 2:
            extra_left, extra_right = outer_x, inner_x
        else:
            extra_left, extra_right = inner_x, outer_x
        # Layout bboxes often start at the bar/line body and omit numeric
        # labels printed just above it. Add top-only headroom without widening
        # into a neighbouring chart column.
        extra_top = round(
            (bottom - top) * (0.02 if exact_geometry else max(padding, 0.35))
        )
        # Layout boxes also tend to stop on the baseline and clip low endpoint
        # annotations (for example the lower series label at the final point).
        # A smaller bottom margin than the top headroom is sufficient.
        extra_bottom = round(
            (bottom - top) * (0.025 if exact_geometry else max(padding, 0.10))
        )
        region = (
            max(0, left - extra_left), max(0, top - extra_top),
            min(width, right + extra_right), min(height, bottom + extra_bottom),
        )
        if region[2] <= region[0] or region[3] <= region[1]:
            raise ValueError(f"invalid chart bbox: {bbox}")
        focused = image.crop(region)
        if focused.width < 1600:
            scale = 1600 / focused.width
            focused = focused.resize(
                (1600, max(1, round(focused.height * scale))), Image.Resampling.LANCZOS
            )
        output = BytesIO()
        focused.save(output, format="PNG")
        return output.getvalue()


def chart_category_band_png(
    chart_png: bytes, index: int, count: int, variant: str = "color"
) -> bytes:
    """Crop one x-category band from the plot body of an in-memory chart."""
    if count < 2 or not 0 <= index < count:
        raise ValueError("invalid chart category band")
    with Image.open(BytesIO(chart_png)) as image:
        image = image.convert("RGB")
        width, height = image.size
        # Titles are normally above the plot and source notes below it. Small
        # horizontal overlap keeps labels near a band boundary intact.
        # Exclude title/subtitle text: it adds unrelated digits such as years
        # and competes with the much smaller bar labels.
        # Stop where the dedicated x-axis crop begins. Including the lower
        # axis makes Nemotron return `2026`/`1Q26` instead of the nearby small
        # value label and was a frequent source of false numeric evidence.
        # Wide time-series crops contain title headroom added by
        # ``chart_region_png``; skip that area so a subtitle such as
        # "6 quý gần nhất" cannot become a chart value. Short categorical
        # charts may place the tallest bar label at the very top, so retain it.
        plot_top = 0 if count <= 4 else round(height * 0.20)
        plot_bottom = height
        # Chart titles and source notes normally span the full crop, while the
        # actual x-axis is inset. Dividing the full image shifts the edge
        # categories into the wrong bands, especially on six-period charts.
        plot_left, plot_right = width * 0.08, width * 0.92
        band_width = (plot_right - plot_left) / count
        # Values are centered on their category. A wide overlap makes a model
        # copy the taller neighbouring bar instead of the local small value.
        overlap = band_width * 0.05
        left = max(0, round(plot_left + index * band_width - overlap))
        right = min(width, round(plot_left + (index + 1) * band_width + overlap))
        # Endpoint annotations are frequently printed just outside the first
        # or last marker rather than over its centre.
        if index == 0:
            left = 0
        if index == count - 1:
            right = width
        band = image.crop((left, plot_top, right, plot_bottom))
        ink = ImageOps.invert(ImageOps.grayscale(band)).point(
            lambda value: 255 if value > 12 else 0
        )
        content_box = ink.getbbox()
        if content_box is not None:
            pad = max(8, round(band.width * 0.04))
            band = band.crop((
                max(0, content_box[0] - pad), max(0, content_box[1] - pad),
                min(band.width, content_box[2] + pad),
                min(band.height, content_box[3] + pad),
            ))
        if variant == "contrast":
            band = ImageOps.autocontrast(ImageOps.grayscale(band), cutoff=1).convert("RGB")
            band = ImageEnhance.Contrast(band).enhance(1.35)
            band = band.filter(ImageFilter.SHARPEN)
        elif variant == "binary":
            gray = ImageOps.autocontrast(ImageOps.grayscale(band), cutoff=1)
            band = gray.point(lambda value: 255 if value >= 175 else 0).convert("RGB")
        elif variant != "color":
            raise ValueError(f"unknown chart band variant: {variant}")
        if band.width < 1000:
            scale = 1000 / band.width
            band = band.resize(
                (1000, max(1, round(band.height * scale))), Image.Resampling.LANCZOS
            )
        output = BytesIO()
        band.save(output, format="PNG")
        return output.getvalue()


def chart_axis_band_png(chart_png: bytes, index: int, count: int = 3) -> bytes:
    """Crop one compact x-axis chunk to avoid extreme panoramic input."""
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid chart axis chunk")
    with Image.open(BytesIO(chart_png)) as image:
        image = image.convert("RGB")
        width, height = image.size
        axis_left, axis_right = width * 0.06, width * 0.94
        chunk_width = (axis_right - axis_left) / count
        overlap = chunk_width * 0.06
        left = max(0, round(axis_left + index * chunk_width - overlap))
        right = min(width, round(axis_left + (index + 1) * chunk_width + overlap))
        axis = image.crop((left, round(height * 0.70),
                           right, round(height * 0.88)))
        axis = ImageOps.autocontrast(ImageOps.grayscale(axis), cutoff=1).convert("RGB")
        axis = ImageEnhance.Contrast(axis).enhance(1.35)
        if axis.width < 1000:
            scale = 1000 / axis.width
            axis = axis.resize(
                (1000, max(1, round(axis.height * scale))), Image.Resampling.LANCZOS
            )
        output = BytesIO()
        axis.save(output, format="PNG")
        return output.getvalue()


def save_debug(
    directory: str | Path | None,
    page_no: int,
    pass_name: str,
    attempt: int,
    raw: str,
    response: dict,
) -> None:
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"page_{page_no + 1:04d}.{pass_name}.attempt{attempt}"
    (root / f"{stem}.raw.txt").write_text(raw, encoding="utf-8")
    (root / f"{stem}.response.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def request_ocr(
    *, endpoint: str, model: str, image_b64: str, prompt: str, max_tokens: int,
    timeout: int, retries: int, page_no: int, pass_name: str,
    raw_output_dir: str | Path | None,
) -> str:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "extra_body": {"repetition_penalty": 1.1, "top_k": 1, "skip_special_tokens": False},
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            choice = data.get("choices", [{}])[0]
            raw = ((choice.get("message") or {}).get("content") or "")
            save_debug(raw_output_dir, page_no, pass_name, attempt, raw, data)
            if raw:
                if choice.get("finish_reason") == "length":
                    print(f"WARNING: page={page_no + 1} {pass_name} reached max_tokens", file=sys.stderr)
                return raw
            last_error = RuntimeError("empty OCR response")
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
        if attempt <= retries:
            print(
                f"WARNING: page={page_no + 1} {pass_name} attempt={attempt} failed: {last_error}; retrying",
                file=sys.stderr,
            )
    raise RuntimeError(
        f"page {page_no + 1} {pass_name} failed after {retries + 1} attempt(s): {last_error}"
    )


def read_chart_region(
    *, target: Segment, page_png: bytes, ocr_prompt: str, page_no: int,
    chart_number: int, endpoint: str, model: str, chart_max_tokens: int,
    timeout: int, retries: int, raw_output_dir: str | None, padding: float,
) -> Segment | None:
    """OCR one in-memory chart region and return text/table data only."""
    try:
        raw = request_ocr(
            endpoint=endpoint, model=model,
            image_b64=base64.b64encode(
                chart_region_png(
                    page_png, target.bbox, padding,
                    exact_geometry=target.source.startswith("native_"),
                )
            ).decode("ascii"),
            prompt=ocr_prompt, max_tokens=min(chart_max_tokens, 1200),
            timeout=timeout, retries=retries, page_no=page_no,
            pass_name=f"focused_chart_{chart_number}",
            raw_output_dir=raw_output_dir,
        )
        parts = parse_segments(raw, "focused_chart")
        text = "\n\n".join(
            part.text for part in parts
            if part.cls.casefold() in {"chart", "picture", "table"} and part.text
        ) or clean_unstructured_output(raw)
        if not text.strip() or not is_plausible_chart_transcription(text):
            return None
        return Segment(
            target.order, "Chart", target.bbox, text.strip(), "focused_chart"
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(
            f"WARNING: page={page_no + 1} focused_chart={chart_number} failed: {exc}",
            file=sys.stderr,
        )
        return None


def read_chart_region_evidence(
    *, target: Segment, page_png: bytes, ocr_prompt: str, page_no: int,
    chart_number: int, endpoint: str, model: str, timeout: int, retries: int,
    raw_output_dir: str | None, padding: float,
) -> tuple[Segment, list[Segment]] | None:
    """Read visible chart labels as unstructured evidence for value masking."""
    try:
        raw = request_ocr(
            endpoint=endpoint, model=model,
            image_b64=base64.b64encode(
                chart_region_png(
                    page_png, target.bbox, padding,
                    exact_geometry=target.source.startswith("native_"),
                )
            ).decode("ascii"),
            prompt=ocr_prompt, max_tokens=800, timeout=timeout, retries=retries,
            page_no=page_no, pass_name=f"focused_chart_evidence_{chart_number}",
            raw_output_dir=raw_output_dir,
        )
        labels = parse_segments(raw, "focused_chart_label")
        text = "\n".join(part.text for part in labels if part.text) or clean_unstructured_output(raw)
        if not text or is_degenerate_ocr(text) or len(text) > 5000:
            return None
        return (
            Segment(target.order, "Chart", target.bbox, text, "focused_chart_evidence"),
            labels,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(
            f"WARNING: page={page_no + 1} chart_evidence={chart_number} failed: {exc}",
            file=sys.stderr,
        )
        return None


def chart_categories_from_candidates(
    candidates: Iterable[Segment | None], periods_only: bool = False,
) -> list[str]:
    """Get an ordered category skeleton without trusting candidate values."""
    for candidate in candidates:
        if candidate is None:
            continue
        chart = candidate.chart_ir or chart_ir_from_markdown(
            candidate.text, candidate.source, min_data_rows=1
        )
        if chart is None:
            continue
        categories = list(dict.fromkeys(chart.categories))
        if (
            2 <= len(categories) <= 16
            and (not periods_only or all(is_period_label(item) for item in categories))
        ):
            return categories
    return []


def looks_like_time_chart_context(context: str) -> bool:
    return bool(re.search(
        r"\b(?:quy|quarter|monthly|month|thang|yearly|year|nam|6t|9t)\b",
        fold_text(context),
    ))


def read_chart_time_axis_categories(
    *, target: Segment, page_png: bytes, ocr_prompt: str, page_no: int,
    chart_number: int, endpoint: str, model: str, timeout: int, retries: int,
    raw_output_dir: str | None, padding: float,
) -> list[str]:
    """Read printed periods from an isolated x-axis crop; never synthesize."""
    try:
        chart_png = chart_region_png(
            page_png, target.bbox, padding,
            exact_geometry=target.source.startswith("native_"),
        )
        raw_parts = []
        for index in range(3):
            raw_parts.append(request_ocr(
                endpoint=endpoint, model=model,
                image_b64=base64.b64encode(
                    chart_axis_band_png(chart_png, index)
                ).decode("ascii"),
                prompt=ocr_prompt, max_tokens=160, timeout=timeout, retries=retries,
                page_no=page_no,
                pass_name=f"focused_chart_{chart_number}_axis_{index + 1}",
                raw_output_dir=raw_output_dir,
            ))
        raw = "\n".join(raw_parts)
    except (RuntimeError, ValueError, OSError):
        return []
    categories: list[str] = []
    seen: set[str] = set()
    for match in OCR_CHART_TOKEN.finditer(clean_unstructured_output(raw)):
        token = match.group().strip()
        key = normalize_text(token)
        if is_period_label(token) and key not in seen:
            categories.append(token)
            seen.add(key)
    return categories if 2 <= len(categories) <= 12 else []


def should_read_chart_category_bands(
    categories: list[str], spatial: ChartIR | None
) -> bool:
    """Use local evidence whenever an ordered category skeleton is available."""
    del spatial
    return 2 <= len(categories) <= 12


def read_chart_category_bands(
    *, target: Segment, page_png: bytes, categories: list[str],
    ocr_prompt: str, page_no: int, chart_number: int, endpoint: str,
    model: str, timeout: int, retries: int, raw_output_dir: str | None,
    padding: float, verification_prompt: str | None = None,
    category_source: str = "category_generated",
) -> list[Segment]:
    """OCR independent x bands and optionally retain only two-prompt consensus.

    Requests are deliberately sequential. ``--workers 1`` must also mean one
    in-flight request inside a page; otherwise small-label reads become less
    reproducible on memory-constrained vLLM servers.
    """
    if not 2 <= len(categories) <= 12:
        return []
    try:
        chart_png = chart_region_png(
            page_png, target.bbox, padding,
            exact_geometry=target.source.startswith("native_"),
        )
    except (ValueError, OSError):
        return []

    def request_band(
        index: int, prompt: str, pass_suffix: str, variant: str,
        vertical_zone: str | None = None,
    ) -> str:
        band_png = chart_category_band_png(
            chart_png, index, len(categories), variant=variant
        )
        if vertical_zone is not None:
            with Image.open(BytesIO(band_png)) as band_image:
                band_image = band_image.convert("RGB")
                split = 0.62 if vertical_zone == "top" else 0.38
                box = (
                    (0, 0, band_image.width, round(band_image.height * split))
                    if vertical_zone == "top"
                    else (0, round(band_image.height * split),
                          band_image.width, band_image.height)
                )
                band_image = band_image.crop(box)
                buffer = BytesIO()
                band_image.save(buffer, format="PNG")
                band_png = buffer.getvalue()
        raw = request_ocr(
            endpoint=endpoint, model=model,
            image_b64=base64.b64encode(band_png).decode("ascii"),
            prompt=prompt, max_tokens=160,
            timeout=timeout, retries=retries,
            page_no=page_no,
            pass_name=(
                f"focused_chart_{chart_number}_band_{index + 1}{pass_suffix}"
            ),
            raw_output_dir=raw_output_dir,
        )
        return clean_unstructured_output(raw)

    def numeric_tokens(text: str) -> list[str]:
        result: list[str] = []
        for match in OCR_CHART_TOKEN.finditer(text):
            token = match.group().strip().replace("−", "-").replace("–", "-")
            if is_period_label(token) or compact_numeric_label(token) is None:
                continue
            if token not in result:
                result.append(token)
        return result

    def common_tokens(primary: list[str], verification: list[str]) -> list[str]:
        available = Counter(numeric_label_fingerprint(token) for token in verification)
        result: list[str] = []
        for token in primary:
            fingerprint = numeric_label_fingerprint(token)
            if available[fingerprint] <= 0:
                continue
            available[fingerprint] -= 1
            result.append(token)
        return result

    def majority_tokens(*reads: list[str]) -> list[str]:
        counts = Counter(
            numeric_label_fingerprint(token)
            for read in reads for token in set(read)
        )
        order = [*reads[0], *reads[1], *(reads[2] if len(reads) > 2 else [])]
        result: list[str] = []
        for token in order:
            fingerprint = numeric_label_fingerprint(token)
            if counts[fingerprint] >= 2 and fingerprint not in {
                numeric_label_fingerprint(item) for item in result
            }:
                result.append(token)
        return result

    reads: dict[int, list[str]] = {}
    primary_reads: dict[int, list[str]] = {}
    for index in range(len(categories)):
        primary_suffix = "" if verification_prompt is None else ".labels"
        try:
            primary = numeric_tokens(
                request_band(index, ocr_prompt, primary_suffix, "color")
            )
        except (RuntimeError, ValueError, OSError):
            primary = []
        primary_reads[index] = primary
        if verification_prompt is None:
            reads[index] = primary
            continue
        try:
            verification = numeric_tokens(
                request_band(index, verification_prompt, ".contrast", "contrast")
            )
        except (RuntimeError, ValueError, OSError):
            verification = []
        agreed = common_tokens(primary, verification)
        # A third view is adaptive: it is requested only when the first
        # two differ or one misses a visible number. Exact 2-of-3 voting
        # resolves lost separators without accepting digit-only matches.
        if primary != verification:
            try:
                binary = numeric_tokens(
                    request_band(index, verification_prompt, ".binary", "binary")
                )
            except (RuntimeError, ValueError, OSError):
                binary = []
            agreed = majority_tokens(primary, verification, binary)
            label_binary: list[str] = []
            if binary and not agreed:
                try:
                    label_binary = numeric_tokens(request_band(
                        index, ocr_prompt, ".labels_binary_retry", "binary"
                    ))
                except (RuntimeError, ValueError, OSError):
                    label_binary = []
                zone_labels: list[str] = []
                for zone in ("top", "bottom"):
                    try:
                        zone_labels.extend(numeric_tokens(request_band(
                            index, ocr_prompt, f".labels_{zone}_retry",
                            "color", vertical_zone=zone,
                        )))
                    except (RuntimeError, ValueError, OSError):
                        pass
                # A number reproduced by the plain and bbox tasks on the
                # binary image is cross-prompt evidence. Prefer its printed
                # separator over a bbox-only colour/contrast pair.
                verified_skeletons = {
                    re.sub(r"[.,]", "", numeric_label_fingerprint(token))
                    for token in [*label_binary, *zone_labels]
                }
                for token in binary:
                    if re.sub(
                        r"[.,]", "", numeric_label_fingerprint(token)
                    ) not in verified_skeletons:
                        continue
                    skeleton = re.sub(
                        r"[.,]", "", numeric_label_fingerprint(token)
                    )
                    agreed = [
                        item for item in agreed
                        if re.sub(r"[.,]", "", numeric_label_fingerprint(item))
                        != skeleton
                    ]
                    agreed.append(token)
            # ``output_plain`` occasionally returns an empty response for a
            # very small, otherwise crisp endpoint. Retry that transformed
            # image with the bbox/label task and still require an exact match
            # with the original colour read.
            confirmed_primary = {
                re.sub(r"[.,]", "", numeric_label_fingerprint(token))
                for token in agreed
            }
            if len(primary) > len(agreed) and any(
                re.sub(r"[.,]", "", numeric_label_fingerprint(token))
                not in confirmed_primary for token in primary
            ):
                try:
                    label_contrast = numeric_tokens(request_band(
                        index, ocr_prompt, ".labels_contrast_retry", "contrast"
                    ))
                except (RuntimeError, ValueError, OSError):
                    label_contrast = []
                for token in common_tokens(primary, label_contrast):
                    skeleton = re.sub(
                        r"[.,]", "", numeric_label_fingerprint(token)
                    )
                    if skeleton not in {
                        re.sub(r"[.,]", "", numeric_label_fingerprint(item))
                        for item in agreed
                    }:
                        agreed.append(token)
        reads[index] = agreed

    def separatorless_fingerprint(token: str) -> str:
        return re.sub(r"[.,]", "", numeric_label_fingerprint(token))

    # When all views read the same digits but disagree only about a decimal
    # mark, prefer the colour image: it retains the original thin punctuation
    # better than contrast/binary variants. Voting still decides the digits.
    for index, agreed in reads.items():
        color = primary_reads.get(index, [])
        color_order = {
            separatorless_fingerprint(token): position
            for position, token in enumerate(color)
        }
        if all(separatorless_fingerprint(token) in color_order for token in agreed):
            agreed.sort(key=lambda token: color_order[
                separatorless_fingerprint(token)
            ])

    # Use a clearly dominant decimal style only for one/two-digit fractional
    # labels. This repairs an isolated `43.2` amid `371,5`, `55,8`, `287,7`
    # without converting thousands such as `1.309` or inventing any value.
    decimal_pattern = re.compile(
        r"^[+\-]?\(?\d+([.,])\d{1,2}\)?(?:%|x)?$", re.IGNORECASE
    )
    separators = Counter()
    for tokens in primary_reads.values():
        for token in tokens:
            match = decimal_pattern.fullmatch(token.strip())
            if match:
                separators[match.group(1)] += 1
    accepted_tokens = [token for tokens in reads.values() for token in tokens]
    percent_only = bool(accepted_tokens) and all(
        token.strip().endswith("%") for token in accepted_tokens
    )
    if separators and not percent_only:
        dominant, dominant_count = separators.most_common(1)[0]
        total = sum(separators.values())
        if total >= 3 and dominant_count / total >= 0.75:
            for agreed in reads.values():
                for position, token in enumerate(agreed):
                    match = decimal_pattern.fullmatch(token.strip())
                    if match and match.group(1) != dominant:
                        agreed[position] = (
                            token[:match.start(1)] + dominant + token[match.end(1):]
                        )

    labels: list[Segment] = []
    for index, category in enumerate(categories):
        center = (index + 0.5) / len(categories)
        labels.append(Segment(
            index, "Text", (center - 0.035, 0.82, center + 0.035, 0.86),
            category, category_source,
        ))
        tokens = reads.get(index, [])
        # More than two values in one x band usually means title/axis leakage.
        for value_index, token in enumerate(tokens[:2]):
            labels.append(Segment(
                len(categories) + index * 2 + value_index,
                "Text",
                (center - 0.035, 0.44 + value_index * 0.12,
                 center + 0.035, 0.48 + value_index * 0.12),
                token, (
                    "focused_chart_band_consensus"
                    if verification_prompt is not None
                    else "focused_chart_band"
                ),
            ))
    value_count = sum(
        segment.source.startswith("focused_chart_band") for segment in labels
    )
    return labels if value_count >= 2 else []


def choose_confirmed_chart(
    target: Segment,
    full_page: Segment | None,
    verification: Segment | None,
    context: str,
    confirmation: str,
    raw_fallbacks: Iterable[Segment] = (),
) -> Segment:
    all_candidates = (target, full_page, verification)
    # Only per-category values reproduced by two different OCR task modes may
    # bypass whole-table agreement. A single bbox read is still a candidate,
    # but is never auto-confirmed merely because it looks spatially plausible.
    if (
        verification is not None
        and verification.source == "focused_chart_spatial_consensus"
        and chart_ir_category_coverage(verification.chart_ir) >= 0.80
        and not chart_quality_issues(verification.text, context)
    ):
        return replace(
            verification, order=target.order, bbox=target.bbox, cls="Chart",
            source="focused_chart_spatial_verified",
        )
    candidates = [
        candidate for candidate in all_candidates
        if candidate and not chart_quality_issues(candidate.text, context)
    ]
    if confirmation == "single" and candidates:
        chosen = next(
            candidate
            for candidate in (full_page, verification, target)
            if candidate is not None and not chart_quality_issues(candidate.text, context)
        )
        return replace(chosen, order=target.order, bbox=target.bbox, cls="Chart")
    # Two different OCR task/views may confirm only an exact, complete
    # cell-for-cell match. This recovers clear bar charts while preventing a
    # merely similar table from filling missing labels.
    for first, second in ((target, full_page), (full_page, verification)):
        if (
            first is not None and second is not None
            and not chart_quality_issues(first.text, context)
            and not chart_quality_issues(second.text, context)
            and chart_reads_agree(first.text, second.text)
        ):
            chosen = first if first.source.startswith("pass2") else second
            return replace(
                chosen, order=target.order, bbox=target.bbox, cls="Chart",
                source=f"{first.source}+{second.source}",
            )
    reasons = sorted({
        issue
        for candidate in all_candidates if candidate
        for issue in chart_quality_issues(candidate.text, context)
    })
    return uncertain_chart(
        target, ";".join(reasons) or "OCR reads disagree",
        (*all_candidates, *raw_fallbacks), context,
    )


def process_page(
    *, pdf_path: str, page_no: int, endpoint: str, model: str, zoom: float,
    max_tokens: int, chart_max_tokens: int, timeout: int, text_prompt: str,
    chart_prompt: str, profile_prompt: str | None, profile: str,
    raw_output_dir: str | None, chart_mode: str, retries: int,
    focused_retry: bool, chart_crop_zoom: float, chart_crop_padding: float,
    use_text_layer: bool, chart_confirmation: str,
    focused_ocr_prompt: str | None = None,
    chart_label_prompt: str | None = None,
    uncertain_asset_dir: str | None = None,
    uncertain_asset_prefix: str | None = None,
) -> PageResult:
    # Asset paths are compatibility-only: chart regions exist in memory solely
    # for OCR and are never saved or embedded in Markdown.
    del uncertain_asset_dir, uncertain_asset_prefix
    started = time.perf_counter()
    page_png = render_page_png_bytes(pdf_path, page_no, zoom)
    with Image.open(BytesIO(page_png)) as image:
        render_width, render_height = image.size
    image_b64 = base64.b64encode(page_png).decode("ascii")

    native_page: PdfTextPage | None = None
    if use_text_layer:
        try:
            extracted = extract_pdf_text_page(pdf_path, page_no)
            native_page = replace(extracted, render_width=render_width, render_height=render_height)
        except (OSError, subprocess.SubprocessError, ET.ParseError, ValueError) as exc:
            print(f"WARNING: page={page_no + 1} text layer unavailable: {exc}", file=sys.stderr)

    if profile_prompt is not None:
        raw = request_ocr(
            endpoint=endpoint, model=model, image_b64=image_b64, prompt=profile_prompt,
            max_tokens=max_tokens, timeout=timeout, retries=retries, page_no=page_no,
            pass_name=f"profile_{profile}", raw_output_dir=raw_output_dir,
        )
        segments = parse_segments(raw, f"profile_{profile}")
        markdown = segments_to_markdown(segments, model) if segments else clean_unstructured_output(raw)
        return PageResult(page_no, markdown, 0, 0, 0, 0, time.perf_counter() - started)

    raw1 = request_ocr(
        endpoint=endpoint, model=model, image_b64=image_b64, prompt=text_prompt,
        max_tokens=max_tokens, timeout=timeout, retries=retries, page_no=page_no,
        pass_name="pass1_text", raw_output_dir=raw_output_dir,
    )
    pass1 = parse_segments(raw1, "pass1")
    if not pass1:
        fallback = fallback_page_segment(raw1, native_page)
        if fallback is None:
            raise RuntimeError(f"page {page_no + 1}: no readable OCR output")
        pass1 = [fallback]
    original_ocr_tables = [
        segment for segment in pass1 if segment.cls.casefold() == "table"
    ]
    initial_native_fallback = any(
        segment.source == "native_page_fallback" for segment in pass1
    )

    text_layer_replacements = 0
    if native_page is not None:
        pass1, text_layer_replacements = reconcile_with_text_layer(pass1, native_page)
        pass1, native_table_replacements = reconcile_tables_with_text_layer(pass1, native_page)
        text_layer_replacements += native_table_replacements
        pass1, native_wide_replacements = coalesce_native_wide_tables(pass1, native_page)
        text_layer_replacements += native_wide_replacements
        pass1, visual_replacements = recover_misclassified_visuals(pass1, native_page)
        text_layer_replacements += visual_replacements
        pass1, metric_card_replacements = coalesce_metric_cards(pass1, native_page)
        text_layer_replacements += metric_card_replacements
        # If the layout OCR collapsed or stopped early, preserve the complete
        # native page first. Inferring chart rectangles from isolated labels in
        # that state can turn a table cell, legend, or infographic card into a
        # bogus chart target.
        native_fallback_present = initial_native_fallback or any(
            segment.source == "native_page_fallback" for segment in pass1
        )
        pass1, full_fallback = use_full_native_text_if_ocr_truncated(pass1, native_page)
        text_layer_replacements += int(full_fallback)
        if full_fallback or native_fallback_present:
            pass1, titled_charts = infer_native_chart_segments_from_titles(
                pass1, native_page
            )
            text_layer_replacements += titled_charts
        else:
            pass1, titled_charts = infer_native_chart_segments_from_titles(
                pass1, native_page
            )
            text_layer_replacements += titled_charts
            pass1, inferred_charts = infer_native_chart_segments(pass1, native_page)
            text_layer_replacements += inferred_charts
            pass1, recovered_text = recover_missing_native_text(pass1, native_page)
            text_layer_replacements += recovered_text
        pass1, profile_charts = infer_native_profile_chart_segment(
            pass1, native_page
        )
        text_layer_replacements += profile_charts

    targets, reason = chart_target_indexes(pass1)
    # A model class alone is not enough evidence. Preserve any visible text but
    # do not serialize an unrouted decorative/logo region as a chart.
    pass1 = [
        replace(
            segment,
            cls="Text" if segment.text.strip() else "Picture",
            source="unrouted_visual",
        )
        if index not in targets and segment.cls.casefold() == "chart"
        else segment
        for index, segment in enumerate(pass1)
    ]
    run_pass2 = chart_mode == "always" or (chart_mode == "auto" and bool(targets))
    if chart_mode == "never":
        run_pass2 = False
    if chart_mode == "always" and not targets:
        targets = {i for i, segment in enumerate(pass1) if segment.cls.casefold() in {"picture", "chart"}}

    pass2_matches: dict[int, Segment] = {}
    focused_matches: dict[int, Segment] = {}
    focused_evidence: dict[int, Segment] = {}
    focused_labels: dict[int, list[Segment]] = {}
    if run_pass2 and targets:
        try:
            raw2 = request_ocr(
                endpoint=endpoint, model=model, image_b64=image_b64, prompt=chart_prompt,
                max_tokens=chart_max_tokens, timeout=timeout, retries=retries, page_no=page_no,
                pass_name="pass2_chart", raw_output_dir=raw_output_dir,
            )
            pass2_matches = match_chart_ocr_output(
                pass1, targets, raw2, "pass2"
            )
        except RuntimeError as exc:
            print(f"WARNING: page={page_no + 1} full-page chart OCR failed: {exc}", file=sys.stderr)

        if focused_retry:
            focused_page_png = (
                render_page_png_bytes(pdf_path, page_no, chart_crop_zoom)
                if chart_crop_zoom > zoom else page_png
            )
            for chart_number, target_index in enumerate(sorted(targets), start=1):
                focused = read_chart_region(
                    target=pass1[target_index], page_png=focused_page_png,
                    ocr_prompt=chart_prompt, page_no=page_no,
                    chart_number=chart_number, endpoint=endpoint, model=model,
                    chart_max_tokens=chart_max_tokens, timeout=timeout,
                    retries=retries, raw_output_dir=raw_output_dir,
                    padding=chart_crop_padding,
                )
                if focused is not None:
                    focused_matches[target_index] = focused
                evidence_result = read_chart_region_evidence(
                    target=pass1[target_index], page_png=focused_page_png,
                    ocr_prompt=chart_label_prompt or focused_ocr_prompt or chart_prompt,
                    page_no=page_no,
                    chart_number=chart_number, endpoint=endpoint, model=model,
                    timeout=timeout, retries=retries,
                    raw_output_dir=raw_output_dir, padding=chart_crop_padding,
                )
                if evidence_result is not None:
                    evidence, labels = evidence_result
                    focused_evidence[target_index] = evidence
                    focused_labels[target_index] = labels
                current_labels = focused_labels.get(target_index, [])
                current_context = local_chart_context(pass1[target_index], pass1)
                spatial = spatial_time_chart_ir(current_labels, current_context)
                if spatial is None:
                    spatial = spatial_categorical_chart_ir(
                        current_labels, current_context
                    )
                time_context = looks_like_time_chart_context(current_context)
                categories = chart_categories_from_candidates((
                    focused_matches.get(target_index),
                    pass2_matches.get(target_index),
                    pass1[target_index],
                ), periods_only=time_context)
                category_source = "category_generated"
                if spatial is not None:
                    spatial_categories = list(dict.fromkeys(spatial.categories))
                    if len(spatial_categories) > len(categories):
                        categories = spatial_categories
                        category_source = "focused_chart_label"
                if (
                    time_context
                    or (categories and all(is_period_label(item) for item in categories))
                ):
                    axis_categories = read_chart_time_axis_categories(
                        target=pass1[target_index], page_png=focused_page_png,
                        ocr_prompt=focused_ocr_prompt or chart_label_prompt or chart_prompt,
                        page_no=page_no, chart_number=chart_number,
                        endpoint=endpoint, model=model, timeout=timeout,
                        retries=retries, raw_output_dir=raw_output_dir,
                        padding=chart_crop_padding,
                    )
                    candidate_periods_valid = bool(categories) and all(
                        is_period_label(item) for item in categories
                    )
                    if axis_categories and (
                        not candidate_periods_valid
                        or len(axis_categories) >= len(categories)
                    ):
                        categories = axis_categories
                        category_source = "focused_chart_axis"
                    elif not candidate_periods_valid:
                        categories = []
                # Whole-chart OCR can look complete while assigning cropped or
                # neighbouring digits to the wrong periods. Time-series charts
                # therefore always receive local, per-period verification.
                if should_read_chart_category_bands(categories, spatial):
                    band_labels = read_chart_category_bands(
                        target=pass1[target_index], page_png=focused_page_png,
                        categories=categories,
                        ocr_prompt=chart_label_prompt or focused_ocr_prompt or chart_prompt,
                        page_no=page_no, chart_number=chart_number,
                        endpoint=endpoint, model=model, timeout=timeout,
                        retries=retries, raw_output_dir=raw_output_dir,
                        padding=chart_crop_padding,
                        verification_prompt=(
                            focused_ocr_prompt
                            if focused_ocr_prompt != chart_label_prompt else None
                        ),
                        category_source=category_source,
                    )
                    if band_labels:
                        # Never let a sparse per-band retry erase a stronger
                        # positioned-label read from the whole chart crop.
                        selected_labels = choose_stronger_spatial_labels(
                            current_labels, band_labels, current_context
                        )
                        if selected_labels is band_labels:
                            focused_labels[target_index] = selected_labels
                            band_text = "\n".join(
                                label.text for label in band_labels
                                if label.source.startswith("focused_chart_band")
                            )
                            focused_evidence[target_index] = Segment(
                                pass1[target_index].order, "Chart",
                                pass1[target_index].bbox, band_text,
                                "focused_chart_band_evidence",
                            )

    replacements: dict[int, Segment] = {}
    for target_index in sorted(targets):
        target = pass1[target_index]
        target_for_choice = target
        context = local_chart_context(target, pass1)
        full_page = pass2_matches.get(target_index)
        focused = focused_matches.get(target_index)
        evidence = focused_evidence.get(target_index)
        spatial_ir = spatial_time_chart_ir(
            focused_labels.get(target_index, []), context
        )
        if spatial_ir is None:
            spatial_ir = spatial_categorical_chart_ir(
                focused_labels.get(target_index, []), context
            )
        if spatial_ir is not None:
            spatial_values = [
                label for label in focused_labels.get(target_index, [])
                if compact_numeric_label(label.text) is not None
                and not is_period_label(label.text)
            ]
            consensus = bool(spatial_values) and all(
                label.source == "focused_chart_band_consensus"
                for label in spatial_values
            ) and all(
                label.source in {"focused_chart_axis", "focused_chart_label"}
                for label in focused_labels.get(target_index, [])
                if compact_numeric_label(label.text) is None
            )
            focused = Segment(
                target.order, "Chart", target.bbox, render_chart_ir(spatial_ir),
                (
                    "focused_chart_spatial_consensus"
                    if consensus else "focused_chart_spatial"
                ),
                spatial_ir,
            )
        if evidence is not None:
            if markdown_tables(target_for_choice.text):
                target_for_choice = replace(
                    target_for_choice,
                    text=mask_combo_line_without_evidence(
                        target_for_choice.text, context, evidence.text
                    ),
                )
            if full_page is not None and markdown_tables(full_page.text):
                full_page = replace(
                    full_page,
                    text=mask_combo_line_without_evidence(
                        full_page.text, context, evidence.text
                    ),
                )
            if (
                focused is not None and markdown_tables(focused.text)
                and not focused.source.startswith("focused_chart_spatial")
            ):
                focused = replace(
                    focused,
                    text=mask_combo_line_without_evidence(
                        focused.text, context, evidence.text
                    ),
                )
        else:
            if markdown_tables(target_for_choice.text):
                target_for_choice = replace(
                    target_for_choice,
                    text=mask_combo_line_without_evidence(
                        target_for_choice.text, context
                    ),
                )
            if full_page is not None and markdown_tables(full_page.text):
                full_page = replace(
                    full_page,
                    text=mask_combo_line_without_evidence(full_page.text, context),
                )
            if focused is not None and markdown_tables(focused.text):
                focused = replace(
                    focused,
                    text=mask_combo_line_without_evidence(focused.text, context),
                )
        raw_fallbacks: list[Segment] = []
        native_time_candidate: Segment | None = None
        if evidence is not None:
            transposed = transposed_period_chart_ir(
                evidence.text, "focused_chart_transposed_evidence"
            )
            if transposed is not None:
                raw_fallbacks.append(Segment(
                    target.order, "Chart", target.bbox,
                    render_chart_ir(transposed),
                    "focused_chart_spatial", transposed,
                ))
        else:
            transposed = None
        combo_candidates = [
            combo_raw_chart_candidate(candidate, transposed, context)
            for candidate in (pass2_matches.get(target_index), target)
        ]
        raw_fallbacks.extend(
            candidate for candidate in combo_candidates if candidate is not None
        )
        generated_candidates = (
            target,
            pass2_matches.get(target_index),
        )
        generated_irs = {
            id(candidate): chart_ir_from_markdown(
                candidate.text, candidate.source, min_data_rows=1
            )
            for candidate in generated_candidates
            if candidate is not None and markdown_tables(candidate.text)
        }
        max_series = max(
            (len(chart.series) for chart in generated_irs.values() if chart is not None),
            default=0,
        )
        for generated_candidate in generated_candidates:
            generated_ir = (
                generated_irs.get(id(generated_candidate))
                if generated_candidate is not None else None
            )
            if generated_ir is not None and len(generated_ir.series) < max_series:
                continue
            evidence_bound = evidence_bound_chart_candidate(
                generated_candidate, evidence, context
            )
            if evidence_bound is not None:
                raw_fallbacks.append(evidence_bound)
        # Label evidence is an input to spatial reconstruction and cell masks;
        # it is not itself a candidate chart table.
        if native_page is not None:
            native_time_ir = native_positioned_time_chart_ir(target, native_page)
            if native_time_ir is not None:
                native_time_candidate = Segment(
                    target.order, "Chart", target.bbox,
                    render_chart_ir(native_time_ir),
                    "native_positioned_time", native_time_ir,
                )
                raw_fallbacks.append(native_time_candidate)
            native_endpoint_ir = native_endpoint_time_chart_ir(target, native_page)
            if native_endpoint_ir is not None:
                raw_fallbacks.append(Segment(
                    target.order, "Chart", target.bbox,
                    render_chart_ir(native_endpoint_ir),
                    "native_endpoint_labels", native_endpoint_ir,
                ))
            native_labels = native_chart_label_table(target, native_page)
            if native_labels is not None:
                raw_fallbacks.append(Segment(
                    target.order, "Chart", target.bbox, native_labels,
                    "native_label_table",
                ))
            native_raw = native_text_for_segment(target, native_page)
            if native_raw:
                raw_fallbacks.append(
                    Segment(
                        target.order, "Chart", target.bbox, native_raw,
                        "text_layer_chart_raw",
                    )
                )
        chosen = choose_confirmed_chart(
            target_for_choice, full_page, focused,
            context, chart_confirmation, raw_fallbacks,
        )
        if native_page is not None:
            chosen = bind_chart_to_native_value_labels(
                chosen, target, native_page, context
            )
        if native_time_candidate is not None:
            # Selectable period/value labels bound by their printed x-position
            # are stronger evidence than agreeing image-model guesses.  The
            # latter can agree on the same invented points and then be reduced
            # to an empty skeleton by the native-label safety mask.
            chosen = uncertain_chart(
                target, "native positioned labels",
                (native_time_candidate,), context,
            )
        replacements[target_index] = chosen

    pass1, replacements = attach_chart_captions(pass1, replacements, native_page)
    final_segments = [replacements.get(index, segment) for index, segment in enumerate(pass1)]
    final_segments, restored_tables = restore_richer_original_tables(
        final_segments, original_ocr_tables
    )
    text_layer_replacements += restored_tables
    final_segments.sort(key=lambda segment: (round(segment.bbox[1], 3), segment.bbox[0], segment.order))
    coverage = native_text_coverage(final_segments, native_page) if native_page is not None else 1.0
    if native_page is not None and coverage < 0.70:
        raise RuntimeError(
            f"page {page_no + 1}: native text coverage {coverage:.1%} is below 70%; "
            "refusing to write an incomplete page"
        )
    markdown = postprocess_document(segments_to_markdown(final_segments, model)).strip()
    if not markdown:
        raise RuntimeError(f"page {page_no + 1}: empty Markdown after processing")
    confirmed = sum(1 for segment in replacements.values() if not segment.source.startswith("uncertain:"))
    uncertain = len(replacements) - confirmed
    print(
        f"page={page_no + 1} targets={len(targets)} confirmed={confirmed} uncertain={uncertain} "
        f"text_layer={text_layer_replacements} native_coverage={coverage:.1%} "
        f"reason={reason!r} elapsed={time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return PageResult(
        page_no, markdown, len(targets), confirmed, uncertain,
        text_layer_replacements, time.perf_counter() - started, coverage,
    )


def pdf_page_count(pdf_path: str) -> int:
    info = subprocess.check_output(["pdfinfo", pdf_path], text=True)
    match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    if not match:
        raise RuntimeError("pdfinfo did not return a page count")
    return int(match.group(1))


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False,
        prefix=path.name + ".", suffix=".tmp",
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def parse_pdf(
    *, pdf_path: str, endpoint: str, model: str, output: str, zoom: float,
    max_tokens: int, chart_max_tokens: int, timeout: int, text_prompt_file: str,
    chart_prompt_file: str, ocr_prompt_file: str, plain_prompt_file: str,
    fast_prompt_file: str, profile: str, raw_output_dir: str | None,
    page: int | None, chart_mode: str, retries: int, focused_retry: bool,
    chart_crop_zoom: float, chart_crop_padding: float, use_text_layer: bool = True,
    postprocess_domain: str = "auto", workers: int = 1,
    chart_confirmation: str = "strict",
    save_uncertain_chart_images: bool = False,
) -> None:
    """Process any PDF without filename-, issuer-, page-, or finance-specific rules."""
    # Backward-compatible options; synthesis and uncertain image assets are
    # intentionally disabled.
    del postprocess_domain, save_uncertain_chart_images
    started = time.perf_counter()
    prompts = {
        "text": load_task_prompt(text_prompt_file),
        "chart": load_task_prompt(chart_prompt_file),
        "label": load_task_prompt(PROMPT_FILES["label"]),
        "ocr": load_task_prompt(ocr_prompt_file),
        "plain": load_task_prompt(plain_prompt_file),
        "fast": load_task_prompt(fast_prompt_file),
    }
    pages = pdf_page_count(pdf_path)
    if page is not None and not 1 <= page <= pages:
        raise ValueError(f"--page must be between 1 and {pages}")
    page_numbers = [page - 1] if page is not None else list(range(pages))
    requested_profile = profile
    effective_profile = "auto" if profile in {"auto", "mixed"} else profile
    if requested_profile == "mixed":
        print(
            "WARNING: --profile mixed is an alias of the canonical auto pipeline; "
            "use --chart-mode always to force chart OCR.",
            file=sys.stderr,
        )
    profile_prompts = {**prompts, "handwriting": prompts["chart"]}
    profile_prompt = (
        None if effective_profile == "auto" else profile_prompts[effective_profile]
    )
    run_raw_output_dir = (
        str(Path(raw_output_dir) / Path(pdf_path).stem)
        if raw_output_dir else None
    )
    kwargs = dict(
        pdf_path=pdf_path, endpoint=endpoint, model=model, zoom=zoom,
        max_tokens=max_tokens, chart_max_tokens=chart_max_tokens, timeout=timeout,
        text_prompt=prompts["text"], chart_prompt=prompts["chart"],
        focused_ocr_prompt=prompts["ocr"],
        chart_label_prompt=prompts["label"],
        profile_prompt=profile_prompt, profile=effective_profile,
        raw_output_dir=run_raw_output_dir,
        chart_mode=chart_mode, retries=retries, focused_retry=focused_retry,
        chart_crop_zoom=chart_crop_zoom, chart_crop_padding=chart_crop_padding,
        use_text_layer=use_text_layer, chart_confirmation=chart_confirmation,
        uncertain_asset_dir=None, uncertain_asset_prefix=None,
    )
    results: list[PageResult] = []
    worker_count = min(max(1, workers), len(page_numbers))
    if worker_count == 1:
        results = [process_page(page_no=number, **kwargs) for number in page_numbers]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(process_page, page_no=number, **kwargs): number for number in page_numbers}
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda result: result.page_no)

    run_metadata = (
        f"<!-- ocr_run requested_profile={requested_profile} "
        f"effective_profile={effective_profile} chart_mode={chart_mode} "
        f"chart_confirmation={chart_confirmation} zoom={zoom:g} "
        f"chart_ocr_zoom={chart_crop_zoom:g} text_layer={str(use_text_layer).lower()} "
        f"raw_output_dir={run_raw_output_dir or 'disabled'} -->"
    )
    content = f"# {Path(pdf_path).name}\n\n{run_metadata}\n\n" + "\n\n---\n\n".join(
        f"<!-- page {result.page_no + 1} -->\n\n{result.markdown}" for result in results
    ) + "\n"
    content = postprocess_document(content)
    if use_text_layer:
        content, document_table_replacements = reconcile_document_tables_with_pdf(
            content, pdf_path
        )
        if document_table_replacements:
            print(
                f"document_text_layer_tables={document_table_replacements}",
                flush=True,
            )
    atomic_write(Path(output), content)
    print(
        f"output={output}\npages={len(results)}\nworkers={worker_count}\n"
        f"requested_profile={requested_profile}\neffective_profile={effective_profile}\n"
        f"chart_confirmation={chart_confirmation}\n"
        f"chart_targets={sum(result.chart_targets for result in results)}\n"
        f"confirmed_charts={sum(result.confirmed_charts for result in results)}\n"
        f"uncertain_charts={sum(result.uncertain_charts for result in results)}\n"
        f"elapsed={time.perf_counter() - started:.2f}s"
    )


def parse_pdf_parallel(*, workers: int, **kwargs) -> None:
    """Backward-compatible entry point."""
    parse_pdf(workers=workers, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evidence-first Nemotron PDF OCR; charts are emitted as table/raw data, never crops"
    )
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--page", type=int, default=None, help="Process one 1-based page")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--profile", choices=("auto", "text", "chart", "mixed", "handwriting", "ocr", "plain", "fast"),
        default="auto",
        help="auto is the canonical pipeline; mixed is a compatibility alias",
    )
    parser.add_argument("--text-prompt-file", default=str(PROMPT_FILES["text"]))
    parser.add_argument("--chart-prompt-file", default=str(PROMPT_FILES["chart"]))
    parser.add_argument("--ocr-prompt-file", default=str(PROMPT_FILES["ocr"]))
    parser.add_argument("--plain-prompt-file", default=str(PROMPT_FILES["plain"]))
    parser.add_argument("--fast-prompt-file", default=str(PROMPT_FILES["fast"]))
    parser.add_argument(
        "--raw-output-dir", default="debug_ocr",
        help="save raw model responses for audit (default: debug_ocr)",
    )
    parser.add_argument("--chart-mode", choices=("auto", "always", "never"), default="auto")
    parser.add_argument(
        "--chart-confirmation", choices=("strict", "single"), default="strict",
        help=(
            "strict requires agreeing reads, or per-category two-prompt consensus; "
            "single accepts any one structurally valid read"
        ),
    )
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--chart-max-tokens", type=int, default=5000)
    parser.add_argument(
        "--chart-ocr-focus", dest="focused_retry",
        action=argparse.BooleanOptionalAction, default=True,
        help="OCR enlarged chart regions in memory; output remains table/raw text only",
    )
    parser.add_argument(
        "--chart-verification-pass", dest="focused_retry",
        action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--focused-retry", dest="focused_retry",
        action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--chart-crop", dest="focused_retry", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-chart-crop", dest="focused_retry", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--chart-ocr-zoom", dest="chart_crop_zoom", metavar="ZOOM",
        type=float, default=3.5,
    )
    parser.add_argument(
        "--chart-verify-zoom", dest="chart_crop_zoom", type=float,
        default=argparse.SUPPRESS, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--chart-crop-zoom", dest="chart_crop_zoom", type=float,
        default=argparse.SUPPRESS, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--chart-crop-padding", type=float, default=0.08,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--uncertain-chart-images", action=argparse.BooleanOptionalAction, default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--text-layer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--postprocess-domain", choices=("auto", "generic", "financial"), default="auto",
        help="Deprecated compatibility option; no domain-specific value synthesis is run",
    )
    args = parser.parse_args()
    try:
        parse_pdf(
            workers=max(1, args.workers), pdf_path=args.pdf, endpoint=args.endpoint,
            model=args.model, output=args.output, zoom=args.zoom,
            max_tokens=args.max_tokens, chart_max_tokens=args.chart_max_tokens,
            timeout=args.timeout, text_prompt_file=args.text_prompt_file,
            chart_prompt_file=args.chart_prompt_file, ocr_prompt_file=args.ocr_prompt_file,
            plain_prompt_file=args.plain_prompt_file, fast_prompt_file=args.fast_prompt_file,
            profile=args.profile, raw_output_dir=args.raw_output_dir, page=args.page,
            chart_mode=args.chart_mode, retries=max(0, args.retries),
            focused_retry=args.focused_retry,
            chart_crop_zoom=max(args.zoom, args.chart_crop_zoom),
            chart_crop_padding=max(0.0, min(0.30, args.chart_crop_padding)),
            use_text_layer=args.text_layer, postprocess_domain=args.postprocess_domain,
            chart_confirmation=args.chart_confirmation,
            save_uncertain_chart_images=args.uncertain_chart_images,
        )
    except (requests.RequestException, RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"OCR failed: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

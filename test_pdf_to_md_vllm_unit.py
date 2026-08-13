import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from pdf_to_md_vllm import (
    PdfTextLine,
    PdfTextPage,
    Segment,
    aligned_plain_table_to_markdown,
    attach_chart_captions,
    chart_ir_from_markdown,
    chart_quality_issues,
    chart_reads_agree,
    chart_target_indexes,
    choose_confirmed_chart,
    coalesce_metric_cards,
    coalesce_native_wide_tables,
    dedupe_consecutive_rows,
    infer_native_chart_segments,
    is_degenerate_ocr,
    match_chart_ocr_output,
    match_pass2_charts,
    mask_unverified_chart_values,
    mask_combo_line_without_evidence,
    native_chart_caption,
    native_metric_card_segments,
    native_text_coverage,
    native_wide_table_segments,
    parse_segments,
    postprocess_document,
    process_page,
    recover_misclassified_visuals,
    recover_missing_native_text,
    read_chart_category_bands,
    reconcile_document_tables_with_pdf,
    remove_degenerate_document_lines,
    reconcile_tables_with_text_layer,
    repair_fused_trailing_comparison,
    simple_latex_table_to_markdown,
    spatial_time_chart_table,
    spatial_categorical_chart_table,
    use_full_native_text_if_ocr_truncated,
)


GOOD_CHART = """### Biểu đồ: Doanh thu
| Kỳ | Giá trị |
| :--- | ---: |
| Q1/2026 | 100 |
| Q2/2026 | 120 |"""


class SegmentParsingTests(unittest.TestCase):
    def test_parses_first_block_when_start_bbox_is_omitted(self):
        raw = "Tiêu đề<x_0.8><y_0.1><class_Title>"
        segments = parse_segments(raw)
        self.assertEqual(1, len(segments))
        self.assertEqual("Tiêu đề", segments[0].text)
        self.assertEqual((0.0, 0.0, 0.8, 0.1), segments[0].bbox)

    def test_cleans_control_tokens_and_model_artifact(self):
        raw = (
            "<predict_bbox><x_0.1><y_0.1>Nội dung\n"
            "For example, if any, the number of items used here is less than 5"
            "<x_0.8><y_0.2><class_Text>"
        )
        segment = parse_segments(raw)[0]
        self.assertEqual("Nội dung", segment.text)

    def test_converts_regular_latex_table_without_changing_values(self):
        latex = r"""\begin{tabular}{ccc}
**Chỉ tiêu** & **Q1** & **Q2**\\
Lãi thuần \& dịch vụ & (10) & 12,5\%\\
\end{tabular}"""
        result = simple_latex_table_to_markdown(latex)
        self.assertIn("| **Chỉ tiêu** | **Q1** | **Q2** |", result)
        self.assertIn("| Lãi thuần & dịch vụ | (10) | 12,5% |", result)

    def test_converts_aligned_plain_ocr_table_without_splitting_prose(self):
        source = "Chỉ tiêu  Q1/2026  Q2/2026\nDoanh thu  100  120\nLợi nhuận  10  12"
        result = aligned_plain_table_to_markdown(source)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("| Chỉ tiêu | Q1/2026 | Q2/2026 |", result)
        self.assertIn("| Doanh thu | 100 | 120 |", result)
        self.assertIsNone(aligned_plain_table_to_markdown("Đây là một đoạn văn bình thường"))

    def test_safe_postprocess_removes_empty_table_shell(self):
        source = """Before

| Q1 | Q2 |
| --- | ---: |
|  |  |

After"""
        result = postprocess_document(source)
        self.assertNotIn("| Q1 | Q2 |", result)
        self.assertIn("Before", result)
        self.assertIn("After", result)

    def test_document_postprocess_drops_repeated_english_artifact(self):
        source = "Nội dung gốc\n" + "\n".join(
            ["The table is the table in the bottom box."] * 12
        )
        result = remove_degenerate_document_lines(source)
        self.assertEqual("Nội dung gốc", result)

    def test_latex_empty_row_is_removed_after_conversion(self):
        source = r"""\begin{tabular}{ccc}
**Q1** & **Q2** & **Q3**\\
 & & \\
\end{tabular}"""
        result = postprocess_document(source)
        self.assertNotIn("| **Q1**", result)

    def test_splits_consistently_fused_trailing_comparison_column(self):
        source = """| Chỉ tiêu | Q2/2026 | Q1/2026 |
| :--- | ---: | ---: |
| Thu nhập | 100 | 90 +11,1 |
| Chi phí | (50) | (40) -25,0 |"""
        result = repair_fused_trailing_comparison(source)
        self.assertIn("| Chỉ tiêu | Q2/2026 | Q1/2026 | %QoQ |", result)
        self.assertIn("| Thu nhập | 100 | 90 | +11,1 |", result)

    def test_dedupes_only_adjacent_identical_native_rows(self):
        rows = [["A", "1"], ["A", "1"], ["B", "1"], ["A", "1"]]
        self.assertEqual(
            [["A", "1"], ["B", "1"], ["A", "1"]],
            dedupe_consecutive_rows(rows),
        )


class ChartEvidenceTests(unittest.TestCase):
    def test_accepts_structured_chart(self):
        self.assertEqual([], chart_quality_issues(GOOD_CHART))

    def test_rejects_prose_with_numbers_but_no_data_table(self):
        issues = chart_quality_issues("Doanh thu đạt 100 và tăng 20%.")
        self.assertIn("no Markdown data table", issues)

    def test_rejects_repeated_model_continuation_artifact(self):
        artifact = "\n".join(
            ["The table is the table in the bottom box."] * 20
        )
        self.assertTrue(is_degenerate_ocr(artifact))

    def test_rejects_inconsistent_columns(self):
        chart = """| Kỳ | A | B |
| :--- | ---: | ---: |
| Q1 | 10 |
| Q2 | 11 | 12 |"""
        self.assertIn("inconsistent table column count", chart_quality_issues(chart))

    def test_rejects_synthetic_long_numeric_category_table(self):
        rows = "\n".join(f"| {index},9 | {-index / 2:.1f} |" for index in range(10, 60))
        chart = f"| Category | Value (%) |\n| --- | ---: |\n{rows}"
        issues = chart_quality_issues(chart)
        self.assertTrue(
            "implausibly many chart rows" in issues
            or "numeric values used as category labels" in issues
        )

    def test_masks_line_values_not_present_in_direct_ocr(self):
        structured = """| Kỳ | Doanh thu | Biên gộp |
| :--- | ---: | ---: |
| 1Q25 | 83,8 | 20,1% |
| 2Q25 | 46,4 | 18,2% |
| 3Q25 | 39,1 | -18,6% |"""
        evidence = "1Q25 83,8 2Q25 46,4 3Q25 39,1 -18,6%"
        result = mask_unverified_chart_values(structured, evidence)
        self.assertIn("| 1Q25 | 83,8 | — |", result)
        self.assertIn("| 2Q25 | 46,4 | — |", result)
        self.assertIn("| 3Q25 | 39,1 | -18,6% |", result)

    def test_value_mask_never_rewrites_numeric_series_headers(self):
        structured = """| Kỳ | Chuỗi 1 | Chuỗi 2 |
| :--- | ---: | ---: |
| 2025 | 10 | 20 |
| 2026 | 11 | 21 |"""
        result = mask_unverified_chart_values(structured, "2025 2026 10 20")
        self.assertIn("| Kỳ | Chuỗi 1 | Chuỗi 2 |", result)

    def test_masks_entire_combo_line_series_when_evidence_is_missing(self):
        structured = """| Kỳ | Doanh thu | Biên gộp (%) |
| :--- | ---: | ---: |
| 1Q25 | 83,8 | 20,1% |
| 2Q25 | 46,4 | 18,2% |"""
        result = mask_combo_line_without_evidence(
            structured, "Cột = doanh thu · đường = biên gộp, trục phải"
        )
        self.assertIn("| 1Q25 | 83,8 | — |", result)
        self.assertIn("| 2Q25 | 46,4 | — |", result)

    def test_spatial_time_chart_binds_periods_bars_and_only_labeled_line_points(self):
        labels = []
        periods = ["1Q25", "2Q25", "3Q25", "4Q25"]
        amounts = ["83,8", "46,4", "39,1", "162,2"]
        for index, (period, amount) in enumerate(zip(periods, amounts)):
            x = 0.15 + index * 0.22
            labels.append(Segment(index, "Text", (x, 0.75, x + 0.06, 0.79), period))
            labels.append(Segment(index + 10, "Text", (x, 0.45, x + 0.06, 0.49), amount))
        labels.append(Segment(30, "Text", (0.59, 0.60, 0.66, 0.64), "-18,6%"))
        result = spatial_time_chart_table(
            labels, "Cột = doanh thu · đường = biên lãi gộp (%, trục phải)"
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("| Kỳ | Giá trị | Giá trị (%) |", result)
        self.assertIn("| 1Q25 | 83,8 | — |", result)
        self.assertIn("| 3Q25 | 39,1 | -18,6% |", result)

    def test_spatial_time_chart_keeps_two_labeled_ratio_series(self):
        labels = []
        for index, period in enumerate(("2023", "2024", "2025", "6T26")):
            x = 0.15 + index * 0.22
            labels.extend([
                Segment(index, "Text", (x, 0.75, x + 0.06, 0.79), period),
                Segment(index + 10, "Text", (x, 0.40, x + 0.06, 0.44), ("3,9x", "3,1x", "3,7x", "2,9x")[index]),
                Segment(index + 20, "Text", (x, 0.55, x + 0.06, 0.59), ("3,3x", "2,3x", "2,7x", "2,2x")[index]),
            ])
        result = spatial_time_chart_table(
            labels, "Tổng nợ vay/EBITDA & Nợ ròng/EBITDA"
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("| 2024 | 3,1x | 2,3x |", result)

    def test_spatial_time_chart_is_domain_agnostic_for_month_labels(self):
        labels = []
        for index, (period, value) in enumerate(
            (("Jan 2026", "18,2"), ("Feb 2026", "21,4"), ("Mar 2026", "19,7"))
        ):
            x = 0.15 + index * 0.30
            labels.extend([
                Segment(index, "Text", (x, 0.76, x + 0.10, 0.80), period),
                Segment(index + 10, "Text", (x, 0.45, x + 0.08, 0.49), value),
            ])
        result = spatial_time_chart_table(labels, "Average temperature")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("| Jan 2026 | 18,2 |", result)
        self.assertIn("| Mar 2026 | 19,7 |", result)

    def test_category_band_fallback_recovers_each_visible_bar_label(self):
        image = Image.new("RGB", (1200, 800), "white")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        categories = ["1Q25", "2Q25", "3Q25", "4Q25", "1Q26", "2Q26"]
        values = ["83,8", "46,4", "39,1 -18,6%", "162,2", "104,4", "117,9 23,3%"]

        def fake_request_ocr(**kwargs):
            index = int(kwargs["pass_name"].rsplit("_", 1)[1]) - 1
            return f"{values[index]}\n{categories[index]}"

        with patch("pdf_to_md_vllm.request_ocr", side_effect=fake_request_ocr):
            labels = read_chart_category_bands(
                target=Segment(0, "Chart", (0.0, 0.0, 1.0, 1.0), ""),
                page_png=buffer.getvalue(), categories=categories,
                ocr_prompt="ocr", page_no=0, chart_number=1,
                endpoint="unused", model=".", timeout=1, retries=0,
                raw_output_dir=None, padding=0.0,
            )
        result = spatial_time_chart_table(labels, "bar and line")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("| 1Q25 | 83,8 | — |", result)
        self.assertIn("| 3Q25 | 39,1 | -18,6% |", result)
        self.assertIn("| 2Q26 | 117,9 | 23,3% |", result)

    def test_category_band_keeps_only_sequential_two_prompt_consensus(self):
        image = Image.new("RGB", (1200, 800), "white")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        categories = ["1Q26", "2Q26", "3Q26"]
        primary = ["100", "9", "300"]
        verification = ["100", "200", "300"]
        calls = []

        def fake_request_ocr(**kwargs):
            calls.append(kwargs["pass_name"])
            index = int(kwargs["pass_name"].split("_band_", 1)[1].split(".", 1)[0]) - 1
            values = primary if kwargs["prompt"].startswith("labels") else verification
            return f"{categories[index]}\n{values[index]}"

        with patch("pdf_to_md_vllm.request_ocr", side_effect=fake_request_ocr):
            labels = read_chart_category_bands(
                target=Segment(0, "Chart", (0.0, 0.0, 1.0, 1.0), ""),
                page_png=buffer.getvalue(), categories=categories,
                ocr_prompt="labels", verification_prompt="plain",
                page_no=0, chart_number=1, endpoint="unused", model=".",
                timeout=1, retries=0, raw_output_dir=None, padding=0.0,
            )
        result = spatial_time_chart_table(labels, "bar")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("| 1Q26 | 100 |", result)
        self.assertIn("| 2Q26 | — |", result)
        self.assertIn("| 3Q26 | 300 |", result)
        self.assertEqual(
            [
                "focused_chart_1_band_1.labels", "focused_chart_1_band_1.contrast",
                "focused_chart_1_band_2.labels", "focused_chart_1_band_2.contrast",
                "focused_chart_1_band_3.labels", "focused_chart_1_band_3.contrast",
            ],
            calls,
        )

    def test_spatial_categorical_chart_pairs_grouped_values_and_categories(self):
        labels = []
        categories = ["Bất động sản", "Công nghiệp", "Khác"]
        values = [("79,2", "135,3"), ("33,0", "61,4"), ("15,2", "19,0")]
        for index, (category, pair) in enumerate(zip(categories, values)):
            x = 0.18 + index * 0.30
            labels.extend([
                Segment(index, "Text", (x, 0.68, x + 0.10, 0.73), category),
                Segment(index + 10, "Text", (x - 0.035, 0.45, x + 0.01, 0.49), pair[0]),
                Segment(index + 20, "Text", (x + 0.055, 0.42, x + 0.10, 0.46), pair[1]),
            ])
        result = spatial_categorical_chart_table(
            labels, "Doanh thu theo nhóm — 6T2026 vs 6T2025"
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("| Hạng mục | Chuỗi 1 | Chuỗi 2 |", result)
        self.assertIn("| Công nghiệp | 33,0 | 61,4 |", result)

    def test_checks_composition_percentage_not_amount_column(self):
        chart = """| Kênh | Doanh thu | Tỷ trọng |
| :--- | ---: | ---: |
| ETC | 371,5 | 55,8% |
| OTC | 287,7 | 43,2% |
| Khác | 7,0 | 1,1% |"""
        self.assertNotIn(
            "composition percentages do not reconcile to 100",
            chart_quality_issues(chart, "Cơ cấu doanh thu theo kênh"),
        )

    def test_two_independent_reads_must_agree(self):
        second = GOOD_CHART.replace("### Biểu đồ: Doanh thu", "### Biểu đồ: Revenue")
        very_different = GOOD_CHART.replace("100", "900").replace("120", "800")
        self.assertTrue(chart_reads_agree(GOOD_CHART, second))
        self.assertFalse(chart_reads_agree(GOOD_CHART, very_different))

    def test_same_number_set_with_swapped_categories_does_not_agree(self):
        first = """| Kỳ | Giá trị |
| :--- | ---: |
| A | 10 |
| B | 20 |"""
        swapped = """| Kỳ | Giá trị |
| :--- | ---: |
| A | 20 |
| B | 10 |"""
        self.assertFalse(chart_reads_agree(first, swapped))

    def test_strict_mode_keeps_single_read_as_unconfirmed_raw_data(self):
        target = Segment(0, "Chart", (0.1, 0.2, 0.9, 0.6), GOOD_CHART, "pass1")
        result = choose_confirmed_chart(target, None, None, "Doanh thu", "strict")
        self.assertTrue(result.source.startswith("uncertain:"))
        self.assertIn("Dữ liệu OCR thô", result.text)
        self.assertIn("| Q1/2026 | 100 |", result.text)

    def test_strict_mode_accepts_agreeing_direct_reads(self):
        target = Segment(0, "Chart", (0.1, 0.2, 0.9, 0.6), GOOD_CHART, "pass1")
        full = Segment(2, "Chart", (0.1, 0.2, 0.9, 0.6), GOOD_CHART, "pass2")
        result = choose_confirmed_chart(target, full, None, "Doanh thu", "strict")
        self.assertFalse(result.source.startswith("uncertain:"))
        self.assertIn("| Q2/2026 | 120 |", result.text)

    def test_strict_mode_rejects_single_position_bound_read(self):
        target = Segment(0, "Chart", (0.1, 0.2, 0.9, 0.6), "", "pass1")
        spatial = Segment(
            2, "Chart", target.bbox, GOOD_CHART, "focused_chart_spatial"
        )
        result = choose_confirmed_chart(
            target, None, spatial, "Doanh thu", "strict"
        )
        self.assertTrue(result.source.startswith("uncertain:"))

    def test_strict_mode_accepts_complete_per_category_consensus(self):
        target = Segment(0, "Chart", (0.1, 0.2, 0.9, 0.6), "", "pass1")
        chart_ir = chart_ir_from_markdown(GOOD_CHART, "band_consensus")
        assert chart_ir is not None
        spatial = Segment(
            2, "Chart", target.bbox, GOOD_CHART,
            "focused_chart_spatial_consensus", chart_ir,
        )
        result = choose_confirmed_chart(
            target, None, spatial, "Doanh thu", "strict"
        )
        self.assertEqual("focused_chart_spatial_verified", result.source)
        self.assertIn("| Q1/2026 | 100 |", result.text)

    def test_single_mode_still_requires_structural_evidence(self):
        target = Segment(0, "Chart", (0.1, 0.2, 0.9, 0.6), "100 120", "pass1")
        result = choose_confirmed_chart(target, None, None, "", "single")
        self.assertTrue(result.source.startswith("uncertain:"))

    def test_single_mode_does_not_prefer_an_invalid_verification_read(self):
        target = Segment(0, "Chart", (0.1, 0.2, 0.9, 0.6), "bad", "pass1")
        full = Segment(1, "Chart", target.bbox, GOOD_CHART, "pass2")
        verification = Segment(2, "Chart", target.bbox, "also bad", "pass2_verify")
        result = choose_confirmed_chart(target, full, verification, "", "single")
        self.assertEqual("pass2", result.source)
        self.assertIn("| Q1/2026 | 100 |", result.text)


class GenericRoutingTests(unittest.TestCase):
    def test_chart_detection_uses_class_and_local_context(self):
        picture = Segment(0, "Picture", (0.1, 0.3, 0.5, 0.6), "", "pass1")
        caption = Segment(1, "Caption", (0.1, 0.25, 0.5, 0.29), "Biểu đồ tăng trưởng", "pass1")
        targets, _ = chart_target_indexes([picture, caption])
        self.assertEqual({0}, targets)

    def test_decorative_picture_does_not_inherit_distant_chart_words(self):
        picture = Segment(0, "Picture", (0.75, 0.10, 0.82, 0.18), "", "pass1")
        distant = Segment(
            1,
            "Text",
            (0.10, 0.70, 0.90, 0.80),
            "Phân tích biểu đồ và xu hướng ở một vùng khác trên trang.",
            "pass1",
        )
        targets, _ = chart_target_indexes([picture, distant])
        self.assertEqual(set(), targets)

    def test_decorative_chart_class_without_data_or_caption_is_rejected(self):
        logo = Segment(0, "Chart", (0.7, 0.02, 0.95, 0.10), "CLICK", "pass1")
        targets, _ = chart_target_indexes([logo])
        self.assertEqual(set(), targets)

    def test_chart_misclassified_as_table_is_detected_from_local_caption(self):
        table = Segment(0, "Table", (0.1, 0.3, 0.5, 0.6), GOOD_CHART, "pass1")
        caption = Segment(1, "Caption", (0.1, 0.25, 0.5, 0.29), "Biểu đồ tăng trưởng", "pass1")
        targets, _ = chart_target_indexes([table, caption])
        self.assertEqual({0}, targets)

    def test_metric_card_table_is_not_misclassified_as_chart(self):
        card = Segment(
            0, "Table", (0.1, 0.2, 0.9, 0.35),
            """| Chỉ tiêu | Giá trị | Chi tiết |
| :--- | ---: | :--- |
| TĂNG TRƯỞNG TÍN DỤNG | 10% | kế hoạch |""",
            "text_layer_metric_cards",
        )
        prose = Segment(
            1, "Text", (0.1, 0.36, 0.9, 0.5),
            "Đề án tái cơ cấu có nhiều thay đổi.", "pass1",
        )
        targets, _ = chart_target_indexes([card, prose])
        self.assertEqual(set(), targets)

    def test_report_table_near_metric_heading_is_not_a_chart(self):
        table = Segment(
            0, "Table", (0.1, 0.3, 0.9, 0.7),
            """| Chỉ tiêu (tỷ đồng) | Q1 | Q2 |
| :--- | ---: | ---: |
| Doanh thu | 100 | 120 |""", "pass1",
        )
        heading = Segment(
            1, "Section-Header", (0.1, 0.24, 0.9, 0.28),
            "BIÊN LỢI NHUẬN GỘP", "pass1",
        )
        targets, _ = chart_target_indexes([table, heading])
        self.assertEqual(set(), targets)

    def test_picture_with_local_valuation_title_is_a_chart(self):
        title = Segment(0, "Caption", (0.1, 0.2, 0.45, 0.24), "Định giá theo quý", "pass1")
        picture = Segment(1, "Picture", (0.1, 0.25, 0.45, 0.55), "", "pass1")
        targets, _ = chart_target_indexes([title, picture])
        self.assertEqual({1}, targets)

    def test_pass2_matching_uses_geometry_then_reading_order(self):
        first = Segment(0, "Chart", (0.1, 0.2, 0.4, 0.5), GOOD_CHART, "pass1")
        second = Segment(1, "Chart", (0.6, 0.2, 0.9, 0.5), GOOD_CHART, "pass1")
        read_right = Segment(0, "Chart", (0.6, 0.2, 0.9, 0.5), GOOD_CHART, "pass2")
        read_left = Segment(1, "Chart", (0.1, 0.2, 0.4, 0.5), GOOD_CHART, "pass2")
        matches = match_pass2_charts([first, second], {0, 1}, [read_right, read_left])
        self.assertEqual((0.1, 0.2, 0.4, 0.5), matches[0].bbox)
        self.assertEqual((0.6, 0.2, 0.9, 0.5), matches[1].bbox)

    def test_single_chart_keeps_unstructured_ocr_output(self):
        target = Segment(0, "Chart", (0.1, 0.2, 0.9, 0.6), "", "pass1")
        raw = "Doanh thu theo quý\nQ1/2026: 100\nQ2/2026: 120"
        matches = match_chart_ocr_output([target], {0}, raw, "pass2")
        self.assertIn(0, matches)
        self.assertEqual("pass2_raw", matches[0].source)
        self.assertIn("Q2/2026: 120", matches[0].text)

    def test_uncertain_chart_uses_native_region_text_as_raw_fallback(self):
        target = Segment(0, "Chart", (0.1, 0.2, 0.9, 0.6), "", "pass1")
        native = Segment(
            0, "Chart", target.bbox,
            "Doanh thu theo quý\nQ1/2026 100\nQ2/2026 120",
            "text_layer_chart_raw",
        )
        result = choose_confirmed_chart(
            target, None, None, "Doanh thu", "strict", [native]
        )
        self.assertTrue(result.source.startswith("uncertain:"))
        self.assertIn("chart_raw_source=text_layer_chart_raw", result.text)
        self.assertIn("Q2/2026 120", result.text)

    def test_truncated_text_only_page_uses_complete_native_text(self):
        segments = [Segment(0, "Text", (0, 0, 1, 1), "Dòng đầu", "pass1")]
        page = PdfTextPage(
            100,
            100,
            tuple(
                PdfTextLine((0, i, 90, i + 1), f"Dòng nội dung đầy đủ số {i}")
                for i in range(20)
            ),
        )
        result, replaced = use_full_native_text_if_ocr_truncated(segments, page)
        self.assertTrue(replaced)
        self.assertEqual("native_page_fallback", result[0].source)

    def test_native_fallback_never_overwrites_page_with_a_chart(self):
        segments = [Segment(0, "Chart", (0, 0, 1, 1), GOOD_CHART, "pass1")]
        page = PdfTextPage(100, 100, (PdfTextLine((0, 0, 90, 1), "Rất nhiều text"),))
        result, replaced = use_full_native_text_if_ocr_truncated(segments, page)
        self.assertFalse(replaced)
        self.assertEqual(segments, result)

    def test_recovers_prose_omitted_below_chart_on_mixed_page(self):
        chart = Segment(0, "Chart", (0.24, 0.29, 0.50, 0.34), "", "pass1")
        page = PdfTextPage(
            1000, 1000,
            (
                PdfTextLine((100, 100, 400, 120), "Nhãn nằm trong biểu đồ"),
                PdfTextLine((100, 500, 900, 520), "2.5. Bảng cân đối và thanh khoản"),
                PdfTextLine((100, 530, 900, 550), "Tổng tài sản tăng trong quý."),
            ),
            1000, 1000,
        )
        result, count = recover_missing_native_text([chart], page)
        recovered = "\n".join(segment.text for segment in result)
        self.assertGreaterEqual(count, 1)
        self.assertIn("Tổng tài sản tăng", recovered)
        self.assertNotIn("Nhãn nằm trong biểu đồ", recovered)

    def test_recovers_long_prose_even_when_model_chart_bbox_is_too_large(self):
        chart = Segment(0, "Chart", (0.05, 0.05, 0.95, 0.90), "", "pass1")
        page = PdfTextPage(
            1000, 1000,
            (
                PdfTextLine((100, 100, 300, 120), "2025 10 20"),
                PdfTextLine((100, 500, 900, 520), "2.5. Experimental results and further detailed discussion"),
                PdfTextLine((100, 530, 900, 550), "This paragraph contains enough prose words that must never disappear from output."),
            ),
            1000, 1000,
        )
        result, count = recover_missing_native_text([chart], page)
        recovered = "\n".join(segment.text for segment in result)
        self.assertGreaterEqual(count, 1)
        self.assertIn("must never disappear", recovered)
        self.assertNotIn("2025 10 20", recovered)
        self.assertGreaterEqual(native_text_coverage(result, page), 0.80)

    def test_does_not_explode_truncated_table_cells_into_loose_text(self):
        table = Segment(
            0, "Table", (0.2, 0.2, 0.8, 0.8),
            "| Chỉ tiêu | 2025 |\n| :--- | ---: |\n| Doanh thu | 100 |",
            "pass1",
        )
        page = PdfTextPage(
            1000, 1000,
            (
                PdfTextLine((100, 200, 900, 220), "Chỉ tiêu 2025"),
                PdfTextLine((100, 240, 900, 260), "Doanh thu 100"),
                PdfTextLine((100, 280, 900, 300), "Lợi nhuận 20"),
            ),
            1000, 1000,
        )
        result, count = recover_missing_native_text([table], page)
        recovered = "\n".join(segment.text for segment in result)
        self.assertEqual(0, count)
        self.assertNotIn("Lợi nhuận 20", recovered)

    def test_native_text_layer_replaces_position_matched_table(self):
        segment = Segment(
            0,
            "Table",
            (0.1, 0.1, 0.9, 0.5),
            """| Chỉ tiêu | Q1/2026 | Q2/2026 |
| --- | ---: | ---: |
| Thu nhập lăi thuãn | 100 | 120 |""",
            "pass1",
        )
        page = PdfTextPage(
            1664,
            2048,
            (
                PdfTextLine((200, 250, 500, 270), "Chỉ tiêu"),
                PdfTextLine((700, 250, 800, 270), "Q1/2026"),
                PdfTextLine((1100, 250, 1200, 270), "Q2/2026"),
                PdfTextLine((200, 350, 550, 370), "Thu nhập lãi thuần"),
                PdfTextLine((700, 350, 800, 370), "100"),
                PdfTextLine((1100, 350, 1200, 370), "120"),
            ),
            1664,
            2048,
        )
        result, count = reconcile_tables_with_text_layer([segment], page)
        self.assertEqual(1, count)
        self.assertIn("Thu nhập lãi thuần", result[0].text)
        self.assertNotIn("lăi thuãn", result[0].text)

    def test_native_wide_table_keeps_empty_cells_and_stops_before_chart(self):
        page = PdfTextPage(
            1000,
            1000,
            (
                PdfTextLine((100, 100, 300, 120), "Chỉ tiêu"),
                PdfTextLine((400, 100, 470, 120), "2024"),
                PdfTextLine((600, 100, 670, 120), "2025"),
                PdfTextLine((800, 100, 870, 120), "2026"),
                PdfTextLine((100, 125, 300, 145), "Doanh thu"),
                PdfTextLine((400, 125, 470, 145), "100"),
                PdfTextLine((600, 125, 670, 145), "120"),
                PdfTextLine((800, 125, 870, 145), "140"),
                PdfTextLine((100, 150, 300, 170), "Tăng trưởng"),
                PdfTextLine((600, 150, 670, 170), "20%"),
                PdfTextLine((800, 150, 870, 170), "16,7%"),
                PdfTextLine((100, 175, 430, 195), "Doanh thu & biên lợi nhuận"),
                PdfTextLine((550, 175, 900, 195), "Lợi nhuận theo quý"),
            ),
            1000,
            1000,
        )
        tables = native_wide_table_segments(page)
        self.assertEqual(1, len(tables))
        self.assertIn("| Tăng trưởng |  | 20% | 16,7% |", tables[0].text)
        self.assertNotIn("Lợi nhuận theo quý", tables[0].text)

        fragmented = [
            Segment(0, "Text", tables[0].bbox, "Chỉ tiêu", "pass1"),
            Segment(1, "Text", tables[0].bbox, "Doanh thu 100 120 140", "pass1"),
        ]
        merged, count = coalesce_native_wide_tables(fragmented, page)
        self.assertEqual(1, count)
        self.assertEqual(1, len(merged))
        self.assertEqual("Table", merged[0].cls)

    def test_native_wide_table_uses_geometry_for_non_financial_data(self):
        page = PdfTextPage(
            1000,
            1000,
            (
                PdfTextLine((80, 100, 280, 120), "Sample"),
                PdfTextLine((500, 100, 570, 120), "Run A"),
                PdfTextLine((750, 100, 820, 120), "Run B"),
                PdfTextLine((80, 125, 280, 145), "Material X"),
                PdfTextLine((500, 125, 570, 145), "12.4"),
                PdfTextLine((750, 125, 820, 145), "12.8"),
                PdfTextLine((80, 150, 280, 170), "Material Y"),
                PdfTextLine((500, 150, 570, 170), "9.1"),
                PdfTextLine((750, 150, 820, 170), "9.3"),
            ),
        )
        tables = native_wide_table_segments(page)
        self.assertEqual(1, len(tables))
        self.assertIn("| Material Y | 9.1 | 9.3 |", tables[0].text)

    def test_document_reconciliation_completes_a_truncated_native_table(self):
        page = PdfTextPage(
            1000,
            1000,
            (
                PdfTextLine((100, 100, 300, 120), "Chỉ tiêu"),
                PdfTextLine((400, 100, 470, 120), "2024"),
                PdfTextLine((600, 100, 670, 120), "2025"),
                PdfTextLine((800, 100, 870, 120), "2026"),
                PdfTextLine((100, 125, 300, 145), "Doanh thu"),
                PdfTextLine((400, 125, 470, 145), "100"),
                PdfTextLine((600, 125, 670, 145), "120"),
                PdfTextLine((800, 125, 870, 145), "140"),
                PdfTextLine((100, 150, 300, 170), "Lợi nhuận"),
                PdfTextLine((400, 150, 470, 170), "10"),
                PdfTextLine((600, 150, 670, 170), "20"),
                PdfTextLine((800, 150, 870, 170), "30"),
            ),
        )
        truncated = """# report

<!-- page 1 -->

| Chỉ tiêu | 2024 | 2025 | 2026 |
| :--- | ---: | ---: | ---: |
| Doanh thu | 100 | 120 | 140 |
"""
        with patch("pdf_to_md_vllm.extract_pdf_text_page", return_value=page):
            repaired, count = reconcile_document_tables_with_pdf(truncated, "sample.pdf")
        self.assertEqual(1, count)
        self.assertIn("| Lợi nhuận | 10 | 20 | 30 |", repaired)

    def test_text_heavy_infographic_is_not_left_as_chart(self):
        chart = Segment(0, "Chart", (0.1, 0.1, 0.9, 0.7), "ĐIỂM NHẤN", "pass1")
        page = PdfTextPage(
            1664, 2048,
            (
                PdfTextLine((200, 300, 1200, 330), "ĐIỂM NHẤN CẦN THEO DÕI"),
                PdfTextLine((200, 380, 1200, 410), "Kết quả hoạt động cải thiện nhờ kiểm soát tốt chi phí"),
                PdfTextLine((200, 450, 1200, 480), "Nhà đầu tư cần theo dõi tốc độ phục hồi trong kỳ tới"),
            ),
            1664, 2048,
        )
        result, count = recover_misclassified_visuals([chart], page)
        self.assertEqual(1, count)
        self.assertEqual("Text", result[0].cls)
        self.assertIn("Kết quả hoạt động", result[0].text)

    def test_recovers_omitted_side_by_side_charts_from_native_geometry(self):
        page = PdfTextPage(
            1000, 1000,
            (
                PdfTextLine((50, 200, 400, 214), "Doanh thu theo nhóm"),
                PdfTextLine((550, 200, 850, 214), "Lãi gộp theo mảng"),
                PdfTextLine((50, 340, 350, 350), "Nguồn: Báo cáo công ty"),
                PdfTextLine((550, 340, 850, 350), "Nguồn: Báo cáo công ty"),
            ),
            1000, 1000,
        )
        result, count = infer_native_chart_segments([], page)
        self.assertEqual(2, count)
        self.assertEqual(["Chart", "Chart"], [segment.cls for segment in result])
        self.assertLess(result[0].bbox[2], result[1].bbox[0])

    def test_does_not_infer_chart_from_full_width_table_source(self):
        page = PdfTextPage(
            1000, 1000,
            (
                PdfTextLine((50, 200, 400, 214), "Bảng cân đối kế toán"),
                PdfTextLine((50, 500, 950, 512), "Nguồn: Báo cáo tài chính và tính toán"),
            ),
            1000, 1000,
        )
        result, count = infer_native_chart_segments([], page)
        self.assertEqual(([], 0), (result, count))

    def test_attaches_only_local_caption_to_uncertain_chart(self):
        chart = Segment(0, "Chart", (0.1, 0.3, 0.45, 0.6), "", "pass1")
        caption = Segment(1, "Caption", (0.1, 0.24, 0.45, 0.28), "Biểu đồ doanh thu", "pass1")
        other = Segment(2, "Caption", (0.55, 0.24, 0.9, 0.28), "Biểu đồ lợi nhuận", "pass1")
        replacement = choose_confirmed_chart(chart, None, None, "", "strict")
        segments, replacements = attach_chart_captions(
            [chart, caption, other], {0: replacement}
        )
        self.assertEqual(2, len(segments))
        self.assertIn("Biểu đồ doanh thu", replacements[0].text)
        self.assertNotIn("Biểu đồ lợi nhuận", replacements[0].text)

    def test_numbered_section_heading_is_not_stolen_as_chart_title(self):
        chart = Segment(0, "Chart", (0.1, 0.3, 0.8, 0.6), "", "pass1")
        heading = Segment(
            1, "Section-Header", (0.1, 0.24, 0.8, 0.28),
            "3.2. Nguồn vốn, chi phí vốn & biên lãi ròng", "pass1",
        )
        replacement = choose_confirmed_chart(chart, None, None, "", "strict")
        _, replacements = attach_chart_captions([chart, heading], {0: replacement})
        self.assertNotIn("3.2.", replacements[0].text)

    def test_uses_exact_native_chart_caption(self):
        page = PdfTextPage(
            1000, 1000,
            (
                PdfTextLine((100, 100, 500, 120), "Doanh thu thuần theo quý"),
                PdfTextLine((100, 125, 500, 140), "Tỷ đồng · 8 quý gần nhất"),
            ),
            1000, 1000,
        )
        chart = Segment(0, "Chart", (0.30, 0.34, 0.65, 0.60), "", "pass1")
        self.assertEqual(
            ["Doanh thu thuần theo quý", "Tỷ đồng · 8 quý gần nhất"],
            native_chart_caption(chart, page),
        )

    def test_coalesces_horizontal_metric_cards_from_native_text(self):
        page = PdfTextPage(
            1000,
            1000,
            (
                PdfTextLine((100, 100, 250, 120), "DOANH THU"),
                PdfTextLine((400, 100, 550, 120), "LỢI NHUẬN"),
                PdfTextLine((700, 100, 850, 120), "BIÊN GỘP"),
                PdfTextLine((100, 125, 250, 145), "100"),
                PdfTextLine((400, 125, 550, 145), "20"),
                PdfTextLine((700, 125, 850, 145), "42%"),
                PdfTextLine((100, 150, 250, 170), "+10% YoY"),
                PdfTextLine((400, 150, 550, 170), "+5% YoY"),
                PdfTextLine((700, 150, 850, 170), "+1 điểm %"),
                PdfTextLine((700, 174, 760, 184), "QoQ"),
            ),
            1000,
            1000,
        )
        cards = native_metric_card_segments(page)
        self.assertEqual(1, len(cards))
        self.assertIn("| DOANH THU | 100 | +10% YoY |", cards[0].text)
        self.assertIn("| BIÊN GỘP | 42% | +1 điểm % QoQ |", cards[0].text)
        merged, replacements = coalesce_metric_cards([], page)
        self.assertEqual(1, replacements)
        self.assertEqual(cards[0].text, merged[0].text)
        false_chart = Segment(0, "Chart", cards[0].bbox, "DOANH THU", "pass1")
        false_heading = Segment(1, "Section-Header", cards[0].bbox, "DOANH THU", "pass1")
        merged, replacements = coalesce_metric_cards([false_chart, false_heading], page)
        self.assertEqual(1, replacements)
        self.assertEqual(1, len(merged))
        self.assertEqual("Table", merged[0].cls)

    def test_merges_vertically_adjacent_metric_card_rows(self):
        lines = []
        for row_offset, prefix in ((100, "TOP"), (180, "BOTTOM")):
            for column, x in enumerate((100, 400, 700), start=1):
                lines.extend([
                    PdfTextLine((x, row_offset, x + 150, row_offset + 20), f"{prefix} LABEL {column}"),
                    PdfTextLine((x, row_offset + 25, x + 150, row_offset + 45), str(column * 10)),
                    PdfTextLine((x, row_offset + 50, x + 150, row_offset + 70), f"detail {column}"),
                ])
        page = PdfTextPage(
            1000, 1000,
            tuple(sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))),
            1000, 1000,
        )
        cards = native_metric_card_segments(page)
        self.assertEqual(1, len(cards))
        self.assertIn("| TOP LABEL 1 | 10 | detail 1 |", cards[0].text)
        self.assertIn("| BOTTOM LABEL 3 | 30 | detail 3 |", cards[0].text)


class PagePipelineTests(unittest.TestCase):
    @staticmethod
    def raw_chart(text: str) -> str:
        return f"<x_0.1><y_0.1>{text}<x_0.9><y_0.9><class_Chart>"

    def run_page(self, reads: list[str], chart_mode: str = "auto"):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "sample.pdf"
            Image.new("RGB", (200, 200), "white").save(pdf_path, "PDF")
            with (
                patch("pdf_to_md_vllm.request_ocr", side_effect=reads),
                patch("pdf_to_md_vllm.read_chart_category_bands", return_value=[]),
            ):
                return process_page(
                    pdf_path=str(pdf_path),
                    page_no=0,
                    endpoint="unused",
                    model=".",
                    zoom=1.0,
                    max_tokens=100,
                    chart_max_tokens=100,
                    timeout=1,
                    text_prompt="text",
                    chart_prompt="chart",
                    profile_prompt=None,
                    profile="auto",
                    raw_output_dir=None,
                    chart_mode=chart_mode,
                    retries=0,
                    focused_retry=True,
                    chart_crop_zoom=1.0,
                    chart_crop_padding=0.0,
                    use_text_layer=False,
                    chart_confirmation="strict",
                )

    def test_end_to_end_page_confirms_two_matching_reads(self):
        raw = self.raw_chart(GOOD_CHART)
        result = self.run_page([raw, raw, raw, raw])
        self.assertEqual(1, result.confirmed_charts)
        self.assertEqual(0, result.uncertain_charts)
        self.assertIn("status=confirmed", result.markdown)

    def test_end_to_end_page_withholds_disagreeing_reads(self):
        reads = [
            self.raw_chart(GOOD_CHART),
            self.raw_chart(GOOD_CHART.replace("100", "900").replace("120", "800")),
            self.raw_chart(GOOD_CHART.replace("100", "700").replace("120", "600")),
            self.raw_chart(GOOD_CHART),
        ]
        result = self.run_page(reads)
        self.assertEqual(0, result.confirmed_charts)
        self.assertEqual(1, result.uncertain_charts)
        self.assertIn("Dữ liệu OCR thô", result.markdown)
        self.assertIn("| Q1/2026 | 100 |", result.markdown)
        self.assertNotIn("![", result.markdown)

    def test_chart_mode_never_makes_no_hidden_crop_request(self):
        raw = self.raw_chart(GOOD_CHART)
        result = self.run_page([raw], chart_mode="never")
        self.assertEqual(0, result.confirmed_charts)
        self.assertEqual(1, result.uncertain_charts)

    def test_unrouted_logo_is_not_serialized_as_a_chart(self):
        raw = self.raw_chart("CLICK")
        result = self.run_page([raw])
        self.assertEqual(0, result.chart_targets)
        self.assertNotIn("chart_region", result.markdown)


if __name__ == "__main__":
    unittest.main()

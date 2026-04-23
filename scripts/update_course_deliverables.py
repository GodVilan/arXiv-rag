from __future__ import annotations

import argparse
import copy
import csv
import struct
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


DOC_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}
PPT_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}
REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
IMAGE_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
EMU_PER_INCH = 914400


def register_namespaces() -> None:
    ET.register_namespace("w", DOC_NS["w"])
    ET.register_namespace("a", DOC_NS["a"])
    ET.register_namespace("pic", DOC_NS["pic"])
    ET.register_namespace("r", DOC_NS["r"])
    ET.register_namespace("wp", DOC_NS["wp"])


def paragraph_text(paragraph: ET.Element) -> str:
    texts = [t.text or "" for t in paragraph.findall(".//w:t", DOC_NS)]
    return "".join(texts).strip()


def cell_text(cell: ET.Element) -> str:
    texts = [t.text or "" for t in cell.findall(".//w:t", DOC_NS)]
    return "".join(texts).strip()


def set_text_nodes(nodes: list[ET.Element], text: str) -> None:
    if not nodes:
        return
    nodes[0].text = text
    for node in nodes[1:]:
        node.text = ""


def set_paragraph_text(paragraph: ET.Element, text: str) -> None:
    nodes = paragraph.findall(".//w:t", DOC_NS)
    set_text_nodes(nodes, text)


def set_cell_text(cell: ET.Element, text: str) -> None:
    nodes = cell.findall(".//w:t", DOC_NS)
    set_text_nodes(nodes, text)


def slide_text_nodes(root: ET.Element) -> list[ET.Element]:
    return root.findall(".//a:t", PPT_NS)


def replace_slide_text(root: ET.Element, replacements: dict[str, str]) -> None:
    for node in slide_text_nodes(root):
        current = node.text or ""
        if current in replacements:
            node.text = replacements[current]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_metric(value: str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_latency_ms(value: str) -> str:
    return f"{float(value):.1f}ms"


def replace_media_files(media_dir: Path, mapping: dict[str, Path]) -> None:
    for relative_name, source_path in mapping.items():
        shutil.copyfile(source_path, media_dir / relative_name)


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Unsupported image format for {path}")
    return struct.unpack(">II", header[16:24])


def next_numeric_suffix(values: list[str], prefix: str) -> int:
    suffixes = []
    for value in values:
        if value.startswith(prefix):
            try:
                suffixes.append(int(value[len(prefix) :]))
            except ValueError:
                continue
    return (max(suffixes) + 1) if suffixes else 1


def clone_paragraph_with_text(source: ET.Element, text: str) -> ET.Element:
    paragraph = copy.deepcopy(source)
    for child in list(paragraph):
        if child.tag != f"{{{DOC_NS['w']}}}pPr":
            paragraph.remove(child)
    run = ET.SubElement(paragraph, f"{{{DOC_NS['w']}}}r")
    text_node = ET.SubElement(run, f"{{{DOC_NS['w']}}}t")
    if text.startswith(" ") or text.endswith(" "):
        text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = text
    return paragraph


def build_inline_image_paragraph(image_name: str, rel_id: str, docpr_id: int, width_px: int, height_px: int) -> ET.Element:
    max_width_emu = int(6.2 * EMU_PER_INCH)
    aspect_ratio = height_px / width_px
    width_emu = max_width_emu
    height_emu = int(width_emu * aspect_ratio)

    paragraph = ET.Element(f"{{{DOC_NS['w']}}}p")
    run = ET.SubElement(paragraph, f"{{{DOC_NS['w']}}}r")
    drawing = ET.SubElement(run, f"{{{DOC_NS['w']}}}drawing")
    inline = ET.SubElement(drawing, f"{{{DOC_NS['wp']}}}inline")
    ET.SubElement(inline, f"{{{DOC_NS['wp']}}}extent", {"cx": str(width_emu), "cy": str(height_emu)})
    ET.SubElement(inline, f"{{{DOC_NS['wp']}}}docPr", {"id": str(docpr_id), "name": image_name})
    c_nv = ET.SubElement(inline, f"{{{DOC_NS['wp']}}}cNvGraphicFramePr")
    ET.SubElement(c_nv, f"{{{DOC_NS['a']}}}graphicFrameLocks", {"noChangeAspect": "1"})
    graphic = ET.SubElement(inline, f"{{{DOC_NS['a']}}}graphic")
    graphic_data = ET.SubElement(
        graphic,
        f"{{{DOC_NS['a']}}}graphicData",
        {"uri": "http://schemas.openxmlformats.org/drawingml/2006/picture"},
    )
    pic = ET.SubElement(graphic_data, f"{{{DOC_NS['pic']}}}pic")
    nv_pic_pr = ET.SubElement(pic, f"{{{DOC_NS['pic']}}}nvPicPr")
    ET.SubElement(nv_pic_pr, f"{{{DOC_NS['pic']}}}cNvPr", {"id": "0", "name": image_name})
    ET.SubElement(nv_pic_pr, f"{{{DOC_NS['pic']}}}cNvPicPr")
    blip_fill = ET.SubElement(pic, f"{{{DOC_NS['pic']}}}blipFill")
    ET.SubElement(blip_fill, f"{{{DOC_NS['a']}}}blip", {f"{{{DOC_NS['r']}}}embed": rel_id})
    stretch = ET.SubElement(blip_fill, f"{{{DOC_NS['a']}}}stretch")
    ET.SubElement(stretch, f"{{{DOC_NS['a']}}}fillRect")
    sp_pr = ET.SubElement(pic, f"{{{DOC_NS['pic']}}}spPr")
    xfrm = ET.SubElement(sp_pr, f"{{{DOC_NS['a']}}}xfrm")
    ET.SubElement(xfrm, f"{{{DOC_NS['a']}}}off", {"x": "0", "y": "0"})
    ET.SubElement(xfrm, f"{{{DOC_NS['a']}}}ext", {"cx": str(width_emu), "cy": str(height_emu)})
    prst = ET.SubElement(sp_pr, f"{{{DOC_NS['a']}}}prstGeom", {"prst": "rect"})
    ET.SubElement(prst, f"{{{DOC_NS['a']}}}avLst")
    return paragraph


def insert_report_screenshots(work_dir: Path, root: ET.Element) -> None:
    if any(paragraph_text(p) == "6.3 Application Screenshots" for p in root.findall(".//w:p", DOC_NS)):
        return

    screenshots = [
        (
            Path("docs/screenshot_welcome.png"),
            "Figure 9. Streamlit welcome screen with model selection, paper browser, and benchmark summary.",
        ),
        (
            Path("docs/screenshot_qa.png"),
            "Figure 10. Streamlit QA workflow showing retrieved context, cited sources, and grounded answer generation.",
        ),
    ]

    for image_path, _ in screenshots:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing screenshot asset: {image_path}")

    body = root.find(".//w:body", DOC_NS)
    if body is None:
        raise ValueError("Could not locate report body")

    heading_source = None
    body_children = list(body)
    insert_index = None
    for index, child in enumerate(body_children):
        if child.tag == f"{{{DOC_NS['w']}}}p":
            text = paragraph_text(child)
            if text == "6.2 Recommendations":
                heading_source = child
            if text == "7. Project Management" and insert_index is None:
                insert_index = index
    if heading_source is None:
        raise ValueError("Could not find heading style source for screenshots section")
    if insert_index is None:
        insert_index = len(body_children)
        if body_children and body_children[-1].tag == f"{{{DOC_NS['w']}}}sectPr":
            insert_index -= 1

    caption_source = next(
        (p for p in root.findall(".//w:p", DOC_NS) if paragraph_text(p).startswith("Figure 1.")),
        None,
    )
    body_source = next(
        (p for p in root.findall(".//w:p", DOC_NS) if paragraph_text(p).startswith("The system is deployed")),
        None,
    )
    if caption_source is None or body_source is None:
        raise ValueError("Could not find paragraph styles for screenshot insertion")

    rels_path = work_dir / "word" / "_rels" / "document.xml.rels"
    rels_tree = ET.parse(rels_path)
    rels_root = rels_tree.getroot()
    next_rel_num = next_numeric_suffix([rel.get("Id", "") for rel in rels_root.findall("./pr:Relationship", REL_NS)], "rId")
    existing_docpr_ids = [
        int(node.get("id"))
        for node in root.findall(".//wp:docPr", DOC_NS)
        if node.get("id", "").isdigit()
    ]
    next_docpr_id = (max(existing_docpr_ids) + 1) if existing_docpr_ids else 1

    media_dir = work_dir / "word" / "media"
    new_elements = [
        clone_paragraph_with_text(heading_source, "6.3 Application Screenshots"),
        clone_paragraph_with_text(
            body_source,
            "Figures 9 and 10 show the deployed Streamlit interface. The welcome screen exposes model selection, paper browsing, and benchmark metadata, while the QA screen demonstrates grounded answer generation with cited source passages and similarity scores.",
        ),
    ]

    for image_path, caption_text in screenshots:
        target_name = image_path.name
        shutil.copyfile(image_path, media_dir / target_name)

        rel_id = f"rId{next_rel_num}"
        next_rel_num += 1
        ET.SubElement(
            rels_root,
            f"{{{REL_NS['pr']}}}Relationship",
            {"Id": rel_id, "Type": IMAGE_REL_TYPE, "Target": f"media/{target_name}"},
        )

        width_px, height_px = png_dimensions(image_path)
        new_elements.append(clone_paragraph_with_text(caption_source, caption_text))
        new_elements.append(
            build_inline_image_paragraph(
                image_name=target_name,
                rel_id=rel_id,
                docpr_id=next_docpr_id,
                width_px=width_px,
                height_px=height_px,
            )
        )
        next_docpr_id += 1

    for offset, element in enumerate(new_elements):
        body.insert(insert_index + offset, element)

    rels_tree.write(rels_path, encoding="utf-8", xml_declaration=True)


def update_report(docx_path: Path, output_path: Path, plots_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        with zipfile.ZipFile(docx_path) as archive:
            archive.extractall(work_dir)

        document_path = work_dir / "word" / "document.xml"
        tree = ET.parse(document_path)
        root = tree.getroot()

        paragraph_replacements = {
            (
                "This report presents the complete design, implementation, evaluation, and deployment "
                "of an end-to-end Retrieval-Augmented Generation (RAG) system for domain-specific "
                "question answering over machine learning research papers. The system retrieves "
                "semantically relevant passages from a corpus of 120 arXiv cs.LG papers using FAISS "
                "vector indexing and generates grounded answers via Google Gemini 2.5 Flash Lite. "
                "Three transformer-based embedding models — all-MiniLM-L6-v2, all-mpnet-base-v2, and "
                "BAAI/bge-large-en — are evaluated against a BM25 sparse retrieval baseline using 20 "
                "manually curated question-answer pairs across three chunk sizes (256, 512, 1024 "
                "tokens) and three retrieval depths. BGE achieves perfect Mean Reciprocal Rank "
                "(MRR = 1.000) and the highest Answer Relevance (0.910), outperforming BM25 by 7.7× "
                "on generation quality. MiniLM provides the best speed-accuracy trade-off at 8 ms per "
                "query with MRR = 0.975. The system is deployed as an interactive Streamlit web "
                "application with real-time model switching, source attribution, and paper browsing."
            ): (
                "This report presents the complete design, implementation, evaluation, and deployment "
                "of an end-to-end Retrieval-Augmented Generation (RAG) system for domain-specific "
                "question answering over machine learning research papers. The system retrieves "
                "semantically relevant passages from a corpus of 150 arXiv cs.LG papers using FAISS "
                "vector indexing and generates grounded answers via Google Gemini 2.5 Flash Lite. "
                "Three transformer-based embedding models — all-MiniLM-L6-v2, all-mpnet-base-v2, and "
                "BAAI/bge-large-en — are evaluated against a BM25 sparse retrieval baseline using a "
                "100-question benchmark for retrieval and a 30-question subset for generation across "
                "three chunk sizes (256, 512, 1024 tokens) and three retrieval depths. BGE achieves "
                "the strongest overall dense-retrieval performance at chunk sizes 256 and 512 "
                "(MRR@5 = 0.990) and the highest Answer Relevance (0.912), while BM25 remains highly "
                "competitive at chunk size 1024. MiniLM provides the best speed-accuracy trade-off at "
                "6.8 ms per query with MRR@5 = 0.975. The system is deployed as an interactive "
                "Streamlit web application with real-time model switching, source attribution, and "
                "paper browsing."
            ),
            (
                "The document corpus consists of 120 research papers retrieved from the arXiv "
                "repository using the public arXiv API. Papers were filtered to the cs.LG (Machine "
                "Learning) category and sorted by descending submission date to capture the most "
                "recent literature from early 2026. After PDF text extraction using PyMuPDF and "
                "standard cleaning, the corpus was chunked into three configurations for ablation:"
            ): (
                "The document corpus consists of 150 research papers retrieved from the arXiv "
                "repository using the public arXiv API. Papers were filtered to the cs.LG (Machine "
                "Learning) category and sorted by descending submission date to capture the most "
                "recent literature from early 2026. After PDF text extraction using PyMuPDF and "
                "standard cleaning, the corpus was chunked into three configurations for ablation:"
            ),
            (
                "The evaluation dataset consists of 20 manually curated question-answer pairs. For "
                "each of 20 selected papers, a domain-expert question targeting the paper's core "
                "contribution was generated using a structured Gemini prompt applied to the paper's "
                "title, abstract, and first content chunk. Relevance was defined at the paper level: "
                "all chunks originating from the same paper are considered relevant."
            ): (
                "The evaluation dataset consists of 100 paper-specific question-answer pairs stored in "
                "manual_qa.json. Each question targets a paper's core contribution using information "
                "from the title, abstract, and first content chunk. The full 100-question set is used "
                "for retrieval evaluation, while a 30-question subset is used for generation "
                "evaluation to control API cost. Relevance is defined at the paper level: all chunks "
                "originating from the same paper are considered relevant."
            ),
            "Document Collection — arXiv API downloads 120 paper PDFs with metadata": (
                "Document Collection — arXiv API downloads 150 paper PDFs with metadata"
            ),
            (
                "Table 3 presents complete retrieval results across all models and chunk sizes. BGE "
                "achieved the highest accuracy: perfect MRR (1.000) at chunk sizes 256 and 512, "
                "Precision@5 of 0.995 at chunk 256. All dense models outperformed BM25 at smaller "
                "chunk sizes. At chunk size 1024, BM25 (MRR = 0.975) matched or exceeded MiniLM "
                "(0.953) and MPNet (0.938)."
            ): (
                "Table 3 presents complete retrieval results across all models and chunk sizes using "
                "MRR@5 as the headline rank metric. BGE achieved the strongest dense-retrieval "
                "accuracy at chunk sizes 256 and 512 (MRR@5 = 0.990 in both settings) and the highest "
                "Precision@5 at every chunk size. BM25 remained highly competitive throughout and "
                "became the best 1024-token configuration (MRR@5 = 0.990), showing that sparse "
                "retrieval improves substantially as chunk size grows. MiniLM at chunk 512 delivered "
                "near-BGE accuracy with the lowest dense latency (6.8 ms)."
            ),
            (
                "Figure 1. MRR by Model and Chunk Size. BGE achieves perfect MRR (1.000) at chunk "
                "sizes 256 and 512."
            ): (
                "Figure 1. MRR@5 by Model and Chunk Size. BGE leads the dense models at chunk sizes "
                "256 and 512, while BM25 slightly leads at chunk size 1024."
            ),
            (
                "Figure 2. Precision@5 by Model and Chunk Size. BGE leads at 0.995 (chunk 256) and "
                "0.975 (chunk 512)."
            ): (
                "Figure 2. Precision@5 by Model and Chunk Size. BGE leads at 0.970 (chunk 256), "
                "0.950 (chunk 512), and 0.888 (chunk 1024)."
            ),
            (
                "Figure 3. Recall@5 by Model and Chunk Size. Recall increases consistently with chunk "
                "size across all models."
            ): (
                "Figure 3. Recall@5 by Model and Chunk Size. Recall increases consistently with chunk "
                "size across all models, and BM25 becomes especially competitive at 1024 tokens."
            ),
            (
                "Figure 4. MRR Heat-map (Model × Chunk Size). BGE row is uniformly darkest — highest "
                "MRR across all configurations."
            ): (
                "Figure 4. MRR@5 Heat-map (Model × Chunk Size). BGE dominates the 256- and 512-token "
                "settings, while BM25 is strongest at 1024 tokens."
            ),
            (
                "Figure 5. BM25 Baseline vs Best Dense Model (BGE). BGE improves on BM25 by +5% at "
                "chunk 256, +6% at chunk 512."
            ): (
                "Figure 5. BM25 Baseline vs Best Dense Model. The best dense configuration improves on "
                "BM25 by about +3% at chunk 256 and +1% at chunk 512, while BM25 slightly leads at "
                "chunk 1024."
            ),
            (
                "Table 4 presents generation quality results at the default chunk size of 512 tokens. "
                "BGE Answer Relevance (0.910) substantially exceeded all other models. The 7.7× gap "
                "between BGE and BM25 in Answer Relevance — despite BM25 achieving competitive "
                "Precision@5 (0.900) — is the central finding: high retrieval precision does not "
                "translate to semantic alignment with query intent. Faithfulness was uniformly high "
                "across all dense models (0.974–0.992), confirming Gemini reliably grounds answers in "
                "retrieved context."
            ): (
                "Table 4 presents generation quality results at the default chunk size of 512 tokens "
                "using 30 benchmark questions. BGE achieved the highest Answer Relevance (0.912) while "
                "maintaining Faithfulness (0.989) and perfect Context Precision (1.000). The 7.3× gap "
                "between BGE and BM25 in Answer Relevance — despite BM25 achieving strong retrieval "
                "metrics — reinforces the central finding: high lexical retrieval accuracy does not "
                "guarantee semantic alignment with query intent. Faithfulness remained uniformly high "
                "across all models (0.986–0.997), aided by deterministic generation at temperature "
                "0.0."
            ),
            (
                "Figure 6. Generation Metrics by Model (chunk size 512). Answer Relevance shows the "
                "largest spread — BGE 0.910 vs BM25 0.118. Faithfulness and Context Precision are "
                "near-perfect for all models."
            ): (
                "Figure 6. Generation Metrics by Model (chunk size 512). Answer Relevance shows the "
                "largest spread — BGE 0.912 vs BM25 0.125 — while Faithfulness and Context Precision "
                "remain near-perfect across the board."
            ),
            (
                "BM25 retrieved in 3.8–20.2 ms depending on chunk size. MiniLM at 8.0 ms closely "
                "matched BM25 at chunk 512 while achieving MRR = 0.975. BGE required 26.5–37.2 ms, "
                "4–4.6× slower than BM25."
            ): (
                "BM25 retrieved in 4.6–23.8 ms depending on chunk size. MiniLM at 6.8 ms delivered "
                "near-BGE quality at chunk 512 (MRR@5 = 0.975 vs 0.990) while remaining the fastest "
                "dense model. BGE required 24.9–34.7 ms and provided the best dense-retrieval "
                "accuracy at small and medium chunk sizes."
            ),
            (
                "Figure 7. Retrieval Latency per Query (chunk size 512). MiniLM (8ms) matches BM25 "
                "speed while achieving near-BGE accuracy."
            ): (
                "Figure 7. Retrieval Latency per Query (chunk size 512). MiniLM (6.8 ms) is the "
                "fastest dense retriever while staying close to BGE on MRR@5."
            ),
            (
                "Figure 8. Accuracy vs Speed Trade-off. MiniLM occupies the optimal position — "
                "top-left quadrant (high MRR, low latency). BGE achieves perfect MRR but at 37ms."
            ): (
                "Figure 8. Accuracy vs Speed Trade-off. MiniLM remains closest to the top-left "
                "quadrant (high MRR@5, low latency), while BGE leads accuracy at roughly 31 ms."
            ),
            (
                "Model selection: BGE for highest accuracy; MiniLM for latency-sensitive deployments. "
                "Avoid BM25 as a standalone RAG retriever — its Answer Relevance of 0.118 makes it "
                "unsuitable for generation quality despite fast retrieval."
            ): (
                "Model selection: use BGE when answer quality is the priority and MiniLM for "
                "latency-sensitive deployments. Treat BM25 as a strong retrieval baseline rather than "
                "a standalone generator — its Answer Relevance of 0.125 remains far below the dense "
                "retrievers despite fast lookup."
            ),
        }

        for paragraph in root.findall(".//w:p", DOC_NS):
            text = paragraph_text(paragraph)
            if text in paragraph_replacements:
                set_paragraph_text(paragraph, paragraph_replacements[text])

        tables = root.findall(".//w:tbl", DOC_NS)

        # Table 1: corpus statistics.
        table1_rows = [
            ["Chunk Size", "Total Chunks", "Avg Tokens/Chunk", "Overlap"],
            ["256 tokens", "10,260", "~206", "64 tokens"],
            ["512 tokens", "5,106", "~379", "64 tokens"],
            ["1024 tokens", "2,332", "~794", "64 tokens"],
        ]
        for row, values in zip(tables[0].findall("./w:tr", DOC_NS), table1_rows):
            for cell, value in zip(row.findall("./w:tc", DOC_NS), values):
                set_cell_text(cell, value)

        # Table 3: retrieval results.
        retrieval_rows = read_csv_rows(Path("results/retrieval_metrics.csv"))
        table3 = tables[2]
        table3_rows = table3.findall("./w:tr", DOC_NS)
        header_values = ["Model", "Chunk", "MRR@5", "P@3", "P@5", "R@5", "R@10", "Latency"]
        for cell, value in zip(table3_rows[0].findall("./w:tc", DOC_NS), header_values):
            set_cell_text(cell, value)
        for row, csv_row in zip(table3_rows[1:], retrieval_rows):
            values = [
                csv_row["model_key"],
                csv_row["chunk_size"],
                fmt_metric(csv_row["MRR@5"]),
                fmt_metric(csv_row["Precision@3"]),
                fmt_metric(csv_row["Precision@5"]),
                fmt_metric(csv_row["Recall@5"]),
                fmt_metric(csv_row["Recall@10"]),
                fmt_latency_ms(csv_row["latency_ms"]),
            ]
            for cell, value in zip(row.findall("./w:tc", DOC_NS), values):
                set_cell_text(cell, value)

        # Table 4: generation results in report-friendly order.
        generation_rows = {row["model_key"]: row for row in read_csv_rows(Path("results/generation_metrics.csv"))}
        generation_order = ["BM25", "MiniLM", "MPNet", "BGE"]
        table4 = tables[3]
        table4_rows = table4.findall("./w:tr", DOC_NS)
        for row, model_key in zip(table4_rows[1:], generation_order):
            csv_row = generation_rows[model_key]
            values = [
                model_key,
                fmt_metric(csv_row["Answer Relevance"]),
                fmt_metric(csv_row["Faithfulness"]),
                fmt_metric(csv_row["Context Precision"]),
                f"{float(csv_row['latency_ms']):.1f}",
            ]
            for cell, value in zip(row.findall("./w:tc", DOC_NS), values):
                set_cell_text(cell, value)

        insert_report_screenshots(work_dir, root)
        tree.write(document_path, encoding="utf-8", xml_declaration=True)

        report_media_map = {
            "3eb63dbab853122d89eca0ab9bf2847f70b265f6.png": plots_dir / "MRR.png",
            "2a7d469c613f3d0c98b6da9e948fb7d2a2bb225a.png": plots_dir / "Precision_at_5.png",
            "73332b6718b84413b4cde31d6a19ad0036ddb808.png": plots_dir / "Recall_at_5.png",
            "6679a66aff209b22025c831ac006ed8e818a3d62.png": plots_dir / "heatmap_MRR.png",
            "611a587b2233404e2a5129355d7abf188e40c454.png": plots_dir / "bm25_vs_dense.png",
            "5109058096beeb6bfb2aa402b3de0e559b21d939.png": plots_dir / "generation_metrics.png",
            "118407f37beb31470159a1b71284240f1dbb0d0d.png": plots_dir / "latency.png",
            "eaa65b95539e07083645a3cd853940f0a5136e83.png": plots_dir / "latency_vs_mrr.png",
        }
        replace_media_files(work_dir / "word" / "media", report_media_map)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(work_dir.rglob("*")):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(work_dir).as_posix())


def update_presentation(pptx_path: Path, output_path: Path, plots_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        with zipfile.ZipFile(pptx_path) as archive:
            archive.extractall(work_dir)

        slide_replacements = {
            1: {
                "120": "150",
                "20": "100",
            },
            3: {
                "120 papers": "150 papers",
                "▸ rag/evaluation/metrics.py  —  Recall@K, Precision@K, MRR, AR, Faithfulness":
                    "▸ rag/evaluation/metrics.py  —  Recall@K, Precision@K, MRR@K, AR, Faithfulness",
            },
            4: {
                "120": "150",
                "8,131": "10,260",
                "3,915": "5,106",
                "20": "100",
                "Manual QA": "Benchmark QA",
            },
            5: {
                "MRR Performance Across All Configurations": "MRR@5 Performance Across All Configurations",
                "Perfect MRR": "Dense Retrieval Leads at 256 / 512",
                "BGE: 1.000 at chunk 256 and 512": "BGE: 0.990 at chunk 256 and 512",
                "BM25 Baseline": "BM25 Remains Competitive",
                "MRR range: 0.946 – 0.975": "MRR@5 range: 0.959 – 0.990",
                "Chunk 1024 Gap Narrows": "Chunk 1024 Favors BM25",
                "BM25 (0.975) surpasses MPNet (0.938) at chunk 1024 — sparse retrieval improves with larger chunks":
                    "BM25 (0.990) edges BGE (0.983) at chunk 1024 — sparse retrieval improves with larger chunks",
                "BGE row is uniformly darkest — highest MRR across every chunk size configuration":
                    "BGE is strongest at 256/512, while BM25 slightly leads at 1024",
            },
            6: {
                "All models — larger chunks introduce more off-topic content per unit. BGE maintains highest precision at every size.":
                    "Larger chunks introduce more mixed content per unit. BGE still maintains the highest Precision@5 at every size.",
                "Recall@5 doubles from chunk 256→512 across all models. More content per chunk = better paper coverage.":
                    "Recall@5 rises sharply with chunk size across all models. More content per chunk improves paper-level coverage.",
                "BGE Precision@5: 0.995 (256), 0.975 (512), 0.915 (1024). BM25 consistently 3–7% lower.":
                    "BGE Precision@5: 0.970 (256), 0.950 (512), 0.888 (1024). BM25 stays close but consistently lower.",
                "BM25 Recall@5 = 0.316 at chunk 1024, matching or exceeding dense models — sparse retrieval catches up at large chunk sizes.":
                    "BM25 Recall@5 = 0.375 at chunk 1024, showing sparse retrieval catches up strongly at large chunk sizes.",
            },
            7: {
                "Sparse Baseline vs Best Dense Model — MRR": "Sparse Baseline vs Best Dense Model — MRR@5",
                "BM25: 0.950   BGE: 1.000": "BM25: 0.959   BGE: 0.990",
                "Dense gain: +5%": "Dense gain: +3%",
                "BM25: 0.946   BGE: 1.000": "BM25: 0.978   BGE: 0.990",
                "Dense gain: +6%": "Dense gain: +1%",
                "BM25: 0.975   BGE: 0.983": "BM25: 0.990   BGE: 0.983",
                "Dense gain: +1%": "Dense gain: -1%",
                "Dense retrieval consistently outperforms BM25 on MRR.":
                    "Dense retrieval leads at 256/512, but BM25 slightly leads at 1024.",
                "The gap narrows at chunk 1024 as term overlap increases.":
                    "The gap narrows sharply at chunk 1024 as term overlap increases.",
            },
            8: {
                "7.7× Answer Relevance Gap": "7.3× Answer Relevance Gap",
                "BGE: 0.910  vs  BM25: 0.118": "BGE: 0.912  vs  BM25: 0.125",
                "All dense models: 0.974 – 0.992": "All models: 0.986 – 0.997",
                "Gemini reliably grounds answers in context": "Deterministic Gemini generation stays strongly grounded in context",
                "Generator performance is stable — embedding choice determines final answer quality":
                    "Retrieval quality still drives semantic answer quality most strongly",
            },
            9: {
                "MRR: 1.000  ·  Latency: 37ms": "MRR@5: 0.990  ·  Latency: 31ms",
                "AR: 0.910": "AR: 0.912",
                "MRR: 0.975  ·  Latency: 18ms": "MRR@5: 0.917  ·  Latency: 20ms",
                "AR: 0.724": "AR: 0.719",
                "Moderate gains at moderate latency cost": "Higher latency without a clear gain over MiniLM at chunk 512",
                "MRR: 0.975  ·  Latency: 8ms": "MRR@5: 0.975  ·  Latency: 7ms",
                "AR: 0.709": "AR: 0.728",
                "MRR: 0.946  ·  Latency: 9ms": "MRR@5: 0.978  ·  Latency: 12ms",
                "AR: 0.118": "AR: 0.125",
                "Fast but poor generation quality — avoid for RAG": "Fast and strong lexically, but still weak on semantic answer quality",
            },
            10: {
                "BM25 P@5 = 0.900 but Answer Relevance = 0.118. High retrieval precision does not guarantee semantic alignment with query intent.":
                    "BM25 P@5 = 0.862 but Answer Relevance = 0.125. High lexical precision does not guarantee semantic alignment with query intent.",
                "BM25 finds the paper, not the passage": "BM25 is a strong lexical baseline",
                "Dense retrieval finds the passage most aligned with the question. For RAG, that distinction determines answer quality.":
                    "It becomes strongest at chunk 1024 on MRR@5, but dense retrievers still surface passages that align better with answer intent.",
                "BGE is the best retriever": "BGE is the best overall dense retriever",
                "Perfect MRR (1.000) at chunks 256 and 512. Model scale (24 layers, 1024d) and task-specific fine-tuning drive the advantage.":
                    "It leads the dense models at chunks 256 and 512 (MRR@5 = 0.990). Model scale and task-specific tuning still pay off.",
                "8ms latency — matches BM25 speed. 97.5% of BGE accuracy. Best speed-accuracy trade-off for latency-sensitive applications.":
                    "6.8ms latency — about 98.5% of BGE's MRR@5 at chunk 512. Best speed-accuracy trade-off for latency-sensitive applications.",
                "Faithfulness 0.974–0.992 across all dense models. Retrieval quality — not generation — is the bottleneck.":
                    "Faithfulness stays very high (0.986–0.997) across all models. Retrieval quality — not generation grounding — remains the bottleneck.",
            },
            11: {
                "▸ Searchable paper browser — all 120 papers in sidebar":
                    "▸ Searchable paper browser — all 150 papers in sidebar",
                "Best accuracy (MRR 1.000). Justified for quality-critical applications.":
                    "Best overall dense retrieval quality (MRR@5 0.990 at chunk 512). Justified for quality-critical applications.",
                "8ms latency at 97.5% BGE accuracy — production default.":
                    "6.8ms latency at about 98.5% of BGE's MRR@5 — production default.",
                "0.118 Answer Relevance makes it unsuitable despite fast retrieval.":
                    "0.125 Answer Relevance keeps it unsuitable as a standalone RAG generator despite fast retrieval.",
            },
            14: {
                "1.000": "0.990",
                "BGE MRR": "BGE MRR@5",
                "7.7×": "7.3×",
                "8ms": "7ms",
            },
        }

        for slide_number, replacements in slide_replacements.items():
            slide_path = work_dir / "ppt" / "slides" / f"slide{slide_number}.xml"
            tree = ET.parse(slide_path)
            root = tree.getroot()
            replace_slide_text(root, replacements)
            tree.write(slide_path, encoding="utf-8", xml_declaration=True)

        presentation_media_map = {
            "image-5-1.png": plots_dir / "MRR.png",
            "image-5-2.png": plots_dir / "heatmap_MRR.png",
            "image-6-1.png": plots_dir / "Precision_at_5.png",
            "image-6-2.png": plots_dir / "Recall_at_5.png",
            "image-7-1.png": plots_dir / "bm25_vs_dense.png",
            "image-8-1.png": plots_dir / "generation_metrics.png",
            "image-9-1.png": plots_dir / "latency.png",
            "image-9-2.png": plots_dir / "latency_vs_mrr.png",
        }
        replace_media_files(work_dir / "ppt" / "media", presentation_media_map)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(work_dir.rglob("*")):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(work_dir).as_posix())


def main() -> None:
    register_namespaces()

    parser = argparse.ArgumentParser(description="Update external course deliverables from latest benchmark results.")
    parser.add_argument("--report-in", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--ppt-in", type=Path, required=True)
    parser.add_argument("--ppt-out", type=Path, required=True)
    parser.add_argument("--plots-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    update_report(args.report_in, args.report_out, args.plots_dir)
    update_presentation(args.ppt_in, args.ppt_out, args.plots_dir)


if __name__ == "__main__":
    main()

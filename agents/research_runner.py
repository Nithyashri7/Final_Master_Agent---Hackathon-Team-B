
"""
research_runner.py
Thin wrapper around research_agent.

Exposes a single function `run_research_pipeline(topic: str)` that:
- searches papers (arxiv/DOAJ/CORE/OpenAlex)
- downloads + summarizes them with Gemini
- compares models and returns the "best model" text + summaries
"""

from typing import Dict, Any, List, Tuple

from pathlib import Path
import concurrent.futures

from .research_agent import (
    create_data_directory,
    search_arxiv,
    search_doaj,
    search_core,
    search_openalex,
    process_paper,
    process_papers_with_gemini,
    compare_models_across_papers,
)


def run_research_pipeline(topic: str) -> Dict[str, Any]:
    data_dir = create_data_directory()
    pdf_dir = data_dir / "pdfs"
    pdf_dir.mkdir(exist_ok=True, parents=True)
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True, parents=True)

    arxiv_papers = search_arxiv(topic)
    doaj_papers = search_doaj(topic)
    core_papers = search_core(topic)
    openalex_papers = search_openalex(topic)

    all_papers = arxiv_papers + doaj_papers + core_papers + openalex_papers

    papers_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_paper = {
            executor.submit(process_paper, paper, pdf_dir): paper for paper in all_papers
        }
        for future in concurrent.futures.as_completed(future_to_paper):
            paper = future.result()
            if paper and paper.get("title"):
                papers_dict[paper["title"]] = paper

    summaries, relevant_papers = process_papers_with_gemini(papers_dict)
    best_model_report = compare_models_across_papers(summaries)

    return {
        "topic": topic,
        "papers_found": len(papers_dict),
        "relevant_papers": len(relevant_papers),
        "summaries": summaries,
        "best_model_report": best_model_report,
    }

"""
rag_evaluator.py — RAGAS-based evaluation harness for DocumentRAGBot.

Measures the pipeline with four standard RAG metrics:

  Reference-free (no ground truth needed — safe to run on ANY live query):
    - faithfulness       : is the answer actually supported by the retrieved
                            context, or is the LLM hallucinating / adding
                            facts not present in the sources?
    - answer_relevancy   : does the answer actually address the question asked
                            (penalizes vague, evasive, or off-topic answers)?

  Reference-based (needs a curated eval set with expected answers):
    - context_precision  : of the chunks retrieved, how many were actually
                            relevant? (retrieval precision)
    - context_recall     : of the information needed to answer correctly,
                            how much did retrieval actually surface? (retrieval
                            recall — did we miss something important?)

Why Groq instead of OpenAI as the judge: RAGAS defaults to OpenAI for both the
LLM judge and embeddings. This project already has a Groq key and a local
sentence-transformers model, so there's no reason to add a second paid
provider just to run evals.

IMPORTANT — dependency pinning:
RAGAS's latest release (0.4.x at time of writing) has a broken import chain
against current langchain-community (it imports a VertexAI integration class
that's been removed from langchain-community's newer releases). The versions
below are verified to import and run together cleanly:

    ragas==0.1.21
    langchain==0.2.16
    langchain-core==0.2.43
    langchain-community==0.2.16
    langchain-groq==0.1.9
    datasets==5.0.1

Add these EXACT pins to requirements.txt. Do not let pip resolve "latest" for
this stack — it will break.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAGEvaluator:
    """
    Wraps RAGAS with a Groq judge LLM + local embedding model. Bot-agnostic —
    takes plain (query, answer, contexts, ground_truth) rows, so it works with
    DocumentRAGBot.ask_for_eval() output or any other RAG pipeline's output.
    """

    def __init__(self,
                 groq_api_key: Optional[str] = None,
                 judge_model: str = "openai/gpt-oss-120b",
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        judge_model: should be a strong, instruction-following model — RAGAS
        metrics ask the judge to do multi-step reasoning (e.g. break an answer
        into claims and check each against context), so don't use your
        smallest/fastest Groq model here even if it's what you use for
        end-user answers.
        """
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY — required for the RAGAS judge LLM")

        judge_llm = ChatGroq(groq_api_key=api_key, model_name=judge_model, temperature=0)
        self.llm = LangchainLLMWrapper(judge_llm)

        judge_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.embeddings = LangchainEmbeddingsWrapper(judge_embeddings)

        # Bind the judge LLM/embeddings onto every metric object. RAGAS metrics
        # default to OpenAI internally unless you explicitly set these.
        for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
            metric.llm = self.llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = self.embeddings

    def _build_dataset(self, rows: List[Dict[str, Any]]) -> Dataset:
        """
        rows: [{ "query": str, "answer": str, "contexts": List[str],
                 "ground_truth": Optional[str] }, ...]
        """
        data = {
            "question": [r["query"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
        }
        has_ground_truth = all(r.get("ground_truth") for r in rows)
        if has_ground_truth:
            data["ground_truth"] = [r["ground_truth"] for r in rows]
        return Dataset.from_dict(data), has_ground_truth

    def evaluate_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Scores a batch of (query, answer, contexts[, ground_truth]) rows in
        ONE ragas.evaluate() call — much cheaper than scoring row-by-row,
        since RAGAS batches its judge-LLM calls internally.

        Automatically selects the metric set:
          - No ground_truth on any row  -> faithfulness + answer_relevancy only
          - Ground_truth present on all -> all four metrics
        """
        if not rows:
            return {"per_query": [], "aggregate": {}}

        dataset, has_ground_truth = self._build_dataset(rows)

        metrics = [faithfulness, answer_relevancy]
        if has_ground_truth:
            metrics += [context_precision, context_recall]
        else:
            print("⚠️ No ground_truth provided — skipping context_precision/"
                  "context_recall (they require a reference answer). Running "
                  "faithfulness + answer_relevancy only.")

        result = evaluate(dataset, metrics=metrics)
        result_df = result.to_pandas()

        per_query = []
        for i, row in result_df.iterrows():
            entry = {
                "query": row["question"],
                "answer": row["answer"],
                "faithfulness": round(float(row["faithfulness"]), 3) if "faithfulness" in row else None,
                "answer_relevancy": round(float(row["answer_relevancy"]), 3) if "answer_relevancy" in row else None,
            }
            if has_ground_truth:
                entry["context_precision"] = round(float(row["context_precision"]), 3)
                entry["context_recall"] = round(float(row["context_recall"]), 3)
            per_query.append(entry)

        aggregate = {}
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if m in result_df.columns:
                aggregate[m] = round(float(result_df[m].mean()), 3)

        return {
            "per_query": per_query,
            "aggregate": aggregate,
            "n_queries": len(rows),
            "had_ground_truth": has_ground_truth,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def evaluate_bot(self, bot, eval_items: List[Dict[str, str]],
                      k: int = 3, mode: str = "hybrid", rerank: bool = True) -> Dict[str, Any]:
        """
        Convenience wrapper: runs each eval item through bot.ask_for_eval(),
        then scores the batch. This is the main entry point for the eval
        dashboard — point it at DocumentRAGBot + a question set.

        eval_items: [{"query": str, "ground_truth": Optional[str]}, ...]
        """
        rows = []
        for item in eval_items:
            result = bot.ask_for_eval(item["query"], k=k, mode=mode, rerank=rerank)
            rows.append({
                "query": result["query"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": item.get("ground_truth"),
            })
        return self.evaluate_rows(rows)

    @staticmethod
    def save_report(report: Dict[str, Any], out_dir: str = "./eval_reports"):
        """
        Saves both a JSON (full detail, for the dashboard to consume) and a
        CSV (per-query scores, easy to eyeball or diff between pipeline
        versions in a spreadsheet) with a timestamped filename.
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        json_path = Path(out_dir) / f"eval_{stamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        csv_path = Path(out_dir) / f"eval_{stamp}.csv"
        if report["per_query"]:
            fieldnames = list(report["per_query"][0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(report["per_query"])

        print(f"📊 Saved eval report:\n   {json_path}\n   {csv_path}")
        return str(json_path), str(csv_path)

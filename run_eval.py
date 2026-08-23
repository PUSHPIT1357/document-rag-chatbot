"""
run_eval.py — bootstrap script for the DocumentRAGBot RAGAS evaluation pipeline.

Usage:
    python run_eval.py

Loads your document into DocumentRAGBot, runs a seed set of Q&A pairs through
it, scores the pipeline with RAGEvaluator, prints a summary, and saves a
timestamped JSON + CSV report to ./eval_reports/ — this is the data source
your eval dashboard should read from.

Expand EVAL_SET below with real questions pulled from your actual SEBI/NISM
document(s) as you go — the more coverage, the more trustworthy the aggregate
score is. 5-10 questions is a starting point, not a finish line.
"""

from pathlib import Path
from main import DocumentRAGBot
from rag_evaluator import RAGEvaluator

# ============================================
# SEED EVAL SET
# ============================================
# ground_truth is optional per-item: leave it out (or None) if you don't have
# a confident reference answer yet — those rows still get faithfulness +
# answer_relevancy, just not context_precision/context_recall.
EVAL_SET = [
    {
        "query": "What is the difference between provisions and reserves?",
        "ground_truth": (
            "Provisions are charged on profits and must be created regardless "
            "of profit level to meet a known liability, often as a legal "
            "requirement; recorded on the debit side of the Profit & Loss "
            "Account. Reserves are an appropriation of profits, created only "
            "when adequate profits exist, to strengthen the business's liquid "
            "resources as a matter of prudence, not law; recorded on the "
            "debit side of the Profit & Loss Appropriation Account and always "
            "shown on the liabilities side of the balance sheet. Reserves can "
            "be invested outside the business as a Reserve Fund; provisions "
            "generally are not."
        ),
    },
    {
        "query": "What are non-current assets?",
        "ground_truth": (
            "Non-current assets are resources owned by a company that are not "
            "expected to be converted into cash or realized within the next "
            "12 months. They are shown on the balance sheet under headings "
            "like investment, property, plant, equipment, and intangible "
            "assets, and include both tangible items (land, buildings, "
            "machinery, vehicles, computer equipment) and intangible items."
        ),
    },
    {
        "query": "What are the components of a balance sheet?",
        "ground_truth": (
            "The components of a balance sheet are assets, liabilities, and "
            "equity. Assets are resources owned by the company legally and "
            "economically, and are split into current and non-current assets."
        ),
    },
    # Add more real questions from your SEBI/NISM document(s) here.
]


def main():
    doc_path = input("Path to the document to load for this eval run: ").strip()
    if not doc_path or not Path(doc_path).exists():
        print(f"❌ File not found: {doc_path}")
        return

    print("🤖 Initializing DocumentRAGBot...")
    bot = DocumentRAGBot()

    print(f"📄 Building pipeline for: {doc_path}")
    if not bot.build_pipeline(doc_path):
        print("❌ Failed to build pipeline — aborting eval run.")
        return

    print("🧪 Initializing RAGAS evaluator (Groq judge + local embeddings)...")
    evaluator = RAGEvaluator()

    print(f"📊 Running {len(EVAL_SET)} eval queries through the pipeline "
          f"(mode=hybrid, rerank=True)...")
    report = evaluator.evaluate_bot(bot, EVAL_SET, k=3, mode="hybrid", rerank=True)

    print("\n" + "=" * 60)
    print("AGGREGATE SCORES")
    print("=" * 60)
    for metric, score in report["aggregate"].items():
        print(f"  {metric:20s}: {score}")

    print("\n" + "=" * 60)
    print("PER-QUERY BREAKDOWN")
    print("=" * 60)
    for row in report["per_query"]:
        print(f"\nQ: {row['query']}")
        for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            if k in row and row[k] is not None:
                print(f"   {k}: {row[k]}")

    evaluator.save_report(report)


if __name__ == "__main__":
    main()

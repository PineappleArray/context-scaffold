"""
eval/run_locomo.py
Run LoCoMo benchmark against your context-scaffold pipeline.

Usage:
    python -m eval.run_locomo --mode scaffold
    python -m eval.run_locomo --mode raw
    python -m eval.run_locomo --compare results/scaffold.json results/raw.json
"""

import json
import time
import argparse
import asyncio
from pathlib import Path
from eval.metrics import compute_f1, compute_bert_score, compute_rouge, aggregate_results
from app import main
from app.inference.model import Model


# ─── Category Mapping ─────────────────────────────────────
# NOTE: The paper lists (1) single-hop (2) multi-hop (3) temporal ...
# but the actual dataset IDs are swapped for 2 and 3.
# Source: LoCoMo source code + memobase evaluation confirms:
#   1=single_hop, 2=temporal, 3=multi_hop, 4=open_domain, 5=adversarial

CATEGORY_MAP = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial"
}


# ─── Load LoCoMo Dataset ─────────────────────────────────

def load_locomo(path="data/locomo10.json"):
    """
    Load LoCoMo dataset from snap-research/locomo repo.
    Clone it: git clone https://github.com/snap-research/locomo
    Copy: cp locomo/data/locomo10.json data/locomo10.json
    """
    with open(path) as f:
        return json.load(f)


def extract_qa_pairs(sample):
    qa_pairs = []
    for qa in sample.get("qa", []):
        answer = qa.get("answer", qa.get("adversarial_answer"))
        if "question" not in qa or answer is None:
            continue
        qa_pairs.append({
            "question": qa["question"],
            "answer": answer,
            "category": CATEGORY_MAP.get(qa.get("category"), "unknown")
        })
    return qa_pairs


def extract_history(sample):
    """
    Build flat conversation history from a LoCoMo sample's
    nested session structure.
    """
    conv_data = sample["conversation"]
    history = []

    session_keys = sorted(
        k for k in conv_data
        if k.startswith("session") and not k.endswith("date_time")
    )

    for key in session_keys:
        for turn in conv_data[key]:
            history.append({
                "speaker": turn["speaker"],
                "content": turn["text"]
            })

    return history


# ─── Run Single QA ────────────────────────────────────────

async def run_qa_with_scaffold(question, conversation_history, pipeline):
    """
    Run a question through your context-scaffold pipeline.
    Pipeline feeds conversation history through topic extraction,
    ACT-R scoring, context building, then generates answer.
    """
    # Feed conversation history into the pipeline
    for msg in conversation_history:
        await pipeline.process_input(
            message=msg["content"],
            user_id=msg["speaker"],
            session_id="locomo_eval"
        )

    # Ask the question
    response = await pipeline.process_input(
        message=question,
        user_id="evaluator",
        session_id="locomo_eval"
    )
    return response.response


async def run_qa_raw(question, conversation_history, llm_client):
    """
    Run a question with raw conversation history (no scaffold).
    Just stuff the whole history into the context window.
    """
    messages = [{"role": "system", "content": "Answer the question based on the conversation history."}]

    # Dump raw history
    for msg in conversation_history:
        messages.append({
            "role": "user",
            "content": f"[{msg['speaker']}]: {msg['content']}"
        })

    # Ask question
    messages.append({"role": "user", "content": question})

    response = await llm_client.generate_raw(messages)
    return response


# ─── Run Full Evaluation ──────────────────────────────────

async def run_eval(mode, pipeline=None, llm_client=None):
    """
    Run LoCoMo evaluation.
    mode: 'scaffold' uses your ACT-R pipeline
          'raw' uses raw context stuffing
    """
    dataset = load_locomo()
    results = {
        "mode": mode,
        "timestamp": time.time(),
        "conversations": [],
        "by_category": {
            "single_hop": [],
            "multi_hop": [],
            "temporal": [],
            "open_domain": [],
            "adversarial": []
        }
    }

    for conv_idx, sample in enumerate(dataset):
        print(f"\n--- Conversation {conv_idx + 1}/{len(dataset)} ---")

        history = extract_history(sample)
        qa_pairs = extract_qa_pairs(sample)
        conv_results = []
        
        for qa_idx, qa in enumerate(qa_pairs):
            print(f"  Q{qa_idx + 1}/{len(qa_pairs)} [{qa['category']}]: {qa['question'][:60]}...")

            start = time.time()
            if mode == "scaffold":
                predicted = await run_qa_with_scaffold(
                    qa["question"], history, pipeline
                )
            else:
                predicted = await run_qa_raw(
                    qa["question"], history, llm_client
                )
            latency = time.time() - start

            # Score
            gold = qa["answer"]
            f1 = compute_f1(predicted, gold)
            bert = compute_bert_score(predicted, gold)
            rouge = compute_rouge(predicted, gold)

            result = {
                "question": qa["question"],
                "gold": gold,
                "predicted": predicted,
                "category": qa["category"],
                "f1": f1,
                "bert_score": bert,
                "rouge_l": rouge,
                "latency": latency
            }

            conv_results.append(result)
            results["by_category"][qa["category"]].append(result)

            print(f"    F1: {f1:.3f} | BERT: {bert:.3f} | ROUGE-L: {rouge:.3f} | {latency:.2f}s")

        results["conversations"].append({
            "conversation_id": conv_idx,
            "results": conv_results
        })

    # Aggregate
    results["summary"] = aggregate_results(results)
    return results


# ─── Compare Two Runs ─────────────────────────────────────

def compare_results(path_a, path_b):
    """
    Compare two evaluation runs side by side.
    Prints per-category and overall comparison.
    """
    with open(path_a) as f:
        results_a = json.load(f)
    with open(path_b) as f:
        results_b = json.load(f)

    print(f"\n{'='*70}")
    print(f"COMPARISON: {results_a['mode']} vs {results_b['mode']}")
    print(f"{'='*70}")

    print(f"\n{'Category':<20} {'Metric':<12} {results_a['mode']:<15} {results_b['mode']:<15} {'Delta':<10}")
    print("-" * 70)

    for category in ["single_hop", "multi_hop", "temporal", "open_domain", "overall"]:
        if category == "overall":
            a_scores = results_a["summary"]
            b_scores = results_b["summary"]
        else:
            a_items = results_a["by_category"].get(category, [])
            b_items = results_b["by_category"].get(category, [])
            a_scores = {
                "avg_f1": sum(i["f1"] for i in a_items) / max(len(a_items), 1),
                "avg_bert": sum(i["bert_score"] for i in a_items) / max(len(a_items), 1),
                "avg_rouge": sum(i["rouge_l"] for i in a_items) / max(len(a_items), 1)
            }
            b_scores = {
                "avg_f1": sum(i["f1"] for i in b_items) / max(len(b_items), 1),
                "avg_bert": sum(i["bert_score"] for i in b_items) / max(len(b_items), 1),
                "avg_rouge": sum(i["rouge_l"] for i in b_items) / max(len(b_items), 1)
            }

        for metric in ["avg_f1", "avg_bert", "avg_rouge"]:
            a_val = a_scores.get(metric, 0)
            b_val = b_scores.get(metric, 0)
            delta = b_val - a_val
            sign = "+" if delta > 0 else ""
            print(f"{category:<20} {metric:<12} {a_val:<15.3f} {b_val:<15.3f} {sign}{delta:<10.3f}")
        print()


# ─── CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scaffold", "raw"], help="Evaluation mode")
    parser.add_argument("--compare", nargs=2, help="Compare two result files")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.mode:
        Path(args.output).mkdir(exist_ok=True)

        if args.mode == "scaffold":
            async def run_scaffold():
                await main.startup()
                results = await run_eval(
                    "scaffold",
                    pipeline=main.chat.production_rules,
                    llm_client=Model()
                )
                return results

            results = asyncio.run(run_scaffold())
            print("Scaffold evaluation complete.")
        else:
            print(args.mode)
            print("Initialize your LLM client in the TODO section above")
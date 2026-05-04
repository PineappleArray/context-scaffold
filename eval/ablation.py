"""
eval/ablation.py
Run the full ablation study.

Compares:
1. Gemma 4B raw (no scaffolding) — baseline
2. Gemma 4B + topic store (no decay) — partial scaffold
3. Gemma 4B + topic store + ACT-R decay — full scaffold
4. Gemma 12B raw — larger model baseline

Usage:
    python -m eval.ablation
"""

import json
import time
import asyncio
from pathlib import Path
from eval.run_locomo import run_eval
from eval.metrics import aggregate_results


async def run_ablation():
    Path("results").mkdir(exist_ok=True)

    configs = [
        {
            "name": "gemma4b_raw",
            "description": "Gemma 4B, no scaffolding, raw context",
            "mode": "raw",
            "model": "gemma3:4b"  # ollama model name
        },
        {
            "name": "gemma4b_topics_only",
            "description": "Gemma 4B + topic store, no ACT-R decay",
            "mode": "scaffold",
            "use_decay": False
        },
        {
            "name": "gemma4b_full_scaffold",
            "description": "Gemma 4B + topic store + ACT-R decay + noise",
            "mode": "scaffold",
            "use_decay": True
        },
        # Run this one on a cloud GPU
        # {
        #     "name": "gemma12b_raw",
        #     "description": "Gemma 12B, no scaffolding — target to beat",
        #     "mode": "raw",
        #     "model": "gemma3:12b"
        # }
    ]

    all_results = {}

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Running: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*70}")

        start = time.time()

        # TODO: Initialize pipeline based on config
        # if config["mode"] == "scaffold":
        #     pipeline = create_pipeline(use_decay=config.get("use_decay", True))
        #     results = await run_eval("scaffold", pipeline=pipeline)
        # else:
        #     llm = LLMClient(model=config.get("model", "gemma3:4b"))
        #     results = await run_eval("raw", llm_client=llm)

        # Placeholder
        results = {"mode": config["name"], "summary": {}}

        elapsed = time.time() - start
        results["wall_time"] = elapsed
        all_results[config["name"]] = results

        # Save individual results
        output_path = f"results/{config['name']}_{int(time.time())}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {output_path} ({elapsed:.1f}s)")

    # Print comparison table
    print(f"\n{'='*70}")
    print("ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Config':<30} {'F1':<10} {'BERT':<10} {'ROUGE-L':<10} {'Latency':<10}")
    print("-" * 70)

    for name, results in all_results.items():
        summary = results.get("summary", {})
        print(f"{name:<30} "
              f"{summary.get('avg_f1', 0):<10.3f} "
              f"{summary.get('avg_bert', 0):<10.3f} "
              f"{summary.get('avg_rouge', 0):<10.3f} "
              f"{summary.get('avg_latency', 0):<10.2f}s")

    # Per-category breakdown
    print(f"\n{'='*70}")
    print("PER-CATEGORY BREAKDOWN (F1)")
    print(f"{'='*70}")
    categories = ["single_hop", "multi_hop", "temporal", "open_domain"]
    print(f"\n{'Config':<30} ", end="")
    for cat in categories:
        print(f"{cat:<15}", end="")
    print()
    print("-" * 90)

    for name, results in all_results.items():
        per_cat = results.get("summary", {}).get("per_category", {})
        print(f"{name:<30} ", end="")
        for cat in categories:
            f1 = per_cat.get(cat, {}).get("avg_f1", 0)
            print(f"{f1:<15.3f}", end="")
        print()


if __name__ == "__main__":
    asyncio.run(run_ablation())
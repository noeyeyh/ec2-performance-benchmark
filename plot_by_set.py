
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV = "results.csv"
OUTDIR = Path("figs_by_set")
OUTDIR.mkdir(exist_ok=True)

metrics = {
    "total_s": ("Total Time (s)", True),
    "llm_s": ("LLM Time (s)", True),
    "cpu_total_s": ("CPU Total (s)", True),
    "rss_mb": ("RSS Memory (MB)", True),
    "prompt_cps": ("Prompt Throughput (chars/s)", False),
    "output_cps": ("Output Throughput (chars/s)", False),
}

df = pd.read_csv(CSV)

for set_name, sdf in df.groupby("set"):
    for col, (ylabel, lower_better) in metrics.items():
        d = sdf.sort_values(col, ascending=lower_better)

        plt.figure()
        plt.bar(d["instance"], d[col])
        plt.title(f"{set_name} - {col}")
        plt.ylabel(ylabel)
        plt.xticks(rotation=20)

        for i, v in enumerate(d[col]):
            plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        out = OUTDIR / f"{set_name}_{col}.png"
        plt.savefig(out, dpi=200)
        plt.close()

print("✅ graphs saved to figs_by_set/")
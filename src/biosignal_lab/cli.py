"""Reproduce the demo as a JSON report and confusion-matrix figure."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from .signals import make_synthetic_dataset
from .train import train_evaluate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run reproducible synthetic waveform classification.")
    parser.add_argument("--subjects", type=int, default=30)
    parser.add_argument("--segments", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    data = make_synthetic_dataset(n_subjects=args.subjects,
                                  samples_per_subject=args.segments, seed=args.seed)
    trained = train_evaluate(data, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    report = args.out / "demo_report.json"
    report.write_text(json.dumps(trained.report, indent=2) + "\n", encoding="utf-8")
    disp = ConfusionMatrixDisplay.from_predictions(
        trained.actual, trained.predicted, labels=[0, 1], display_labels=["regular", "irregular"],
        cmap="Blues", colorbar=False,
    )
    disp.ax_.set_title("Synthetic waveforms: subject-held-out test")
    disp.figure_.tight_layout()
    disp.figure_.savefig(args.out / "confusion_matrix.png", dpi=150)
    plt.close(disp.figure_)
    print(json.dumps(trained.report, indent=2))
    print(f"Wrote {report} and {args.out / 'confusion_matrix.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

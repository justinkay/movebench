from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from movebench import paths
from visualizations.plot_energy_scores_by_distance_group import (
    NORMALIZE_BY,
    OPTIONAL_PLOT_METHODS,
    PLOT_METHODS,
    SPLIT_ORDER,
    TIMESCALE_ORDER,
    _method_label,
    load_plot_results,
)

STUDY_CITATIONS_PATH = Path(__file__).with_name("study_citations.csv")
OUTPUT_PATH = (
    paths.repo_root()
    / "results"
    / "bench"
    / "report"
    / "final_two_row_energy_scores_table.tex"
)
METHOD_ORDER = [NORMALIZE_BY, *PLOT_METHODS, *OPTIONAL_PLOT_METHODS]


def _format_energy_score(value: float) -> str:
    if pd.isna(value):
        return "--"
    if abs(value) >= 100:
        text = f"{value:.0f}"
    elif abs(value) >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def build_table_lines() -> list[str]:
    roster = pd.read_csv(STUDY_CITATIONS_PATH, keep_default_na=False)
    results = load_plot_results()
    scores = results[results["n_steps_scored"] > 0].pivot(
        index=["study_id", "method"],
        columns=["split_key", "timescale"],
        values="energy_score",
    )
    scores = scores.reindex(
        index=pd.MultiIndex.from_product([roster["study_id"], METHOD_ORDER]),
        columns=pd.MultiIndex.from_product([SPLIT_ORDER, TIMESCALE_ORDER]),
    )
    lines = r"""% Energy score appendix table.
% Required packages in preamble:
% \usepackage{array}
% \usepackage{booktabs}
% \usepackage{longtable}
% \usepackage{multirow}
% \usepackage{ragged2e}
% \usepackage{natbib}
\begingroup
\footnotesize
\setlength{\tabcolsep}{1.5pt}
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\renewcommand{\arraystretch}{1.05}
\newcolumntype{L}[1]{>{\RaggedRight\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\Centering\arraybackslash}p{#1}}
\begin{longtable}{@{}L{0.16\textwidth}L{0.10\textwidth}*{8}{C{0.078\textwidth}}@{}}
\caption{Raw energy scores for each study and method. Lower is better. Dashes indicate unavailable results. MoveFormer denotes the shared model.}\label{tab:final_two_row_energy_scores}\\
\toprule
\multirow{2}{*}{Study} & \multirow{2}{*}{Method} & \multicolumn{4}{c}{ES, Temporal split $\downarrow$} & \multicolumn{4}{c}{ES, Individual split $\downarrow$} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
& & 1h & 12h & 1d & 7d & 1h & 12h & 1d & 7d \\
\midrule
\endfirsthead
\caption[]{Energy score results (continued).}\\
\toprule
\multirow{2}{*}{Study} & \multirow{2}{*}{Method} & \multicolumn{4}{c}{ES, Temporal split $\downarrow$} & \multicolumn{4}{c}{ES, Individual split $\downarrow$} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
& & 1h & 12h & 1d & 7d & 1h & 12h & 1d & 7d \\
\midrule
\endhead
\midrule
\multicolumn{10}{r}{\emph{Continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot""".splitlines()
    for study_idx, study in enumerate(roster.itertuples(index=False)):
        if study_idx:
            lines.append(r"\midrule")
        for method_idx, method in enumerate(METHOD_ORDER):
            study_label = (
                rf"\multirow{{{len(METHOD_ORDER)}}}{{=}}{{{study.citation}}}"
                if method_idx == 0
                else ""
            )
            method_label = (
                "don't move" if method == NORMALIZE_BY else _method_label(method).strip()
            )
            values = " & ".join(
                _format_energy_score(value)
                for value in scores.loc[(study.study_id, method)]
            )
            lines.append(f"{study_label} & {method_label} & {values} " + r"\\")
    return [*lines, r"\end{longtable}", r"\endgroup", ""]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the energy-score LaTeX table from current plot results."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    lines = build_table_lines()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

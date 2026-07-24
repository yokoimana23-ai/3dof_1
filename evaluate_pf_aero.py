"""Evaluate standalone flat-plate fin (PF) aerodynamic characteristics.

This script reads ``pf_aero_rev.csv`` using the source CFD sign convention as-is.
No sign conversion is applied here; future dynamics-facing conversion belongs in
``aero_pf.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = ["Mach", "Alpha", "Delta", "CD", "CL", "CM"]
COEFFS = ["CD", "CL", "CM"]
OUTDIR = Path("pf_aero_evaluation")

LABELS = {
    "Alpha": "Alpha [deg] (positive: nose-down angle of attack)",
    "Delta": "Delta [deg] (positive: fin deflection nose-up)",
    "Alpha_fin": "Alpha_fin = Alpha - Delta [deg]",
    "CD": "CD (drag coefficient)",
    "CL": "CL (positive: downward aerodynamic force)",
    "CM": "CM (positive: nose-down moment)",
}


def fmt_num(x: float) -> str:
    if pd.isna(x):
        return "nan"
    return f"{x:g}"


def safe_name(x: float) -> str:
    return fmt_num(x).replace("-", "m").replace(".", "p")


def ensure_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def grid_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_columns(df)
    nan_counts = df[REQUIRED_COLUMNS].isna().sum()
    duplicate_mask = df.duplicated(subset=["Mach", "Alpha", "Delta"], keep=False)
    machs = np.sort(df["Mach"].unique())
    alphas = np.sort(df["Alpha"].unique())
    deltas = np.sort(df["Delta"].unique())
    full = pd.MultiIndex.from_product([machs, alphas, deltas], names=["Mach", "Alpha", "Delta"])
    present = pd.MultiIndex.from_frame(df[["Mach", "Alpha", "Delta"]].drop_duplicates())
    missing = full.difference(present).to_frame(index=False)
    diagnostics = {
        "rows": len(df),
        "nan_counts": nan_counts,
        "duplicate_rows": df.loc[duplicate_mask, REQUIRED_COLUMNS].sort_values(["Mach", "Alpha", "Delta"]),
        "machs": machs,
        "alphas": alphas,
        "deltas": deltas,
        "missing_grid_points": missing,
    }
    return missing, diagnostics


def save_basic_plots(df: pd.DataFrame) -> None:
    for delta, dfd in df.groupby("Delta"):
        for coeff in COEFFS:
            fig, ax = plt.subplots(figsize=(7, 5))
            for mach, g in dfd.groupby("Mach"):
                g = g.sort_values("Alpha")
                ax.plot(g["Alpha"], g[coeff], marker="o", label=f"Mach {fmt_num(mach)}")
            ax.set_xlabel(LABELS["Alpha"])
            ax.set_ylabel(LABELS[coeff])
            ax.set_title(f"{coeff} vs Alpha, Delta={fmt_num(delta)} deg ({LABELS['Delta']})")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUTDIR / f"basic_{coeff}_vs_alpha_delta_{safe_name(delta)}.png", dpi=160)
            plt.close(fig)


def save_alpha_fin_plots(df: pd.DataFrame) -> None:
    for mach, dfm in df.groupby("Mach"):
        for coeff in COEFFS:
            fig, ax = plt.subplots(figsize=(7, 5))
            for delta, g in dfm.groupby("Delta"):
                g = g.sort_values("Alpha_fin")
                ax.plot(g["Alpha_fin"], g[coeff], marker="o", label=f"Delta {fmt_num(delta)} deg")
            ax.set_xlabel(LABELS["Alpha_fin"])
            ax.set_ylabel(LABELS[coeff])
            ax.set_title(f"{coeff} vs Alpha_fin, Mach={fmt_num(mach)}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUTDIR / f"alpha_fin_{coeff}_mach_{safe_name(mach)}.png", dpi=160)
            plt.close(fig)


def delta_effects(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = ["Mach", "Alpha"]
    tables = {delta: g.set_index(idx)[COEFFS] for delta, g in df.groupby("Delta")}
    diff_rows = []
    even_rows = []
    deriv_rows = []
    if 0 in tables:
        for delta in (8, -8):
            if delta in tables:
                common = tables[0].index.intersection(tables[delta].index)
                diff = tables[delta].loc[common] - tables[0].loc[common]
                for (mach, alpha), row in diff.iterrows():
                    diff_rows.append({"Mach": mach, "Alpha": alpha, "Delta": delta, **{f"d{c}": row[c] for c in COEFFS}})
    if 8 in tables and -8 in tables:
        common = tables[8].index.intersection(tables[-8].index)
        deriv = (tables[8].loc[common] - tables[-8].loc[common]) / 16.0
        for (mach, alpha), row in deriv.iterrows():
            deriv_rows.append({"Mach": mach, "Alpha": alpha, **{f"{c}_delta_per_deg": row[c] for c in COEFFS}})
        if 0 in tables:
            common3 = common.intersection(tables[0].index)
            even = (tables[8].loc[common3] + tables[-8].loc[common3]) / 2.0 - tables[0].loc[common3]
            for (mach, alpha), row in even.iterrows():
                even_rows.append({"Mach": mach, "Alpha": alpha, **{f"{c}_even": row[c] for c in COEFFS}})
    diff_cols = ["Mach", "Alpha", "Delta", "dCD", "dCL", "dCM"]
    deriv_cols = ["Mach", "Alpha", "CD_delta_per_deg", "CL_delta_per_deg", "CM_delta_per_deg"]
    even_cols = ["Mach", "Alpha", "CD_even", "CL_even", "CM_even"]
    return (
        pd.DataFrame(diff_rows, columns=diff_cols),
        pd.DataFrame(deriv_rows, columns=deriv_cols),
        pd.DataFrame(even_rows, columns=even_cols),
    )


def plot_metric_by_alpha(data: pd.DataFrame, ycols: list[str], stem: str, title_prefix: str, delta_col: bool = False) -> None:
    if data.empty:
        return
    for y in ycols:
        fig, ax = plt.subplots(figsize=(7, 5))
        if delta_col:
            for (mach, delta), g in data.groupby(["Mach", "Delta"]):
                g = g.sort_values("Alpha")
                ax.plot(g["Alpha"], g[y], marker="o", label=f"Mach {fmt_num(mach)}, Delta {fmt_num(delta)}")
        else:
            for mach, g in data.groupby("Mach"):
                g = g.sort_values("Alpha")
                ax.plot(g["Alpha"], g[y], marker="o", label=f"Mach {fmt_num(mach)}")
        ax.set_xlabel(LABELS["Alpha"])
        ax.set_ylabel(y)
        ax.set_title(f"{title_prefix}: {y}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTDIR / f"{stem}_{y}.png", dpi=160)
        plt.close(fig)


def trim_static_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mach, delta), g in df.groupby(["Mach", "Delta"]):
        g = g.sort_values("Alpha").drop_duplicates("Alpha")
        a = g["Alpha"].to_numpy(float)
        cm = g["CM"].to_numpy(float)
        for i in range(len(g) - 1):
            cm0, cm1 = cm[i], cm[i + 1]
            if cm0 == 0:
                trim = a[i]
                slope = (cm1 - cm0) / (a[i + 1] - a[i])
            elif cm0 * cm1 <= 0 and cm1 != cm0:
                trim = a[i] - cm0 * (a[i + 1] - a[i]) / (cm1 - cm0)
                slope = (cm1 - cm0) / (a[i + 1] - a[i])
            else:
                continue
            rows.append({"Mach": mach, "Delta": delta, "trim_alpha_deg": trim, "dCM_dAlpha_per_deg": slope, "static_stable": bool(slope < 0)})
    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame, diagnostics: dict[str, object], diff: pd.DataFrame, deriv: pd.DataFrame, even: pd.DataFrame, trim: pd.DataFrame) -> None:
    lines = ["# PF単体 空力特性評価", "", "## データ範囲と健全性", ""]
    lines.append(f"- 行数: {diagnostics['rows']}")
    lines.append(f"- Mach: {', '.join(fmt_num(x) for x in diagnostics['machs'])}")
    lines.append(f"- Alpha [deg]（正: nose-down）: {', '.join(fmt_num(x) for x in diagnostics['alphas'])}")
    lines.append(f"- Delta [deg]（正: nose-upフィン偏向）: {', '.join(fmt_num(x) for x in diagnostics['deltas'])}")
    nan = diagnostics['nan_counts']
    lines.append(f"- NaN数: " + ", ".join(f"{k}={int(v)}" for k, v in nan.items()))
    lines.append(f"- 重複格子行数: {len(diagnostics['duplicate_rows'])}")
    lines.append(f"- 欠けた格子点数（全Mach×全Alpha×全Deltaの直積基準）: {len(diagnostics['missing_grid_points'])}")
    lines += ["", "## 舵効きと正負舵角の差", ""]
    if diff.empty:
        lines.append("- Delta=0°を基準にしたDelta=+8°/-8°差分を評価できる共通格子点はありません。")
    else:
        for delta in sorted(diff["Delta"].unique()):
            subset = diff[diff["Delta"] == delta]
            lines.append(f"- Delta={fmt_num(delta)}° と Delta=0° の差分 ΔC を {len(subset)} 点で評価しました。")
    if deriv.empty:
        lines.append("- Delta=±8°の共通格子点がなく、中央差分の舵効きは評価できません。")
    else:
        lines.append(f"- 中央差分 C_delta=[C(+8)-C(-8)]/16° を {len(deriv)} 点で評価しました。")
        lines.append(f"- CL_delta範囲: {deriv['CL_delta_per_deg'].min():.6g} ～ {deriv['CL_delta_per_deg'].max():.6g} /deg")
        lines.append(f"- CM_delta範囲: {deriv['CM_delta_per_deg'].min():.6g} ～ {deriv['CM_delta_per_deg'].max():.6g} /deg")
    if not even.empty:
        lines.append(f"- 偶成分 C_even=[C(+8)+C(-8)]/2-C(0) を {len(even)} 点で評価しました。これは非対称性や迎角・舵角干渉の指標であり、単純に誤差とは断定しません。")
    lines += ["", "## Alpha_finによる整理", ""]
    lines.append("- Alpha_fin=Alpha-Delta を補助指標として追加し、CD/CL/CMをAlpha_finに対して描画しました。")
    lines.append("- Deltaごとの曲線が一致しない場合、胴体寄与・干渉・非線形性により、ロケット全体の係数は単純なフィン局所迎角モデルだけでは表せません。")
    lines += ["", "## トリム・静安定性", ""]
    if trim.empty:
        lines.append("- データ範囲内でCM=0の符号交差が見つからず、外挿なしではトリム迎角を評価できません。")
    else:
        stable_count = int(trim['static_stable'].sum())
        lines.append(f"- データ範囲内の線形補間で {len(trim)} 個のCM=0トリムを検出しました。")
        lines.append(f"- CSV規約では dCM/dAlpha < 0 を静安定とし、静安定判定は {stable_count}/{len(trim)} 件です。")
    lines += ["", "## データ上の制約", "", "- 評価はCSVの符号規約をそのまま用い、符号変換は行っていません。", "- 格子が疎な場合、差分・トリム勾配は局所線形近似に依存します。GF比較、軌道計算、制御器実装は含みません。"]
    (OUTDIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    df = pd.read_csv("pf_aero_rev.csv")
    ensure_columns(df)
    df = df[REQUIRED_COLUMNS].copy()
    df["Alpha_fin"] = df["Alpha"] - df["Delta"]
    missing, diagnostics = grid_diagnostics(df)
    missing.to_csv(OUTDIR / "missing_grid_points.csv", index=False)
    diagnostics["duplicate_rows"].to_csv(OUTDIR / "duplicate_rows.csv", index=False)
    df.to_csv(OUTDIR / "pf_aero_with_alpha_fin.csv", index=False)
    save_basic_plots(df)
    save_alpha_fin_plots(df)
    diff, deriv, even = delta_effects(df)
    diff.to_csv(OUTDIR / "delta_effects_vs_delta0.csv", index=False)
    deriv.to_csv(OUTDIR / "control_effectiveness_central_difference.csv", index=False)
    even.to_csv(OUTDIR / "even_components.csv", index=False)
    plot_metric_by_alpha(diff, ["dCD", "dCL", "dCM"], "delta_effect", "Delta effect relative to Delta=0", True)
    plot_metric_by_alpha(deriv, ["CL_delta_per_deg", "CM_delta_per_deg"], "control_effectiveness", "Central difference control effectiveness")
    plot_metric_by_alpha(even, ["CD_even", "CL_even", "CM_even"], "even_component", "Even component")
    trim = trim_static_stability(df)
    trim.to_csv(OUTDIR / "trim_static_stability.csv", index=False)
    write_summary(df, diagnostics, diff, deriv, even, trim)
    print(f"Wrote evaluation outputs to {OUTDIR}")


if __name__ == "__main__":
    main()

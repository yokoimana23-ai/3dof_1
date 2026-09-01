"""Evaluate standalone grid fin (GF) aerodynamic characteristics.

This script reads ``gf_aero_rev.csv`` using the source CFD sign convention as-is.
No sign conversion is applied here; any future dynamics-facing adapter is
responsible for convention conversion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = ["Mach", "Alpha", "Delta", "CD", "CL", "CM"]
COEFFS = ["CD", "CL", "CM"]
OUTDIR = Path("gf_aero_evaluation")

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
    """Return changes from zero, interval slopes, and 4-to-8 deg nonlinearity.

    GF data has only positive deflections, so these are secant/interval slopes,
    not a central difference.  The nonlinearity is the departure at 8 degrees
    from a linear extrapolation of the 0-to-4-degree response.
    """
    idx = ["Mach", "Alpha"]
    tables = {delta: g.set_index(idx)[COEFFS] for delta, g in df.groupby("Delta")}
    diff_rows = []
    nonlinear_rows = []
    slope_rows = []
    if 0 in tables:
        for delta in (4, 8):
            if delta in tables:
                common = tables[0].index.intersection(tables[delta].index)
                diff = tables[delta].loc[common] - tables[0].loc[common]
                for (mach, alpha), row in diff.iterrows():
                    diff_rows.append({"Mach": mach, "Alpha": alpha, "Delta": delta, **{f"d{c}": row[c] for c in COEFFS}})
    if all(delta in tables for delta in (0, 4, 8)):
        common = tables[0].index.intersection(tables[4].index).intersection(tables[8].index)
        slope_0_4 = (tables[4].loc[common] - tables[0].loc[common]) / 4.0
        slope_0_8 = (tables[8].loc[common] - tables[0].loc[common]) / 8.0
        slope_4_8 = (tables[8].loc[common] - tables[4].loc[common]) / 4.0
        nonlinear = tables[8].loc[common] - (2.0 * tables[4].loc[common] - tables[0].loc[common])
        for mach, alpha in common:
            for interval, slope in (("0_to_4", slope_0_4), ("0_to_8", slope_0_8), ("4_to_8", slope_4_8)):
                row = slope.loc[(mach, alpha)]
                slope_rows.append({"Mach": mach, "Alpha": alpha, "interval": interval,
                                   **{f"{c}_delta_per_deg": row[c] for c in COEFFS}})
            row = nonlinear.loc[(mach, alpha)]
            nonlinear_rows.append({"Mach": mach, "Alpha": alpha,
                                   **{f"{c}_nonlinearity": row[c] for c in COEFFS}})
    diff_cols = ["Mach", "Alpha", "Delta", "dCD", "dCL", "dCM"]
    slope_cols = ["Mach", "Alpha", "interval", "CD_delta_per_deg", "CL_delta_per_deg", "CM_delta_per_deg"]
    nonlinear_cols = ["Mach", "Alpha", "CD_nonlinearity", "CL_nonlinearity", "CM_nonlinearity"]
    return (
        pd.DataFrame(diff_rows, columns=diff_cols),
        pd.DataFrame(slope_rows, columns=slope_cols),
        pd.DataFrame(nonlinear_rows, columns=nonlinear_cols),
    )


def plot_metric_by_alpha(data: pd.DataFrame, ycols: list[str], stem: str, title_prefix: str,
                         series_col: str | None = None) -> None:
    if data.empty:
        return
    for y in ycols:
        fig, ax = plt.subplots(figsize=(7, 5))
        if series_col:
            for (mach, series), g in data.groupby(["Mach", series_col]):
                g = g.sort_values("Alpha")
                label = f"Mach {fmt_num(mach)}, {series_col} {series}"
                ax.plot(g["Alpha"], g[y], marker="o", label=label)
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


def write_summary(df: pd.DataFrame, diagnostics: dict[str, object], diff: pd.DataFrame,
                  slopes: pd.DataFrame, nonlinear: pd.DataFrame, trim: pd.DataFrame) -> None:
    lines = ["# GF単体 空力特性評価", "", "## データ範囲と健全性", ""]
    lines.append(f"- 行数: {diagnostics['rows']}")
    lines.append(f"- Mach: {', '.join(fmt_num(x) for x in diagnostics['machs'])}")
    lines.append(f"- Alpha [deg]（正: nose-down）: {', '.join(fmt_num(x) for x in diagnostics['alphas'])}")
    lines.append(f"- Delta [deg]（正: nose-upフィン偏向）: {', '.join(fmt_num(x) for x in diagnostics['deltas'])}")
    nan = diagnostics['nan_counts']
    lines.append(f"- NaN数: " + ", ".join(f"{k}={int(v)}" for k, v in nan.items()))
    lines.append(f"- 重複格子行数: {len(diagnostics['duplicate_rows'])}")
    lines.append(f"- 欠けた格子点数（全Mach×全Alpha×全Deltaの直積基準）: {len(diagnostics['missing_grid_points'])}")
    lines += ["", "## 舵角効果と舵効き", ""]
    if diff.empty:
        lines.append("- Delta=0°を基準にしたDelta=+4°/+8°差分を評価できる共通格子点はありません。")
    else:
        for delta in sorted(diff["Delta"].unique()):
            subset = diff[diff["Delta"] == delta]
            lines.append(f"- Delta={fmt_num(delta)}° と Delta=0° の差分 ΔC を {len(subset)} 点で評価しました。")
    if slopes.empty:
        lines.append("- Delta=0°/4°/8°の共通格子点がなく、区間舵効きは評価できません。")
    else:
        points = len(slopes[slopes["interval"] == "0_to_4"])
        lines.append(f"- [C(4)-C(0)]/4°、[C(8)-C(0)]/8°、[C(8)-C(4)]/4°を各 {points} 点で評価しました。これらは片側データによる区間平均舵効きであり、中央差分ではありません。")
        lines.append(f"- CL_delta範囲（全区間）: {slopes['CL_delta_per_deg'].min():.6g} ～ {slopes['CL_delta_per_deg'].max():.6g} /deg")
        lines.append(f"- CM_delta範囲（全区間）: {slopes['CM_delta_per_deg'].min():.6g} ～ {slopes['CM_delta_per_deg'].max():.6g} /deg")
    if not nonlinear.empty:
        lines.append(f"- 8°での線形外挿からの偏差 C(8)-[2C(4)-C(0)] を {len(nonlinear)} 点で評価しました。これは舵角応答の非線形性・干渉の指標です。")
    lines.append("- 負舵角データがないため、正負舵角の偶成分やDelta=0°まわりの中央差分は評価しません。")
    lines += ["", "## Alpha_finによる整理", ""]
    lines.append("- Alpha_fin=Alpha-Delta を補助指標として追加し、CD/CL/CMをAlpha_finに対して描画しました。")
    alpha_fin_sets = [set(group["Alpha_fin"]) for _, group in df.groupby("Delta")]
    common_alpha_fin = set.intersection(*alpha_fin_sets) if alpha_fin_sets else set()
    if common_alpha_fin:
        lines.append(f"- 全Deltaに共通するAlpha_finは {', '.join(fmt_num(x) for x in sorted(common_alpha_fin))}°です。曲線が一致しない場合、胴体寄与・干渉・非線形性により、ロケット全体の係数は単純なフィン局所迎角モデルだけでは表せません。")
    else:
        lines.append("- Delta間で共通するAlpha_fin格子点がないため、同一Alpha_finでの係数一致を直接判定できません。診断図による傾向確認に限られ、単純な局所迎角モデルで整理可能とは断定しません。")
    lines += ["", "## トリム・静安定性", ""]
    if trim.empty:
        lines.append("- データ範囲内でCM=0の符号交差が見つからず、外挿なしではトリム迎角を評価できません。")
    else:
        stable_count = int(trim['static_stable'].sum())
        lines.append(f"- データ範囲内の線形補間で {len(trim)} 個のCM=0トリムを検出しました。")
        lines.append(f"- CSV規約では dCM/dAlpha < 0 を静安定とし、静安定判定は {stable_count}/{len(trim)} 件です。")
    lines += ["", "## データ上の制約", "", "- 評価はCSVの符号規約をそのまま用い、符号変換は行っていません。", "- 格子が疎な場合、差分・トリム勾配は局所線形近似に依存します。PF比較、軌道計算、制御器実装は含みません。"]
    (OUTDIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    df = pd.read_csv("gf_aero_rev.csv")
    ensure_columns(df)
    df = df[REQUIRED_COLUMNS].copy()
    df["Alpha_fin"] = df["Alpha"] - df["Delta"]
    missing, diagnostics = grid_diagnostics(df)
    missing.to_csv(OUTDIR / "missing_grid_points.csv", index=False)
    diagnostics["duplicate_rows"].to_csv(OUTDIR / "duplicate_rows.csv", index=False)
    df.to_csv(OUTDIR / "gf_aero_with_alpha_fin.csv", index=False)
    save_basic_plots(df)
    save_alpha_fin_plots(df)
    diff, slopes, nonlinear = delta_effects(df)
    diff.to_csv(OUTDIR / "delta_effects_vs_delta0.csv", index=False)
    slopes.to_csv(OUTDIR / "control_effectiveness_intervals.csv", index=False)
    nonlinear.to_csv(OUTDIR / "deflection_nonlinearity.csv", index=False)
    plot_metric_by_alpha(diff, ["dCD", "dCL", "dCM"], "delta_effect", "Delta effect relative to Delta=0", "Delta")
    plot_metric_by_alpha(slopes, ["CL_delta_per_deg", "CM_delta_per_deg"], "control_effectiveness", "Interval control effectiveness", "interval")
    plot_metric_by_alpha(nonlinear, ["CD_nonlinearity", "CL_nonlinearity", "CM_nonlinearity"], "deflection_nonlinearity", "Departure from 0-to-4 deg linear response")
    trim = trim_static_stability(df)
    trim.to_csv(OUTDIR / "trim_static_stability.csv", index=False)
    write_summary(df, diagnostics, diff, slopes, nonlinear, trim)
    print(f"Wrote evaluation outputs to {OUTDIR}")


if __name__ == "__main__":
    main()

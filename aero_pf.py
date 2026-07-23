import os

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

csv_path = os.path.join(
    os.path.dirname(__file__),
    "pf_aero.csv"
)

# pf_aero.csv is CFD data with the following sign convention:
#   +Alpha: nose-down angle of attack
#   +CL: downward aerodynamic force
#   +CM: nose-down pitching moment
# aero_coeff() converts that table internally and returns coefficients in the
# equations-of-motion convention:
#   +alpha_deg: nose-up angle of attack
#   +CL: upward aerodynamic force
#   +CM: nose-up pitching moment

df = pd.read_csv(csv_path)

# Remove any residual zero-angle CL/CM offsets at each Mach while preserving
# the CD value at zero angle as base drag.
zero_alpha = df[df["Alpha"] == 0.0].set_index("Mach")
for coeff_name in ("CL", "CM"):
    df[coeff_name] = df.apply(
        lambda row: row[coeff_name] - zero_alpha.loc[row["Mach"], coeff_name],
        axis=1,
    )

points = np.column_stack(
    (
        df["Mach"],
        df["Alpha"]
    )
)

CD_interp = LinearNDInterpolator(
    points,
    df["CD"]
)

CL_interp = LinearNDInterpolator(
    points,
    df["CL"]
)

CM_interp = LinearNDInterpolator(
    points,
    df["CM"]
)

ALPHA_LIMIT_DEG = 10.0
MACH_MIN = float(df["Mach"].min())
MACH_MAX = float(df["Mach"].max())


def aero_coeff(M, alpha_deg):
    """Return CD, CL, CM in the EOM sign convention.

    The input alpha_deg is positive for a nose-up angle of attack. The table is
    only valid for |alpha| <= 10 deg; callers should terminate integration at
    that limit rather than continuing with saturated coefficients.
    """

    if abs(alpha_deg) > ALPHA_LIMIT_DEG:
        raise ValueError(
            f"alpha_deg={alpha_deg} is outside the valid aero range "
            f"|alpha| <= {ALPHA_LIMIT_DEG} deg"
        )

    M = np.clip(M, MACH_MIN, MACH_MAX)

    alpha_abs_deg = abs(alpha_deg)
    alpha_sign = np.sign(alpha_deg)

    CD = float(CD_interp(M, alpha_abs_deg))
    CL = alpha_sign*float(CL_interp(M, alpha_abs_deg))
    CM = alpha_sign*float(CM_interp(M, alpha_abs_deg))

    return CD, CL, CM

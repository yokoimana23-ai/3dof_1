import pandas as pd
import numpy as np
import os

from scipy.interpolate import LinearNDInterpolator

csv_path = os.path.join(
    os.path.dirname(__file__),
    "pf_aero.csv"
)

df = pd.read_csv(csv_path)

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

def aero_coeff(M, alpha_deg):

    M = np.clip(M, 0.6, 2.0)
    alpha_deg = np.clip(alpha_deg, 0.0, 10.0)

    CD = float(CD_interp(M, alpha_deg))
    CL = float(CL_interp(M, alpha_deg))
    CM = float(CM_interp(M, alpha_deg))

    return CD, CL, CM
#!/usr/bin/env python3
"""TRM — Plots für den Praktikumsbericht.
Liest die in 'auswertung.py' erzeugte 'ergebnisse.json' und produziert
sämtliche Abbildungen als PDF und PNG im Output-Ordner.
"""

from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from pathlib import Path

OUT = Path(__file__).parent

with open(OUT / "ergebnisse.json") as f:
    R = json.load(f)

# Globale Plotparameter (wissenschaftlicher Stil)
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.formatter.use_locale": False,
    "errorbar.capsize": 2.5,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def german(x, _pos=None):
    """Komma als Dezimaltrenner."""
    s = f"{x:g}"
    return s.replace(".", ",")


# ---------------------------------------------------------------
# Abb. 1: Statische Bestimmung D* (Puppe) — M(φ)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.4, 3.0))
phi_deg = np.array(R["phi_deg_p"])
F_betrag = np.abs(np.array(R["F_p"]))
M = np.array(R["M_p"]) * 1000  # in mNm
phi_rad = np.deg2rad(phi_deg)

F_err = np.sqrt((0.01 * F_betrag)**2 + (0.02/(2*np.sqrt(6)))**2)
M_err = np.sqrt((F_err * R["r_p"])**2 + (F_betrag * 0.001)**2) * 1000

ax.errorbar(phi_deg, M, yerr=M_err, fmt='o', color='tab:blue',
            mec='tab:blue', mfc='white', mew=1.0, markersize=4.5,
            elinewidth=0.8, label='Messpunkte')
phi_fit = np.linspace(-200, 200, 200)
M_fit = (R["fit_p_a0"] + R["fit_p_a1"] * np.deg2rad(phi_fit)) * 1000
ax.plot(phi_fit, M_fit, '-', color='tab:red', lw=1.0, label='Lin. Regression')
ax.set_xlabel(r'Auslenkungswinkel $\varphi$ in °')
ax.set_ylabel(r'Drehmoment $M$ in mNm')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.xaxis.set_major_locator(MultipleLocator(45))
ax.xaxis.set_major_formatter(FuncFormatter(german))
ax.yaxis.set_major_formatter(FuncFormatter(german))
ax.set_xlim(-210, 210)
ax.legend(loc='lower left', fontsize=9, frameon=False)
fig.savefig(OUT / "abb_statisch_puppe.pdf")
fig.savefig(OUT / "abb_statisch_puppe.png")
plt.close(fig)


# ---------------------------------------------------------------
# Abb. 2: T² vs Jz für die kleine Drehachse (Aufg. 11)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.4, 3.0))
abst_cm = np.array(R["abst_cm"])
m_paar = 0.0974
abst_m = abst_cm / 100
Jz = m_paar * abst_m**2 * 1e6   # in 10⁻⁶ kg·m²

T90 = np.array(R["T_90"])
T135 = np.array(R["T_135"])

# Unsicherheit auf T² (volle Periode): 2T·u_T. An der kleinen Drehachse
# wurde mit Lichtschranke gemessen (digitale Auflösung d = 1 ms,
# Rechteckverteilung), daher u_T = 2·d/(2·√3) = d/√3 ≈ 0,58 ms pro
# Halbschwingungsmessung × 2 (volle Periode) ≈ 1,15 ms; für einen
# einzelnen Punkt aus einer Auslenkung nehmen wir konservativ den
# Beitrag der Einzelmessung (ohne Mittelung über die Amplituden).
u_T_val = 2 * (0.001 / (2 * np.sqrt(3)))   # ≈ 0,58 ms
ax.errorbar(Jz, T90**2, yerr=2*T90*u_T_val, fmt='s', color='tab:blue',
            mec='tab:blue', mfc='white', mew=1.0, markersize=5,
            label=r'$\varphi_0 \approx 90°$', elinewidth=0.8)
ax.errorbar(Jz, T135**2, yerr=2*T135*u_T_val, fmt='^', color='tab:green',
            mec='tab:green', mfc='white', mew=1.0, markersize=5,
            label=r'$\varphi_0 \approx 135°$', elinewidth=0.8)

x_fit = np.linspace(0, max(Jz)*1.05, 100)
y_fit = R["lin_a0_d"] + R["lin_a1_d"] * x_fit * 1e-6
ax.plot(x_fit, y_fit, '-', color='tab:red', lw=1.0, label='Lin. Regression')

ax.set_xlabel(r'$J_z = 2 m r^2$ in $10^{-6}\,\mathrm{kg\,m^2}$')
ax.set_ylabel(r'$T^2$ in $\mathrm{s^2}$')
ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.xaxis.set_major_formatter(FuncFormatter(german))
ax.yaxis.set_major_formatter(FuncFormatter(german))
fig.savefig(OUT / "abb_dynamisch_puppe.pdf")
fig.savefig(OUT / "abb_dynamisch_puppe.png")
plt.close(fig)


# ---------------------------------------------------------------
# Abb. 3: Statische Bestimmung D* (großer Drehteller) — M(φ)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.4, 3.0))
phi_deg_d = np.array(R["phi_deg_d"])
M_d = np.array(R["M_d"])
F_d = np.array(R["F_d"])

F_err_d = np.sqrt((0.01 * np.abs(F_d))**2 + (0.2/(2*np.sqrt(6)))**2)
M_err_d = np.sqrt((F_err_d * R["r_d"])**2 + (np.abs(F_d) * 0.002)**2)

ax.errorbar(phi_deg_d, M_d, yerr=M_err_d, fmt='o', color='tab:blue',
            mec='tab:blue', mfc='white', mew=1.0, markersize=4.5,
            elinewidth=0.8, label='Messpunkte')
phi_fit = np.linspace(-100, 100, 200)
M_fit = R["fit_d_a0"] + R["fit_d_a1"] * np.deg2rad(phi_fit)
ax.plot(phi_fit, M_fit, '-', color='tab:red', lw=1.0, label='Lin. Regression')
ax.set_xlabel(r'Auslenkungswinkel $\varphi$ in °')
ax.set_ylabel(r'Drehmoment $M$ in Nm')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.xaxis.set_major_locator(MultipleLocator(30))
ax.xaxis.set_major_formatter(FuncFormatter(german))
ax.yaxis.set_major_formatter(FuncFormatter(german))
ax.set_xlim(-100, 100)
ax.legend(loc='lower left', fontsize=9, frameon=False)
fig.savefig(OUT / "abb_statisch_drehteller.pdf")
fig.savefig(OUT / "abb_statisch_drehteller.png")
plt.close(fig)


# ---------------------------------------------------------------
# Abb. 4: Vergleich aller J_Mensch (Arme angelegt + ausgestreckt)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.4))
labels = [
    'Messung\n($D^*_{\\mathrm{stat}}$)',
    'Messung\n($D^*_{\\mathrm{dyn}}$)',
    'Modell\n(Zyl./Kugel)',
    'Zyl.-Näh.\n(Gl. 17)',
    'Extrapol.\n(Gl. 15)',
    'Extrapol.\n(Gl. 16)',
]
vals_g = [R["J_men_g_stat"], R["J_men_g"], R["J_men_theo_an"],
          R["J_men_zyl_simple"], R["J_extra_a"], R["J_extra_b"]]
errs_g = [R["u_J_men_g_stat"], R["u_J_men_g"], R["u_J_men_theo_an"],
          R["u_J_men_zyl_simple"], R["u_J_extra_a"], R["u_J_extra_b"]]
vals_a = [R["J_men_a_stat"], R["J_men_a"], R["J_men_theo_ges"], None, None, None]
errs_a = [R["u_J_men_a_stat"], R["u_J_men_a"], R["u_J_men_theo_ges"], None, None, None]

x = np.arange(len(labels))
width = 0.38
ax.bar(x - width/2, vals_g, width, yerr=errs_g, capsize=3,
       color='tab:blue', alpha=0.75, edgecolor='tab:blue',
       label='Arme angelegt', ecolor='black')
mask = np.array([v is not None for v in vals_a])
ax.bar(x[mask] + width/2,
       [v for v, m in zip(vals_a, mask) if m], width,
       yerr=[e for e, m in zip(errs_a, mask) if m], capsize=3,
       color='tab:orange', alpha=0.75, edgecolor='tab:orange',
       label='Arme ausgestreckt', ecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel(r'$J_\mathrm{Mensch}$ in $\mathrm{kg\,m^2}$')
ax.legend(loc='upper right', fontsize=9, frameon=False)
ax.yaxis.set_major_formatter(FuncFormatter(german))
ax.set_ylim(0, 4.0)
fig.savefig(OUT / "abb_vergleich_mensch.pdf")
fig.savefig(OUT / "abb_vergleich_mensch.png")
plt.close(fig)


# ---------------------------------------------------------------
# Abb. 5: Schwingungsdauern (Mensch) — drei Wiederholungen je Haltung
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.0, 2.8))
T_g = np.array([5.586, 5.566, 5.604])
T_a = np.array([7.7275, 7.78, 7.918])
ax.errorbar([1, 2, 3], T_g, yerr=R["u_T_men_g"], fmt='o', color='tab:blue',
            mec='tab:blue', mfc='white', mew=1.0, markersize=5,
            label='Arme angelegt', elinewidth=0.8)
ax.errorbar([1, 2, 3], T_a, yerr=R["u_T_men_a"], fmt='s', color='tab:orange',
            mec='tab:orange', mfc='white', mew=1.0, markersize=5,
            label='Arme ausgestreckt', elinewidth=0.8)
ax.axhline(np.mean(T_g), color='tab:blue', lw=0.6, alpha=0.5,
           linestyle=':', label='Mittelwert')
ax.axhline(np.mean(T_a), color='tab:orange', lw=0.6, alpha=0.5, linestyle=':')
ax.set_xlabel('Messung Nr.')
ax.set_ylabel(r'$T$ in s')
ax.set_xticks([1, 2, 3])
ax.set_xlim(0.5, 3.5)
ax.legend(loc='center right', fontsize=8.5, frameon=False)
ax.yaxis.set_major_formatter(FuncFormatter(german))
fig.savefig(OUT / "abb_schwingungsdauer_mensch.pdf")
fig.savefig(OUT / "abb_schwingungsdauer_mensch.png")
plt.close(fig)

print("Plots geschrieben:")
for p in sorted(OUT.glob("abb_*.pdf")):
    print(f"  {p.name}")

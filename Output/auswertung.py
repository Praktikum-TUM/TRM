#!/usr/bin/env python3
"""
TRM - Trägheitsmoment - Auswertung mit Fehlerrechnung
Physikalisches Praktikum TU München
Team Tamastian: Sebastian Richter, Tamino Bruckmoser
Durchführung: 2026-05-05

Diese Datei führt die komplette numerische Auswertung durch und schreibt
alle Ergebnisse in 'ergebnisse.txt'. Plots werden in 'plots.py' erzeugt.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
RESULTS = OUT / "ergebnisse.txt"
LOG_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    LOG_LINES.append(msg)


# ------------------------------------------------------------------
# Hilfsfunktionen (Fehlerrechnung)
# ------------------------------------------------------------------

def linreg_unweighted(x: np.ndarray, y: np.ndarray) -> dict:
    """Ungewichtete lineare Regression y = a0 + a1*x. Liefert Werte und
    Unsicherheiten gemäß Gl. (34)/(35) der Praktikumshinweise."""
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx, sxy = (x * x).sum(), (x * y).sum()
    D = n * sxx - sx * sx
    a1 = (n * sxy - sx * sy) / D
    a0 = (sxx * sy - sx * sxy) / D
    yhat = a0 + a1 * x
    sigma_y = np.sqrt(((y - yhat) ** 2).sum() / (n - 2))
    u_a1 = sigma_y * np.sqrt(n / D)
    u_a0 = sigma_y * np.sqrt(sxx / D)
    return dict(a0=a0, a1=a1, u_a0=u_a0, u_a1=u_a1, sigma_y=sigma_y, n=n)


def round_unc(value: float, unc: float, digits: int = 2) -> tuple[float, float, int]:
    """Runde Unsicherheit auf 'digits' signifikante Stellen, Wert auf gleiche Stelle.
    Gibt (gerundeter Wert, gerundete Unsicherheit, Anzahl Nachkommastellen) zurück."""
    if unc <= 0 or not np.isfinite(unc):
        return value, unc, 0
    exp = int(np.floor(np.log10(abs(unc))))
    factor = 10 ** (exp - (digits - 1))
    unc_r = round(unc / factor) * factor
    val_r = round(value / factor) * factor
    nd = max(0, -(exp - (digits - 1)))
    return val_r, unc_r, nd


def fmt(value: float, unc: float, unit: str = "") -> str:
    val_r, unc_r, nd = round_unc(value, unc)
    s = f"({val_r:.{nd}f} ± {unc_r:.{nd}f})"
    return f"{s} {unit}".strip()


# ------------------------------------------------------------------
# Konstanten
# ------------------------------------------------------------------

# Federwaage Puppe (kleine Federwaage, ~1 N Skala)
# Annahme: Klasse 1 (1 % vom Endwert) + Ableseunsicherheit Dreieck
F_END_KLEIN = 1.0           # N (Endwert der kleinen Federwaage, Annahme)
A_KLEIN = 0.02              # N (Strichabstand, 0,02 N)
# Federwaage Drehteller (große Federwaage, ~20 N Skala)
F_END_GROSS = 20.0          # N (Endwert)
A_GROSS = 0.2               # N (Strichabstand)

# Zeitmessung:
# - Kleine Drehachse (Puppe, Aufg. 11+12): elektronische Lichtschranke,
#   digitale Anzeige mit Auflösung d = 1 ms ⇒ Rechteckverteilung
#   u_B(T_halb) = d/(2·√3). Reaktionszeit eines Bedieners spielt hier
#   keine Rolle, da der Trigger optoelektronisch erfolgt.
# - Großer Drehteller (Aufg. 15+16): Handstoppuhr mit Computer-Anzeige
#   ⇒ Typ-B durch menschliche Reaktionszeit. Bei einem regelmäßig pendelnden,
#   optisch klar erkennbaren Maximum (vorhersehbares Ereignis) ca. 0,05 s
#   pro Trigger; bei einem unvorhergesehenen Trigger 0,2 s.
D_LICHTSCHRANKE = 0.001     # s (Auflösung der Lichtschranken-Anzeige)
U_REACTION      = 0.05      # s pro Trigger (Stoppuhr, vorhersehbares Maximum)

# Winkel-Auflösung Drehscheibe (Praktikum, Annahme 1° Skalierung)
A_PHI_DEG = 1.0             # Strichabstand 1°


def u_F_klein(F: float) -> float:
    """Typ-B-Unsicherheit Federwaage Puppe: Skalierung + Auflösung (Dreieck)."""
    u_skal = 0.01 * F                     # 1 % Skalierungsklasse 1
    u_aufl = A_KLEIN / (2 * np.sqrt(6))   # Dreieck (Analoganzeige)
    return np.sqrt(u_skal ** 2 + u_aufl ** 2)


def u_F_gross(F: float) -> float:
    u_skal = 0.01 * F
    u_aufl = A_GROSS / (2 * np.sqrt(6))
    return np.sqrt(u_skal ** 2 + u_aufl ** 2)


def u_phi_rad(phi_rad: float) -> float:
    """Unsicherheit Auslenkungswinkel (Dreieck), in Rad."""
    a = np.deg2rad(A_PHI_DEG)
    return a / (2 * np.sqrt(6))


def u_T_einzel(t_total: float, n: int) -> float:
    """Typ-B-Unsicherheit der Periodendauer aus Handstoppuhr (großer Drehteller).
    Über n (Halb-)Schwingungen treten zwei unabhängige Reaktionen auf
    (Start und Stop), die quadratisch addiert und auf eine Periode normiert
    werden."""
    u_t = np.sqrt(2) * U_REACTION         # Stoppuhr-Reaktion (Start+Stop)
    return u_t / n


def u_T_lichtschranke_halb() -> float:
    """Typ-B-Unsicherheit einer einzelnen Halbschwingungsmessung an der
    kleinen Drehachse (digitale Lichtschranke, Rechteckverteilung)."""
    return D_LICHTSCHRANKE / (2 * np.sqrt(3))



# ------------------------------------------------------------------
# Aufgabe 10: Statische Bestimmung D* (Puppe / kleine Drehachse)
# ------------------------------------------------------------------
log("=" * 72)
log("Aufgabe 10: Statische Bestimmung D* der kleinen Drehachse (Puppe)")
log("=" * 72)

# Sheet 4.1.1 — Hebelarm 15,5 cm, Tangentialkraft in N gegen Auslenkungswinkel
# Hinweis: in den Rohdaten wurde nur der Betrag von F notiert; das Vorzeichen
# wird aus dem Drehsinn (sign(phi)) rekonstruiert: rückstellendes Drehmoment
# zeigt entgegen der Auslenkung, also M = F·r·sign(-phi). Die Federwaage zieht
# bei phi>0 in andere Richtung als bei phi<0 — das wird hier umsigniert.
phi_deg_p = np.array([-180, -135, -90, -45, 0, 45, 90, 135, 180])
F_betrag_p = np.array([0.44, 0.35, 0.23, 0.14, 0.0, 0.17, 0.29, 0.40, 0.52])
F_p = -np.sign(phi_deg_p) * F_betrag_p   # F·r·sign(-phi) = rückstell. Drehmoment
r_p = 0.155  # m
u_r_p = 0.001  # m (Lineal)

phi_rad_p = np.deg2rad(phi_deg_p)
M_p = F_p * r_p  # Drehmoment in Nm
# M ≈ -D*·phi → Steigung im Plot M(phi) ist -D*; wir kehren das Vorzeichen am Ende

# Lineare Regression M = a0 + a1·phi.
# Bei umsignierten Werten (F·sign(-phi)) ist M_rückstellend > 0 für phi < 0,
# d.h. M = -D*·phi, also a1 = -D*. Wir nehmen |a1|.
fit_p = linreg_unweighted(phi_rad_p, M_p)
D_p_stat = abs(fit_p["a1"])
u_D_p_stat_fit = fit_p["u_a1"]
u_a0_p = fit_p["u_a0"]
log(f"  Hebelarm r = {r_p*100:.1f} cm, u(r) = {u_r_p*1000:.1f} mm")
log(f"  Fit: M = a0 + a1·phi")
log(f"    a0 = ({fit_p['a0']*1e3:.3f} ± {u_a0_p*1e3:.3f}) mNm  (Achsenabschnitt)")
log(f"    a1 = ({fit_p['a1']*1e3:.4f} ± {u_D_p_stat_fit*1e3:.4f}) mNm/rad  (Steigung = D*)")

# Zusätzliche Typ-B-Unsicherheit aus r und Federkraft:
# Da relativer Anteil von r quadratisch addiert wird:
rel_r = u_r_p / r_p
u_D_p_stat = D_p_stat * np.sqrt((u_D_p_stat_fit / D_p_stat) ** 2 + rel_r ** 2)

# Plus Skalierungsunsicherheit der Federwaage (1 %, vollkorreliert -> direkt 1%):
u_D_p_stat = np.sqrt(u_D_p_stat ** 2 + (0.01 * D_p_stat) ** 2)

log(f"  D*_stat (mit Hebelarm- und Federwaagen-Skalierungs-Unsicherheit):")
log(f"    D*_stat = {fmt(D_p_stat*1e3, u_D_p_stat*1e3, 'mNm/rad')}")
log(f"    D*_stat = {fmt(D_p_stat, u_D_p_stat, 'Nm/rad')}")

# Ursprung im Vertrauensbereich?
log(f"  Achsenabschnitt a0 = {fit_p['a0']*1e3:.3f} mNm, "
    f"|a0|/u(a0) = {abs(fit_p['a0'])/u_a0_p:.2f} ⇒ "
    f"{'innerhalb' if abs(fit_p['a0']) < 2*u_a0_p else 'außerhalb'} der 2σ-Umgebung des Ursprungs.")
log("")


# ------------------------------------------------------------------
# Aufgabe 11: Dynamische Bestimmung D*, J0 (Puppe)
# ------------------------------------------------------------------
log("=" * 72)
log("Aufgabe 11: Dynamische Bestimmung D* und J0 (kleine Drehachse)")
log("=" * 72)

# Sheet 4.1.2 — Halbschwingungen bei 90° und 135° Auslenkung
# Werte sind Halbschwingungen (Notiz: "Halbeschwingung")
abst_cm = np.array([0, 5, 10, 15, 20])
T_halb_90 = np.array([0.364, 0.485, 0.741, 1.012, 1.321])
T_halb_135 = np.array([0.363, 0.483, 0.738, 1.000, 1.321])

# Volle Schwingungsdauer:
T_90 = 2 * T_halb_90
T_135 = 2 * T_halb_135

# Mittelwert über die beiden Auslenkungen
T_mean = 0.5 * (T_90 + T_135)
# Streuung als Typ-A (zwei Messungen pro Abstand)
diffs = np.abs(T_90 - T_135)
sigma_T_A = np.maximum(diffs / 2, 0.001)        # halbe Spannweite ≈ Streuung

# Typ B: Lichtschranke, digital, Auflösung d = 1 ms (Rechteck-Verteilung).
# Eine Halbschwingung wird direkt als ein Zeitintervall zwischen zwei
# Triggerflanken angezeigt ⇒ u(T_halb)_B = d/(2·√3).
# Volle Periode T = 2·T_halb aus einer einzigen Halbschwingungsmessung
# ⇒ u(T)_B = 2·u(T_halb)_B. Mittelung über 2 unabhängige Auslenkungen
# (90°/135°) reduziert um √2.
u_T_halb_B = u_T_lichtschranke_halb()                  # ≈ 0,29 ms
u_T_B      = 2 * u_T_halb_B / np.sqrt(2)               # ≈ 0,41 ms

u_T = np.sqrt(sigma_T_A ** 2 + u_T_B ** 2)
log(f"  T (volle Schwingungsdauer) bei Abstand r=0 cm: {T_mean[0]:.3f} s")
log(f"  Typ-B-Unsicherheit pro T (Lichtschranke 1 ms): u_B = {u_T_B*1000:.2f} ms")

# Trägheitsmoment Jz der Gewichte: m_paar = 97,4 g
m_paar = 0.0974          # kg
u_m_paar = 0.0001        # kg (Waage Auflösung 0,1 g, Annahme)
abst_m = abst_cm / 100
u_abst_m = 0.001         # m

# Jz = 2·m·r² mit m = Masse pro Gewicht, also = m_paar · r²
Jz = m_paar * abst_m ** 2
u_Jz = Jz * np.sqrt((u_m_paar / m_paar) ** 2 + (2 * u_abst_m / np.maximum(abst_m, 1e-9)) ** 2)
u_Jz[0] = u_m_paar * abst_m[0] ** 2 + 2 * m_paar * abst_m[0] * u_abst_m  # bei r=0 trivial 0

# T² gegen Jz auftragen → Steigung 4π²/D*, Achsenabschnitt = (4π²/D*)·J0
T_sq = T_mean ** 2
u_T_sq = 2 * T_mean * u_T

# Gewichtete Regression (1/u(T²)²)
w = 1.0 / u_T_sq ** 2
sw = w.sum()
swx = (w * Jz).sum()
swy = (w * T_sq).sum()
swxx = (w * Jz ** 2).sum()
swxy = (w * Jz * T_sq).sum()
Dlin = sw * swxx - swx ** 2
a1_d = (sw * swxy - swx * swy) / Dlin
a0_d = (swxx * swy - swx * swxy) / Dlin

# Streuung um die Gerade
yhat = a0_d + a1_d * Jz
sigma_y = np.sqrt(((T_sq - yhat) ** 2 * w).sum() / (len(Jz) - 2))
u_a1_d = sigma_y * np.sqrt(sw / Dlin)
u_a0_d = sigma_y * np.sqrt(swxx / Dlin)

log(f"  Lineare Regression T² vs Jz:")
log(f"    Steigung a1 = ({a1_d:.4g} ± {u_a1_d:.2g}) s²/(kg m²)")
log(f"    Achsenabschnitt a0 = ({a0_d:.4g} ± {u_a0_d:.2g}) s²")

D_p_dyn = 4 * np.pi ** 2 / a1_d
u_D_p_dyn = D_p_dyn * (u_a1_d / a1_d)
J0_p = a0_d / a1_d
u_J0_p = J0_p * np.sqrt((u_a0_d / a0_d) ** 2 + (u_a1_d / a1_d) ** 2)

log(f"  D*_dyn = {fmt(D_p_dyn*1e3, u_D_p_dyn*1e3, 'mNm/rad')}")
log(f"  J0     = {fmt(J0_p*1e6, u_J0_p*1e6, 'g·cm² (×10⁻⁶ kg·m²)')}")

# Vergleich
diff = abs(D_p_stat - D_p_dyn)
u_diff = np.sqrt(u_D_p_stat ** 2 + u_D_p_dyn ** 2)
log(f"  Vergleich D*_stat vs D*_dyn:  Δ = {diff*1e3:.3f} mNm/rad, "
    f"u(Δ) = {u_diff*1e3:.3f} mNm/rad   "
    f"⇒ {'verträglich' if diff < 2*u_diff else 'nicht verträglich'} innerhalb 2σ")
# Wir verwenden im Folgenden D*_dyn (dynamisch ist die direktere Bestimmung im Schwingungs-Setup)
D_p = D_p_dyn
u_D_p = u_D_p_dyn
log(f"  → Wir rechnen weiter mit D*_dyn.")
log("")


# ------------------------------------------------------------------
# Aufgabe 12: Trägheitsmoment der Puppe
# ------------------------------------------------------------------
log("=" * 72)
log("Aufgabe 12: Trägheitsmoment der Puppe")
log("=" * 72)

# Sheet 4.1.3 — Halbschwingungen mit Puppe (zwei Haltungen, zwei Auslenkungen)
T_pup_gerade = 2 * np.array([0.405, 0.405])      # Halbschw. → volle T
T_pup_arme = 2 * np.array([0.500, 0.496])
T_pup_g = T_pup_gerade.mean()
T_pup_a = T_pup_arme.mean()

# Unsicherheit auf T: Typ-A pro Haltung getrennt (n=2, Student-t/√n = 1,30
# bei 68 % Vertrauensniveau), kombiniert mit Lichtschranken-Typ-B (s.o.).
t_n2 = 1.30
u_T_pup_g_A = t_n2 * np.std(T_pup_gerade, ddof=1) / np.sqrt(2)
u_T_pup_a_A = t_n2 * np.std(T_pup_arme,   ddof=1) / np.sqrt(2)
u_T_pup_g   = np.sqrt(u_T_pup_g_A**2 + u_T_B**2)
u_T_pup_a   = np.sqrt(u_T_pup_a_A**2 + u_T_B**2)
log(f"  u(T_Puppe, angelegt)    = {u_T_pup_g*1000:.2f} ms "
    f"(A: {u_T_pup_g_A*1000:.2f}, B: {u_T_B*1000:.2f})")
log(f"  u(T_Puppe, ausgestreckt) = {u_T_pup_a*1000:.2f} ms "
    f"(A: {u_T_pup_a_A*1000:.2f}, B: {u_T_B*1000:.2f})")

# T0: ohne Puppe, mit Hilfsstab → das ist Sheet 4.1.2 mit Abstand 0:
T0 = T_mean[0]
u_T0 = u_T[0]

# J_Puppe = D*/(4π²) · (T_pup² - T0²)
def J_aus_diff(T_pup, u_T_pup_val, T0_val, u_T0_val, D, u_D):
    delta = T_pup ** 2 - T0_val ** 2
    u_delta = np.sqrt((2 * T_pup * u_T_pup_val) ** 2 + (2 * T0_val * u_T0_val) ** 2)
    factor = D / (4 * np.pi ** 2)
    u_factor = u_D / (4 * np.pi ** 2)
    J = factor * delta
    u_J = np.sqrt((u_factor * delta) ** 2 + (factor * u_delta) ** 2)
    return J, u_J


J_pup_g, u_J_pup_g = J_aus_diff(T_pup_g, u_T_pup_g, T0, u_T0, D_p, u_D_p)
J_pup_a, u_J_pup_a = J_aus_diff(T_pup_a, u_T_pup_a, T0, u_T0, D_p, u_D_p)
log(f"  T(Puppe gerade) = {T_pup_g:.4f} s,  T(Puppe Arme) = {T_pup_a:.4f} s")
log(f"  T0 (ohne Puppe) = {T0:.4f} s")
log(f"  J_Puppe (Arme angelegt)    = {fmt(J_pup_g*1e6, u_J_pup_g*1e6, '×10⁻⁶ kg·m²')}")
log(f"  J_Puppe (Arme ausgestreckt) = {fmt(J_pup_a*1e6, u_J_pup_a*1e6, '×10⁻⁶ kg·m²')}")

ratio_pup = J_pup_a / J_pup_g
u_ratio_pup = ratio_pup * np.sqrt((u_J_pup_a/J_pup_a)**2 + (u_J_pup_g/J_pup_g)**2)
log(f"  Verhältnis J_arme / J_gerade = {fmt(ratio_pup, u_ratio_pup)}")
log("")


# ------------------------------------------------------------------
# Aufgabe 13: Statische D* — großer Drehteller
# ------------------------------------------------------------------
log("=" * 72)
log("Aufgabe 13: Statische Bestimmung D* großer Drehteller")
log("=" * 72)

# Sheet 4.2.4 — Hebelarm = Radius des Drehtellers = 30 cm (Annahme)
# Wieder Beträge der Federwaage; Vorzeichen aus sign(-phi).
phi_deg_d = np.array([-90, -80, -70, -60, -50, -40, -30, -20, -10,
                      0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
F_betrag_d = np.array([13.5, 12.5, 10.5, 9.8, 7.9, 6.4, 4.7, 3.5, 1.8,
                       0.0, 1.6, 2.5, 3.2, 4.0, 5.5, 7.0, 7.8, 9.5, 10.0])
F_d = -np.sign(phi_deg_d) * F_betrag_d
r_d = 0.30   # m, Radius des Drehtellers
u_r_d = 0.002  # m

phi_rad_d = np.deg2rad(phi_deg_d)
M_d = F_d * r_d

fit_d = linreg_unweighted(phi_rad_d, M_d)
D_d_stat = abs(fit_d["a1"])
u_D_d_stat_fit = fit_d["u_a1"]

# Gesamt-Unsicherheit inkl. r und Federwaagen-Skalierung
u_D_d_stat = D_d_stat * np.sqrt((u_D_d_stat_fit/D_d_stat)**2
                                + (u_r_d/r_d)**2 + 0.01**2)
log(f"  Hebelarm r = {r_d*100:.0f} cm")
log(f"  Fit: a0 = ({fit_d['a0']:.3f} ± {fit_d['u_a0']:.3f}) Nm")
log(f"       a1 = ({fit_d['a1']:.3f} ± {u_D_d_stat_fit:.3f}) Nm/rad")
log(f"  D*_stat = {fmt(D_d_stat, u_D_d_stat, 'Nm/rad')}")
log("")


# ------------------------------------------------------------------
# Aufgabe 14: Eigenträgheitsmoment des Drehtellers (geometrisch)
# ------------------------------------------------------------------
log("=" * 72)
log("Aufgabe 14: Eigenträgheitsmoment J0 des Drehtellers (geometrisch)")
log("=" * 72)

rho_Al = 2700.0          # kg/m³
u_rho = 0.0              # exakt (Tabellenwert)

# Aluminium-Scheibe: D = 60 cm, h = 2 cm
R_disk = 0.30
u_R_disk = 0.002
h_disk = 0.02
u_h_disk = 0.001

m_disk = rho_Al * np.pi * R_disk**2 * h_disk
u_m_disk = m_disk * np.sqrt((2*u_R_disk/R_disk)**2 + (u_h_disk/h_disk)**2)
J_disk = 0.5 * m_disk * R_disk**2
u_J_disk = J_disk * np.sqrt((u_m_disk/m_disk)**2 + (2*u_R_disk/R_disk)**2)

# Stange (Zylinder): D = 2,5 cm, h = 75 cm — liegt auf Drehachse, J = 0,5·m·r²
R_st = 0.0125
u_R_st = 0.0005
h_st = 0.75
u_h_st = 0.005
m_st = rho_Al * np.pi * R_st**2 * h_st
J_st = 0.5 * m_st * R_st**2
u_m_st = m_st * np.sqrt((2*u_R_st/R_st)**2 + (u_h_st/h_st)**2)
u_J_st = J_st * np.sqrt((u_m_st/m_st)**2 + (2*u_R_st/R_st)**2)

# Stab (vernachlässigbar laut Notiz)
J0_d = J_disk + J_st
u_J0_d = np.sqrt(u_J_disk**2 + u_J_st**2)
log(f"  Aluminium-Scheibe: m = {fmt(m_disk, u_m_disk, 'kg')}")
log(f"                     J = {fmt(J_disk, u_J_disk, 'kg·m²')}")
log(f"  Stange:            m = {m_st*1000:.2f} g,  J = {J_st*1e6:.2f} ×10⁻⁶ kg·m² (vernachlässigbar)")
log(f"  J0 (gesamt) = {fmt(J0_d, u_J0_d, 'kg·m²')}")
log("")


# ------------------------------------------------------------------
# Aufgabe 15: Dynamische D* großer Drehteller
# ------------------------------------------------------------------
log("=" * 72)
log("Aufgabe 15: Dynamische D* großer Drehteller (mit J0 aus Geometrie)")
log("=" * 72)

# Sheet 4.2.5 — t = 38,97 s für 23 Halbschwingungen (analog zur Puppen-Apparatur)
# Volle Periode = 2 · (t/n)
t_total_d = 38.97
n_d = 23
T_d_halb = t_total_d / n_d
T_d = 2 * T_d_halb
u_T_d = 2 * u_T_einzel(t_total_d, n_d)
log(f"  T_halb = {T_d_halb:.4f} s, volle Periode T = ({T_d:.4f} ± {u_T_d:.4f}) s   "
    f"(aus {n_d} Halbschwingungen in {t_total_d} s)")

D_d_dyn = 4 * np.pi**2 * J0_d / T_d**2
u_D_d_dyn = D_d_dyn * np.sqrt((u_J0_d/J0_d)**2 + (2*u_T_d/T_d)**2)
log(f"  D*_dyn = {fmt(D_d_dyn, u_D_d_dyn, 'Nm/rad')}")

# Vergleich
diff_d = abs(D_d_stat - D_d_dyn)
u_diff_d = np.sqrt(u_D_d_stat**2 + u_D_d_dyn**2)
log(f"  Vergleich D*_stat vs D*_dyn:  Δ = {diff_d:.3f} Nm/rad, "
    f"u(Δ) = {u_diff_d:.3f} Nm/rad   "
    f"⇒ {'verträglich' if diff_d < 2*u_diff_d else 'nicht verträglich'} innerhalb 2σ.")
# Wir nehmen den dynamischen Wert (J0 aus Geometrie konsistent, D*_dyn konsistent zu T)
D_d = D_d_dyn
u_D_d = u_D_d_dyn
log(f"  → Wir rechnen weiter mit D*_dyn.")
log("")


# ------------------------------------------------------------------
# Aufgabe 16: Trägheitsmoment des Menschen (gemessen)
# ------------------------------------------------------------------
log("=" * 72)
log("Aufgabe 16: Trägheitsmoment des Menschen (Messung)")
log("=" * 72)

# Sheet 4.2.7 — drei Wiederholungen je Haltung
t_tot_g = np.array([27.93, 27.83, 28.02])
n_g = np.array([5, 5, 5])
T_men_g_arr = t_tot_g / n_g
T_men_g = T_men_g_arr.mean()
sigma_T_men_g = T_men_g_arr.std(ddof=1)
# Typ-A mit Student-t (n=3, 68 % → t/sqrt(n) = 0,76)
u_T_men_g_A = 0.76 * sigma_T_men_g
u_T_men_g_B = u_T_einzel(np.mean(t_tot_g), 5) / np.sqrt(3)  # gemittelt 3x
u_T_men_g = np.sqrt(u_T_men_g_A**2 + u_T_men_g_B**2)

t_tot_a = np.array([30.91, 31.12, 39.59])
n_a = np.array([4, 4, 5])
T_men_a_arr = t_tot_a / n_a
T_men_a = T_men_a_arr.mean()
sigma_T_men_a = T_men_a_arr.std(ddof=1)
u_T_men_a_A = 0.76 * sigma_T_men_a
u_T_men_a_B = u_T_einzel(np.mean(t_tot_a), np.mean(n_a)) / np.sqrt(3)
u_T_men_a = np.sqrt(u_T_men_a_A**2 + u_T_men_a_B**2)

log(f"  T (Mensch, Arme angelegt)   = {fmt(T_men_g, u_T_men_g, 's')}")
log(f"  T (Mensch, Arme ausgestreckt) = {fmt(T_men_a, u_T_men_a, 's')}")

# Auswertung mit D*_dyn (aus Aufg. 15)
J_men_g, u_J_men_g = J_aus_diff(T_men_g, u_T_men_g, T_d, u_T_d, D_d, u_D_d)
J_men_a, u_J_men_a = J_aus_diff(T_men_a, u_T_men_a, T_d, u_T_d, D_d, u_D_d)
log(f"  Mit D*_dyn = {D_d:.2f} Nm/rad:")
log(f"    J_Mensch (Arme angelegt)    = {fmt(J_men_g, u_J_men_g, 'kg·m²')}")
log(f"    J_Mensch (Arme ausgestreckt) = {fmt(J_men_a, u_J_men_a, 'kg·m²')}")

# Auswertung mit D*_stat (aus Aufg. 13)
J_men_g_stat, u_J_men_g_stat = J_aus_diff(T_men_g, u_T_men_g, T_d, u_T_d, D_d_stat, u_D_d_stat)
J_men_a_stat, u_J_men_a_stat = J_aus_diff(T_men_a, u_T_men_a, T_d, u_T_d, D_d_stat, u_D_d_stat)
log(f"  Mit D*_stat = {D_d_stat:.2f} Nm/rad:")
log(f"    J_Mensch (Arme angelegt)    = {fmt(J_men_g_stat, u_J_men_g_stat, 'kg·m²')}")
log(f"    J_Mensch (Arme ausgestreckt) = {fmt(J_men_a_stat, u_J_men_a_stat, 'kg·m²')}")

log("")


# ------------------------------------------------------------------
# Aufgabe 17: Theoretisches J_Mensch aus Modell (Zylinder + Kugel)
# ------------------------------------------------------------------
log("=" * 72)
log("Aufgabe 17: Theoretisches J_Mensch aus geometrischem Modell")
log("=" * 72)

M_men = 70.0          # kg
u_M_men = 0.5         # kg (Personenwaage)
H_men = 1.83          # m
u_H_men = 0.005

# Massenanteile (Tabelle 2)
ant_kopf = 0.073
ant_rumpf = 0.489
ant_oarm_einzeln = 0.027
ant_uarm_einzeln = 0.017
ant_hand_einzeln = 0.008
ant_oschenkel_einzeln = 0.097
ant_uschenkel_einzeln = 0.053
ant_fuss_einzeln = 0.017

# Geometrie (Sheet 4.2.9), Annahme: erste Zahl = Länge, zweite Zahl = Umfang [cm]
U_kopf = 0.59
L_kopf = 0.22
U_rumpf = 0.83
L_oarm, U_oarm = 0.32, 0.30
L_uarm, U_uarm = 0.47, 0.18
L_oschenkel, U_oschenkel = 0.38, 0.43
L_uschenkel, U_uschenkel = 0.47, 0.32

# Drehachse: vertikale Körperlängsachse durch Schwerpunkt.

# Kopf — Modell: Kugel um den Schwerpunkt
m_kopf = ant_kopf * M_men
R_kopf = U_kopf / (2 * np.pi)
J_kopf = (2/5) * m_kopf * R_kopf**2

# Rumpf — Zylinder, Drehachse parallel zur Achse
m_rumpf = ant_rumpf * M_men
R_rumpf = U_rumpf / (2 * np.pi)
J_rumpf = 0.5 * m_rumpf * R_rumpf**2

# Arme — Zylinder, Drehachse senkrecht zur Achse, Steiner-Satz
# Position 1 (angelegt): Arme dicht am Körper. Schwerpunkt liegt auf Achse Rumpfradius + R_arm
# Position 2 (ausgestreckt): Arme horizontal, Schwerpunkt bei R_rumpf + L_oarm/2 (Oberarm)
#   bzw. R_rumpf + L_oarm + L_uarm/2 (Unterarm) etc.
def J_arm_angelegt(m_segment, R_segment, abst):
    """Zylinder, Drehachse parallel zur Achse, am Rumpf anliegend."""
    return 0.5 * m_segment * R_segment**2 + m_segment * abst**2

def J_arm_ausgestreckt(m_segment, L_segment, R_segment, abst_zur_drehachse):
    """Zylinder, Drehachse senkrecht zur Längsachse, im Abstand abst."""
    J_eigen = m_segment * (R_segment**2/4 + L_segment**2/12)
    return J_eigen + m_segment * abst_zur_drehachse**2

# Oberarm
m_oarm = ant_oarm_einzeln * M_men
R_oarm = U_oarm / (2 * np.pi)
# Unterarm
m_uarm = ant_uarm_einzeln * M_men
R_uarm = U_uarm / (2 * np.pi)
# Hand (3 % der Länge des Unterarms vereinfacht; Punktmasse am Ende)
m_hand = ant_hand_einzeln * M_men
# Oberschenkel
m_oschenkel = ant_oschenkel_einzeln * M_men
R_oschenkel = U_oschenkel / (2 * np.pi)
# Unterschenkel
m_uschenkel = ant_uschenkel_einzeln * M_men
R_uschenkel = U_uschenkel / (2 * np.pi)
# Fuss (Punktmasse)
m_fuss = ant_fuss_einzeln * M_men

# --- Position 1: Arme angelegt ---
# Arme parallel zur Körperlängsachse, dicht am Rumpf.
abst_oarm_an = R_rumpf + R_oarm
abst_uarm_an = R_rumpf + R_uarm
J_oarm_an = 2 * J_arm_angelegt(m_oarm, R_oarm, abst_oarm_an)
J_uarm_an = 2 * J_arm_angelegt(m_uarm, R_uarm, abst_uarm_an)
J_hand_an = 2 * m_hand * (R_rumpf + R_uarm)**2  # Punktmasse am Rumpf

# Beine ebenfalls parallel zur Körperachse:
abst_oschenkel = 0.5 * R_rumpf  # ungefähr halber Rumpfradius (zwei Beine seitlich)
abst_uschenkel = 0.5 * R_rumpf
J_oschenkel_an = 2 * (0.5 * m_oschenkel * R_oschenkel**2 + m_oschenkel * abst_oschenkel**2)
J_uschenkel_an = 2 * (0.5 * m_uschenkel * R_uschenkel**2 + m_uschenkel * abst_uschenkel**2)
J_fuss_an = 2 * m_fuss * abst_uschenkel**2

J_men_theo_an = (J_kopf + J_rumpf + J_oarm_an + J_uarm_an + J_hand_an
                 + J_oschenkel_an + J_uschenkel_an + J_fuss_an)

# --- Position 2: Arme ausgestreckt ---
# Schwerpunkt Oberarm: r = R_rumpf + L_oarm/2
# Schwerpunkt Unterarm: r = R_rumpf + L_oarm + L_uarm/2
# Hand als Punktmasse: r = R_rumpf + L_oarm + L_uarm
r_sp_oarm = R_rumpf + L_oarm / 2
r_sp_uarm = R_rumpf + L_oarm + L_uarm / 2
r_hand = R_rumpf + L_oarm + L_uarm
J_oarm_ges = 2 * J_arm_ausgestreckt(m_oarm, L_oarm, R_oarm, r_sp_oarm)
J_uarm_ges = 2 * J_arm_ausgestreckt(m_uarm, L_uarm, R_uarm, r_sp_uarm)
J_hand_ges = 2 * m_hand * r_hand**2

J_men_theo_ges = (J_kopf + J_rumpf + J_oarm_ges + J_uarm_ges + J_hand_ges
                  + J_oschenkel_an + J_uschenkel_an + J_fuss_an)

# Unsicherheit (~10 %): grobe Modellannahmen, Abschätzung
u_J_men_theo_an = 0.10 * J_men_theo_an
u_J_men_theo_ges = 0.10 * J_men_theo_ges

log(f"  Modell: Kopf=Kugel, Rumpf+Glieder=Zylinder, Hand+Fuß=Punktmassen")
log(f"  J_theo (Arme angelegt)    = {fmt(J_men_theo_an, u_J_men_theo_an, 'kg·m²')}")
log(f"  J_theo (Arme ausgestreckt) = {fmt(J_men_theo_ges, u_J_men_theo_ges, 'kg·m²')}")
log(f"  Beiträge (angelegt):  Kopf={J_kopf:.4f}, Rumpf={J_rumpf:.4f}, "
    f"Arme={J_oarm_an+J_uarm_an+J_hand_an:.4f}, Beine={J_oschenkel_an+J_uschenkel_an+J_fuss_an:.4f}")
log(f"  Beiträge (ausgestreckt): Arme={J_oarm_ges+J_uarm_ges+J_hand_ges:.4f}, Rest unverändert")

# Auch das einfache Zylindermodell aus Aufg. 2.6.1
U_rumpf_full = 0.83  # Hüftumfang
J_men_zyl_simple = M_men / (8 * np.pi**2) * U_rumpf_full**2
u_J_men_zyl_simple = J_men_zyl_simple * np.sqrt((u_M_men/M_men)**2 + (2*0.005/U_rumpf_full)**2)
log(f"  Zylinder-Näherung (Mensch als Hüft-Zylinder, Gl. 17): "
    f"J = {fmt(J_men_zyl_simple, u_J_men_zyl_simple, 'kg·m²')}")
log("")


# ------------------------------------------------------------------
# Aufgabe 18: Extrapolation Puppe → Mensch
# ------------------------------------------------------------------
log("=" * 72)
log("Aufgabe 18: Extrapolation Puppe → Mensch")
log("=" * 72)

# Puppe: Länge l = 32 cm, Hüftdurchmesser 5,5 cm (gerade) → Hüftumfang 17,3 cm
# Masse Puppe (ohne Stab) m_puppe = 183 g = 0,183 kg
m_puppe = 0.183
u_m_puppe = 0.001
l_puppe = 0.32
u_l_puppe = 0.005
U_puppe = np.pi * 0.055  # Hüftumfang aus Durchmesser 5,5 cm
u_U_puppe = np.pi * 0.001

# Annahme gleiche Form (Gl. 15): J_Mensch = M/m · (L/l)² · J_Puppe
J_extra_a = (M_men / m_puppe) * (H_men / l_puppe)**2 * J_pup_g
u_rel = np.sqrt((u_M_men/M_men)**2 + (u_m_puppe/m_puppe)**2
                + (2*u_H_men/H_men)**2 + (2*u_l_puppe/l_puppe)**2
                + (u_J_pup_g/J_pup_g)**2)
u_J_extra_a = J_extra_a * u_rel

# Annahme gleiche Form und Dichte (Gl. 16): (L/l)^5 · J_puppe (nur wenn ρ gleich)
# Dafür Verhältnis der Dichten:
V_puppe = np.pi * (0.055/2)**2 * l_puppe   # Zylindernäherung Puppe
rho_puppe = m_puppe / V_puppe
V_men_zyl = np.pi * (U_rumpf/(2*np.pi))**2 * H_men
rho_men_zyl = M_men / V_men_zyl
J_extra_b = (rho_men_zyl / rho_puppe) * (H_men / l_puppe)**5 * J_pup_g
u_J_extra_b = 0.20 * J_extra_b   # grobe Schätzung wegen Modellannahme

log(f"  Puppe: m = {m_puppe*1000:.0f} g, l = {l_puppe*100:.0f} cm")
log(f"  Mensch: M = {M_men:.0f} kg, H = {H_men*100:.0f} cm")
log(f"  Extrapolation (Gl. 15, M/m·(L/l)²): "
    f"J = {fmt(J_extra_a, u_J_extra_a, 'kg·m²')}")
log(f"  Extrapolation (Gl. 16, ρ·(L/l)⁵):    "
    f"J = {fmt(J_extra_b, u_J_extra_b, 'kg·m²')}")
log("")


# ------------------------------------------------------------------
# Aufgabe 19: Vergleich
# ------------------------------------------------------------------
log("=" * 72)
log("Aufgabe 19: Vergleich aller berechneten/gemessenen J_Mensch")
log("=" * 72)

log(f"  Messung mit D*_dyn (Arme angelegt):     {fmt(J_men_g, u_J_men_g, 'kg·m²')}")
log(f"  Messung mit D*_stat (Arme angelegt):    {fmt(J_men_g_stat, u_J_men_g_stat, 'kg·m²')}")
log(f"  Modell (Arme angelegt):                  {fmt(J_men_theo_an, u_J_men_theo_an, 'kg·m²')}")
log(f"  Zylinder-Näherung:                       {fmt(J_men_zyl_simple, u_J_men_zyl_simple, 'kg·m²')}")
log(f"  Extrapolation (Gl. 15):                  {fmt(J_extra_a, u_J_extra_a, 'kg·m²')}")
log(f"  Extrapolation (Gl. 16):                  {fmt(J_extra_b, u_J_extra_b, 'kg·m²')}")
log("")
log(f"  Messung mit D*_dyn (Arme ausgestreckt):  {fmt(J_men_a, u_J_men_a, 'kg·m²')}")
log(f"  Messung mit D*_stat (Arme ausgestreckt): {fmt(J_men_a_stat, u_J_men_a_stat, 'kg·m²')}")
log(f"  Modell (Arme ausgestreckt):              {fmt(J_men_theo_ges, u_J_men_theo_ges, 'kg·m²')}")
log("")
log(f"  Verhältnis (Messung) ausgestreckt / angelegt = "
    f"{J_men_a/J_men_g:.2f} ± {J_men_a/J_men_g*np.sqrt((u_J_men_a/J_men_a)**2 + (u_J_men_g/J_men_g)**2):.2f}")


# ------------------------------------------------------------------
# Vorüberlegung: maximale Winkelabweichung theta für 1 % Fehler
# ------------------------------------------------------------------
log("=" * 72)
log("Vorüberlegung: maximale Winkelabweichung θ für < 1 % Drehmomentfehler")
log("=" * 72)
# M = F·r·sin(90°+θ) = F·r·cos(θ) (für Abweichung in 'Hebelarmrichtung')
# 1 - cos(θ) < 0,01 → θ < acos(0,99)
theta_max = np.rad2deg(np.arccos(0.99))
log(f"  Bei Abweichung in Hebelarmrichtung: θ_max = ±{theta_max:.1f}°")
# Bei Abweichung 'aus der Ebene heraus': sin(90°-θ) gleich
log(f"  (Schwächere Richtung; Abweichung in Drehebene zählt linear: < 1° bedingt < 0,02 %)")
log("")
log("Vorüberlegung: maximaler Schwerpunktsabstand für < 10 % Fehler bei Puppe/Mensch")
log("=" * 72)
# Mensch als Zylinder: J_S = m·R²/2. Bei Versatz d: J = J_S + m·d² ≤ 1.10·J_S
# m·d² ≤ 0,10·m·R²/2 → d ≤ R/sqrt(20)
R_men_eff = U_rumpf_full / (2*np.pi)
d_max_men = R_men_eff / np.sqrt(20)
log(f"  Mensch (R = {R_men_eff*100:.1f} cm):  d_max = {d_max_men*100:.1f} cm")
R_pup_eff = 0.055/2
d_max_pup = R_pup_eff / np.sqrt(20)
log(f"  Puppe (R = {R_pup_eff*100:.1f} cm):   d_max = {d_max_pup*100:.1f} cm")
log("")

# ------------------------------------------------------------------
# Endergebnis-Zusammenfassung
# ------------------------------------------------------------------
log("=" * 72)
log("ZUSAMMENFASSUNG (Endergebnisse)")
log("=" * 72)
log(f"  D*_klein (Puppe):            {fmt(D_p*1e3, u_D_p*1e3, 'mNm/rad')}")
log(f"  J0_klein (Eigen):            {fmt(J0_p*1e6, u_J0_p*1e6, '×10⁻⁶ kg·m²')}")
log(f"  J_Puppe (angelegt):          {fmt(J_pup_g*1e6, u_J_pup_g*1e6, '×10⁻⁶ kg·m²')}")
log(f"  J_Puppe (ausgestreckt):      {fmt(J_pup_a*1e6, u_J_pup_a*1e6, '×10⁻⁶ kg·m²')}")
log(f"  D*_groß (Drehteller):        {fmt(D_d, u_D_d, 'Nm/rad')}")
log(f"  J0_groß (Drehteller):        {fmt(J0_d, u_J0_d, 'kg·m²')}")
log(f"  J_Mensch (angelegt):         {fmt(J_men_g, u_J_men_g, 'kg·m²')}")
log(f"  J_Mensch (ausgestreckt):     {fmt(J_men_a, u_J_men_a, 'kg·m²')}")
log("")

# Save numerical values to be reused by plots and LaTeX
import json
results = {
    "D_p_stat": D_p_stat, "u_D_p_stat": u_D_p_stat,
    "D_p_dyn": D_p_dyn, "u_D_p_dyn": u_D_p_dyn,
    "J0_p": J0_p, "u_J0_p": u_J0_p,
    "J_pup_g": J_pup_g, "u_J_pup_g": u_J_pup_g,
    "J_pup_a": J_pup_a, "u_J_pup_a": u_J_pup_a,
    "D_d_stat": D_d_stat, "u_D_d_stat": u_D_d_stat,
    "D_d_dyn": D_d_dyn, "u_D_d_dyn": u_D_d_dyn,
    "J0_d": J0_d, "u_J0_d": u_J0_d,
    "J_men_g": J_men_g, "u_J_men_g": u_J_men_g,
    "J_men_a": J_men_a, "u_J_men_a": u_J_men_a,
    "J_men_g_stat": J_men_g_stat, "u_J_men_g_stat": u_J_men_g_stat,
    "J_men_a_stat": J_men_a_stat, "u_J_men_a_stat": u_J_men_a_stat,
    "J_men_theo_an": J_men_theo_an, "u_J_men_theo_an": u_J_men_theo_an,
    "J_men_theo_ges": J_men_theo_ges, "u_J_men_theo_ges": u_J_men_theo_ges,
    "J_men_zyl_simple": J_men_zyl_simple, "u_J_men_zyl_simple": u_J_men_zyl_simple,
    "J_extra_a": J_extra_a, "u_J_extra_a": u_J_extra_a,
    "J_extra_b": J_extra_b, "u_J_extra_b": u_J_extra_b,
    "T_d": T_d, "u_T_d": u_T_d,
    "T_men_g": T_men_g, "u_T_men_g": u_T_men_g,
    "T_men_a": T_men_a, "u_T_men_a": u_T_men_a,
    "T0_klein": T0, "u_T0_klein": u_T0,
    "T_pup_g": T_pup_g, "T_pup_a": T_pup_a,
    "u_T_pup_g": u_T_pup_g, "u_T_pup_a": u_T_pup_a,
    "fit_p_a0": fit_p["a0"], "fit_p_a1": fit_p["a1"],
    "fit_p_u_a0": fit_p["u_a0"], "fit_p_u_a1": fit_p["u_a1"],
    "fit_d_a0": fit_d["a0"], "fit_d_a1": fit_d["a1"],
    "fit_d_u_a0": fit_d["u_a0"], "fit_d_u_a1": fit_d["u_a1"],
    "lin_a0_d": a0_d, "lin_a1_d": a1_d,
    "lin_u_a0_d": u_a0_d, "lin_u_a1_d": u_a1_d,
    "Jz": Jz.tolist(), "u_Jz": u_Jz.tolist(),
    "T_sq_mean": T_sq.tolist(), "u_T_sq": u_T_sq.tolist(),
    "abst_cm": abst_cm.tolist(),
    "T_90": T_90.tolist(), "T_135": T_135.tolist(),
    "phi_deg_p": phi_deg_p.tolist(), "F_p": F_p.tolist(), "M_p": M_p.tolist(),
    "phi_deg_d": phi_deg_d.tolist(), "F_d": F_d.tolist(), "M_d": M_d.tolist(),
    "r_p": r_p, "r_d": r_d,
    "theta_max_deg": theta_max,
    "d_max_men_cm": d_max_men*100, "d_max_pup_cm": d_max_pup*100,
}
with open(OUT / "ergebnisse.json", "w") as f:
    json.dump(results, f, indent=2)

with open(RESULTS, "w") as f:
    f.write("\n".join(LOG_LINES))
print(f"Ergebnisse geschrieben: {RESULTS}")
print(f"JSON-Daten geschrieben: {OUT/'ergebnisse.json'}")

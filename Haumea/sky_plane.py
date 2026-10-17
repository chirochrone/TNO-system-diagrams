"""Earth-view sky-plane plot of Haumea, its ring, Hi'iaka, and Namaka.

Adopted sources:
- Proudfoot et al. (2026b), 
  arXiv: https://arxiv.org/pdf/2609.36037: Haumea pole, mass, and moons
- Proudfoot et al. (2026a), Planetary Science Journal 7, 156,
  DOI: 10.3847/PSJ/ae77fa: Haumea triaxial dimensions
- Proudfoot et al. (2024), Planetary Science Journal 5, 69,
  DOI: 10.3847/PSJ/ad26e9: Haumea mass/pole and HST-only satellite
  orbital fit (osculating/mean fitted elements at JD 2454615.0).
- Ortiz et al. (2017), Nature 550, 219-223, DOI: 10.1038/nature24051:
  ring radius/width and ring pole.
- Fernández-Valenzuela et al. (2025), Nature Communications 16, 10926,
  DOI: 10.1038/s41467-025-65749-1: Hi'iaka triaxial shape.

The satellite paths are propagated with a two-body Keplerian approximation
from the published epoch elements. Proudfoot et al.'s fitted solutions are
non-Keplerian and include J2 and mutual perturbations, so this is a useful
visualization rather than a precision ephemeris, particularly for Namaka.
Namaka's shape and both moons' rotation phases are not measured well enough
for a unique 3-D shape/orientation; see assumptions below.
"""

import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from astropy.time import Time

# ============================================================
# User settings
# ============================================================
JD = 2461200.5
DARK_MODE = False
SCALE_BAR_KM = 10000.0
N_ORBIT = 1600
N_LON = 12          # sparse meridian curves on each ellipsoid
N_LAT = 6           # sparse parallel curves on each ellipsoid
N_CURVE = 240       # samples per curve; controls smoothness only
BODY_OUTLINE_LW = 1.7

fs_annotation = 14
fs_axis = 16
fs_title = 18

# ============================================================
# Constants and adopted physical parameters
# ============================================================
AU_KM = 149597870.7
ARCSEC_PER_RAD = 206264.80624709636
G_KM = 6.67430e-20  # km^3 kg^-1 s^-2
OBLIQUITY_DEG = 23.439291111  # J2000 mean obliquity

# Haumea: semiaxes from Proudfoot et al. (2026a).
HAUMEA_AXES = np.array([1061.0, 844.0, 514.0])
HAUMEA_MASS_KG = 3.904e21  # Proudfoot et al. (2026b)
HAUMEA_POLE_RA_DEG = 284.8
HAUMEA_POLE_DEC_DEG = -9.7

# Ring: Ortiz et al. (2017)
RING_RADIUS_KM = 2287.0
RING_WIDTH_KM = 70.0
RING_POLE_RA_DEG = 284.8
RING_POLE_DEC_DEG = -9.7

# Hi'iaka: Fernández-Valenzuela et al. (2025), semiaxes in km.
HIIAKA_AXES = np.array([240.0, 180.0, 143.0])
HIIAKA_MASS_KG = 16.5e18  # Proudfoot et al. (2026b)
# 2025 occultation analysis suggests the spin axis is aligned with Haumea's
# within uncertainties; use the Haumea pole as the nominal spin pole.
HIIAKA_POLE_RA_DEG = HAUMEA_POLE_RA_DEG
HIIAKA_POLE_DEC_DEG = HAUMEA_POLE_DEC_DEG

# Namaka has no published triaxial shape or spin pole comparable to Hi'iaka's
# 2025 occultation result. Its ellipsoid dimensions and spin pole below are
# explicitly illustrative, not measured values.
NAMAKA_AXES = np.array([85.0, 65.0, 50.0])
NAMAKA_MASS_KG = 1.35e18  # Proudfoot et al. (2026b)
NAMAKA_POLE_RA_DEG = HAUMEA_POLE_RA_DEG
NAMAKA_POLE_DEC_DEG = HAUMEA_POLE_DEC_DEG

# Proudfoot et al. (2026b), Table 1.
# Elements are relative to the J2000 ecliptic, at Haumea-centric JD 2454615.0.
ELEMENT_EPOCH_JD = 2454615.0
ORBITAL_ELEMENTS = {
    "Hi'iaka": {
        "a_km": 49186.0, "e": 0.0604, "i_deg": 77.24,
        "Omega_deg": 13.09, "omega_deg": 108.4, "M0_deg": 144.5,
        "mass_kg": HIIAKA_MASS_KG,
    },
    "Namaka": {
        "a_km": 25365.0, "e": 0.220, "i_deg": 68.9,
        "Omega_deg": 23.90, "omega_deg": 118.3, "M0_deg": 184.7,
        "mass_kg": NAMAKA_MASS_KG,
    },
}

# Rendering colors
if DARK_MODE:
    FIG_BG = "black"
    AX_BG = "black"
    FG = "white"
    BODY_COLOR = "white"
    RING_COLOR = "#b9c5d6"
    HIIAKA_COLOR = "#48c9b0"
    NAMAKA_COLOR = "#ffb454"
    GRID_COLOR = "#666666"
else:
    FIG_BG = "white"
    AX_BG = "white"
    FG = "black"
    BODY_COLOR = "black"
    RING_COLOR = "#7c8798"
    HIIAKA_COLOR = "#168a78"
    NAMAKA_COLOR = "#c66b00"
    GRID_COLOR = "#d0d0d0"

# ============================================================
# Coordinate and orbit utilities
# ============================================================
def unit(vec):
    vec = np.asarray(vec, dtype=float)
    return vec / np.linalg.norm(vec)


def pole_vector(ra_deg, dec_deg):
    ra, dec = np.deg2rad([ra_deg, dec_deg])
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


def plane_basis(ra_deg, dec_deg):
    pole = pole_vector(ra_deg, dec_deg)
    ref = np.array([0.0, 0.0, 1.0])
    u = ref - np.dot(ref, pole)*pole
    if np.linalg.norm(u) < 1e-10:
        ref = np.array([1.0, 0.0, 0.0])
        u = ref - np.dot(ref, pole)*pole
    u = unit(u)
    v = unit(np.cross(pole, u))
    return u, v, pole


def ecliptic_to_equatorial(xyz):
    eps = np.deg2rad(OBLIQUITY_DEG)
    x, y, z = np.moveaxis(np.asarray(xyz), -1, 0)
    return np.stack([x, y*np.cos(eps)-z*np.sin(eps),
                     y*np.sin(eps)+z*np.cos(eps)], axis=-1)


def solve_kepler(M, e, tol=1e-12, max_iter=80):
    M = np.asarray(M, dtype=float)
    E = M.copy()
    # A good initial guess for modest eccentricities.
    E = np.where(e < 0.8, M, np.pi*np.ones_like(M))
    for _ in range(max_iter):
        step = (E - e*np.sin(E) - M) / (1.0 - e*np.cos(E))
        E -= step
        if np.max(np.abs(step)) < tol:
            break
    return E


def orbit_position(elements, jd, samples=None):
    """Return the moon's Haumea-relative position in equatorial km."""
    a = elements["a_km"]
    e = elements["e"]
    inc, node, argp = np.deg2rad([
        elements["i_deg"], elements["Omega_deg"], elements["omega_deg"]
    ])
    # Use Kepler's third law with the adopted fitted masses.
    mu = G_KM * (HAUMEA_MASS_KG + elements["mass_kg"])
    n = np.sqrt(mu / a**3)  # rad/s
    dt_seconds = (jd - ELEMENT_EPOCH_JD) * 86400.0

    if samples is None:
        M = np.deg2rad(elements["M0_deg"]) + n*dt_seconds
    else:
        M = np.linspace(0.0, 2.0*np.pi, samples)
    M = np.mod(M, 2.0*np.pi)
    E = solve_kepler(np.asarray(M), e)
    x_pf = a * (np.cos(E) - e)
    y_pf = a * np.sqrt(1.0 - e**2) * np.sin(E)

    cO, sO = np.cos(node), np.sin(node)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    rot = np.array([
        [cO*cw-sO*sw*ci, -cO*sw-sO*cw*ci, sO*si],
        [sO*cw+cO*sw*ci, -sO*sw+cO*cw*ci, -cO*si],
        [sw*si, cw*si, ci],
    ])
    xyz_ecl = np.stack([x_pf, y_pf, np.zeros_like(x_pf)], axis=-1) @ rot.T
    return ecliptic_to_equatorial(xyz_ecl)


def ellipsoid_wireframe(axes, pole_ra, pole_dec, center=None):
    """Create sparse, smooth 3-D meridians and parallels for an ellipsoid."""
    if center is None:
        center = np.zeros(3)
    u, v, pole = plane_basis(pole_ra, pole_dec)
    # Long-axis orientation in the equatorial plane is arbitrary; use a
    # fixed zero phase for reproducibility rather than implying a measured phase.
    lon = np.linspace(0.0, 2*np.pi, N_LON, endpoint=False)
    lat = np.linspace(-np.pi/2 + 0.12, np.pi/2 - 0.12, N_LAT)
    t = np.linspace(0.0, 2*np.pi, N_CURVE)
    meridians, parallels = [], []
    for lam in lon:
        pts = (axes[0]*np.outer(np.cos(t)*np.cos(lam), u)
               + axes[1]*np.outer(np.cos(t)*np.sin(lam), v)
               + axes[2]*np.outer(np.sin(t), pole))
        meridians.append(pts + center)
    for phi in lat:
        pts = (axes[0]*np.outer(np.cos(t)*np.cos(phi), u)
               + axes[1]*np.outer(np.sin(t)*np.cos(phi), v)
               + axes[2]*np.outer(np.ones_like(t)*np.sin(phi), pole))
        parallels.append(pts + center)
    return meridians, parallels, u, v, pole

# ============================================================
# Get Earth-view geometry from JPL Horizons
# ============================================================
# Target 136108 is Haumea. Horizons provides geocentric RA/Dec and distance.
eph = Horizons(id="136108", id_type="smallbody", location="500@399",
               epochs=JD).ephemerides()
ra_q = np.deg2rad(float(eph["RA"][0]))
dec_q = np.deg2rad(float(eph["DEC"][0]))
delta_au = float(eph["delta"][0])
delta_km = delta_au * AU_KM

east = np.array([-np.sin(ra_q), np.cos(ra_q), 0.0])
north = np.array([-np.sin(dec_q)*np.cos(ra_q),
                   -np.sin(dec_q)*np.sin(ra_q), np.cos(dec_q)])
observer_to_target = np.array([np.cos(dec_q)*np.cos(ra_q),
                               np.cos(dec_q)*np.sin(ra_q), np.sin(dec_q)])
# Direction from Haumea toward Earth; surface points facing this vector are visible.
toward_earth = -observer_to_target


def project_sky(xyz_km):
    xyz_km = np.asarray(xyz_km)
    east_arcsec = (xyz_km @ east) / delta_km * ARCSEC_PER_RAD
    north_arcsec = (xyz_km @ north) / delta_km * ARCSEC_PER_RAD
    return -east_arcsec, north_arcsec  # East to the left


def surface_visible(curve, axes, u, v, pole, center):
    rel = curve - center
    x, y, z = rel @ u, rel @ v, rel @ pole
    normals = ((x/axes[0]**2)[:, None]*u
               + (y/axes[1]**2)[:, None]*v
               + (z/axes[2]**2)[:, None]*pole)
    return normals @ toward_earth > 0


def plot_wireframe(ax, curves, axes, u, v, pole, center, color, lw, alpha, zorder):
    for curve in curves:
        vis = surface_visible(curve, axes, u, v, pole, center)
        idx = np.flatnonzero(vis)
        if len(idx) < 2:
            continue
        # Break curves at hidden-side gaps to avoid drawing across the body.
        segments = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for seg in segments:
            if len(seg) >= 2:
                x, y = project_sky(curve[seg])
                ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=zorder)


def plot_projected_ellipsoid_outline(ax, axes, u, v, pole, center, color, lw, zorder):
    """Exact projected ellipse from the ellipsoid's covariance matrix."""
    shape = (axes[0]**2*np.outer(u, u)
             + axes[1]**2*np.outer(v, v)
             + axes[2]**2*np.outer(pole, pole))
    sky_basis = np.vstack([east, north])
    cov2 = sky_basis @ shape @ sky_basis.T
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.maximum(vals, 0.0)
    t = np.linspace(0, 2*np.pi, 700)
    ellipse = vecs @ np.diag(np.sqrt(vals)) @ np.vstack([np.cos(t), np.sin(t)])
    center_e = np.dot(center, east)
    center_n = np.dot(center, north)
    east_km = center_e + ellipse[0]
    north_km = center_n + ellipse[1]
    x = -east_km / delta_km * ARCSEC_PER_RAD
    y = north_km / delta_km * ARCSEC_PER_RAD
    ax.plot(x, y, color=color, lw=lw, zorder=zorder)

# Moon positions and full orbital curves at the requested JD.
moon_states = {}
moon_orbits = {}
for name, el in ORBITAL_ELEMENTS.items():
    moon_states[name] = orbit_position(el, JD)
    moon_orbits[name] = orbit_position(el, JD, samples=N_ORBIT)

# Body axes oriented using the published Haumea pole; moon long-axis phases
# are arbitrary, but their pole assumptions are documented above.
haumea_center = np.zeros(3)
h_curves = ellipsoid_wireframe(HAUMEA_AXES, HAUMEA_POLE_RA_DEG,
                               HAUMEA_POLE_DEC_DEG, haumea_center)
hi_curves = ellipsoid_wireframe(HIIAKA_AXES, HIIAKA_POLE_RA_DEG,
                                HIIAKA_POLE_DEC_DEG, moon_states["Hi'iaka"])
na_curves = ellipsoid_wireframe(NAMAKA_AXES, NAMAKA_POLE_RA_DEG,
                                NAMAKA_POLE_DEC_DEG, moon_states["Namaka"])

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(11, 10), facecolor=FIG_BG)
ax.set_facecolor(AX_BG)

# Ring plane: construct circles using the published ring pole, then transform
# them from Haumea-centric equatorial coordinates into sky-plane offsets.
ring_u, ring_v, ring_pole = plane_basis(RING_POLE_RA_DEG, RING_POLE_DEC_DEG)
t = np.linspace(0.0, 2*np.pi, N_ORBIT)
def ring_circle(radius):
    return radius*np.cos(t)[:, None]*ring_u + radius*np.sin(t)[:, None]*ring_v

for radius, style, width, alpha, label in [
    (RING_RADIUS_KM, "-", 1.15, 1.0, "Ring centerline"),
    (RING_RADIUS_KM - RING_WIDTH_KM/2, "--", 0.8, 0.8, "Ring edges"),
    (RING_RADIUS_KM + RING_WIDTH_KM/2, "--", 0.8, 0.8, None),
]:
    x, y = project_sky(ring_circle(radius))
    ax.plot(x, y, color=RING_COLOR, ls=style, lw=width, alpha=alpha,
            label=label, zorder=2)

# Satellite orbital paths. These are Keplerian approximations from the epoch fit.
for name, color in [("Hi'iaka", HIIAKA_COLOR), ("Namaka", NAMAKA_COLOR)]:
    x, y = project_sky(moon_orbits[name])
    ax.plot(x, y, color=color, ls="--", lw=0.9, alpha=0.85,
            label=f"{name} orbit (Keplerian approximation)", zorder=3)

# Body wireframes and analytically smooth projected limb outlines.
plot_wireframe(ax, h_curves[0] + h_curves[1], HAUMEA_AXES,
               h_curves[2], h_curves[3], h_curves[4], haumea_center,
               BODY_COLOR, 0.75, 0.95, 10)
plot_projected_ellipsoid_outline(ax, HAUMEA_AXES, h_curves[2], h_curves[3],
                                 h_curves[4], haumea_center, BODY_COLOR,
                                 BODY_OUTLINE_LW, 12)

for name, axes, curves, color, center in [
    ("Hi'iaka", HIIAKA_AXES, hi_curves, HIIAKA_COLOR, moon_states["Hi'iaka"]),
    ("Namaka", NAMAKA_AXES, na_curves, NAMAKA_COLOR, moon_states["Namaka"]),
]:
    plot_wireframe(ax, curves[0] + curves[1], axes,
                   curves[2], curves[3], curves[4], center,
                   "black", 0.6, 0.95, 14)
    plot_projected_ellipsoid_outline(ax, axes, curves[2], curves[3],
                                     curves[4], center, "black",
                                     BODY_OUTLINE_LW, 15)
    x, y = project_sky(center)
    ax.annotate(name, (float(x), float(y)), xytext=(7, 6),
                textcoords="offset points", color=color,
                fontsize=fs_annotation, ha="left", zorder=20)

# Axes and styling
ax.set_xlabel("East ←  Offset (arcsec)  → West", fontsize=fs_axis, color=FG)
ax.set_ylabel("South ←  Offset (arcsec)  → North", fontsize=fs_axis, color=FG)

t = Time(JD, format='jd', scale='utc')
ax.set_title(f"Haumea system as seen from Earth\n{t.datetime} UTC",
             fontsize=fs_title, color=FG)
ax.set_aspect("equal", adjustable="datalim")
ax.tick_params(axis="both", colors=FG, labelsize=11)
for spine in ax.spines.values():
    spine.set_color(FG)
ax.grid(True, alpha=0.22, color=GRID_COLOR)

# Secondary axes: projected sky-plane distance at Haumea's geocentric distance.
km_per_arcsec = delta_km / ARCSEC_PER_RAD
def arcsec_to_km(x): return np.asarray(x) * km_per_arcsec
def km_to_arcsec(x): return np.asarray(x) / km_per_arcsec
secx = ax.secondary_xaxis("top", functions=(arcsec_to_km, km_to_arcsec))
secy = ax.secondary_yaxis("right", functions=(arcsec_to_km, km_to_arcsec))
secx.set_xlabel("Projected distance at Haumea (km)", fontsize=fs_axis, color=FG)
secy.set_ylabel("Projected distance at Haumea (km)", fontsize=fs_axis, color=FG)
secx.tick_params(colors=FG, labelsize=10)
secy.tick_params(colors=FG, labelsize=10)
secx.spines["top"].set_color(FG)
secy.spines["right"].set_color(FG)

# Scale bar in the lower-left of the sky plot.
fig.canvas.draw()
xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
bar_arcsec = SCALE_BAR_KM / km_per_arcsec
x0 = xmin + 0.06*(xmax-xmin)
y0 = ymin + 0.07*(ymax-ymin)
ax.plot([x0, x0+bar_arcsec], [y0, y0], color=FG, lw=2.2,
        solid_capstyle="butt", zorder=30)
ax.text(x0+bar_arcsec/2, y0+0.018*(ymax-ymin), f"{SCALE_BAR_KM:g} km",
        ha="center", va="bottom", color=FG, fontsize=fs_axis, zorder=30)

'''
# Legend has been kept compact; comment this out if preferred without a legend.
leg = ax.legend(loc="best", fontsize=8, framealpha=0.85)
leg.get_frame().set_facecolor(AX_BG)
leg.get_frame().set_edgecolor(FG)
for text in leg.get_texts():
    text.set_color(FG)
'''

plt.tight_layout()
output_name = "Haumea_system_sky_view.svg"
plt.savefig(output_name, dpi=200, facecolor=FIG_BG, bbox_inches="tight")
plt.show()

print(f"Observation JD: {JD:.1f}")
print(f"Haumea geocentric distance: {delta_au:.6f} AU")
print(f"Haumea RA, Dec: {np.rad2deg(ra_q):.6f}, {np.rad2deg(dec_q):.6f} deg")
print(f"Haumea mass adopted: {HAUMEA_MASS_KG:.5e} kg")
print(f"Haumea pole adopted from Proudfoot et al. (2024): RA={HAUMEA_POLE_RA_DEG} deg, Dec={HAUMEA_POLE_DEC_DEG} deg")
print(f"Ring: radius={RING_RADIUS_KM:g} km, width={RING_WIDTH_KM:g} km; pole from Ortiz et al. (2017)")
for name, el in ORBITAL_ELEMENTS.items():
    pos = moon_states[name]
    x, y = project_sky(pos)
    print(f"{name}: M0={el['M0_deg']:.2f} deg at JD {ELEMENT_EPOCH_JD:.1f}; "
          f"projected offset (East-left, North-up)=({float(x):.4f}, {float(y):.4f}) arcsec")
print("Caution: satellite positions are Keplerian approximations; Proudfoot et al.'s fitted solution includes non-Keplerian dynamics.")
print("Namaka's ellipsoid dimensions and the moon spin phases are illustrative assumptions, not measured values.")

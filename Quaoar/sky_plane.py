import numpy as np
import matplotlib.pyplot as plt
import requests

from astroquery.jplhorizons import Horizons
from astropy.time import Time


# ============================================================
# Global plot settings
# ============================================================

# Fonts
fs_annotation = 12
fs_legend = 12
fs_axis = 16
fs_title = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

# Colors
color_ellipsoid = 'black'
color_moon = 'red'
color_ring = 'gray'

# Opacities
alpha_moon = 1.
alpha_Q1R = 0.65
alpha_Q2R = 0.65

# Linewidths
lw_moon = 1.
lw_Q1R = 0.7
lw_Q2R = 0.6


DARK_MODE = input("Dark mode? (y/n): ")  # False = light background

if (DARK_MODE == 'y') or (DARK_MODE == 'Y'):
    print("Dark mode enabled.")
    plt.style.use('dark_background')
    color_moon = 'lightcoral'
    color_ellipsoid = 'white'
    color_ring = 'lightgray'

else:
    print("No dark mode, moving on...")

# ============================================================
# Observation and system parameters
# ============================================================

JD = 2461200.5
AU_KM = 149597870.7
ARCSEC_PER_RAD = 206264.806247

# Quaoar's preferred occultation-derived dimensions
# Full diameters in km (Proudfoot et al. 2025)
QUAOAR_AXES = np.array([1166.6, 1110.6, 1020.0]) / 2.0

# Quaoar pole: ring pole used as a proxy for the spin pole
QUAOAR_POLE_RA = 259.5
QUAOAR_POLE_DEC = 55.0

# Weywot principal semiaxes (Collyer et al. 2026), km
WEYWOT_AXES = np.array([82.0, 73.0, 65.0])

# Weywot spin pole aligned with its orbit pole, as requested
WEYWOT_POLE_RA = 267.38
WEYWOT_POLE_DEC = 53.00

WEYWOT_ORBIT_RADIUS = 13329.0  # km, approximate orbital semimajor axis
WEYWOT_PERIOD_DAYS = 12.42727

# Weywot orbital elements at reference epoch JD 2454000.0
WEYWOT_EPOCH_JD = 2454000.0
WEYWOT_OMEGA_DEG = 97.0
WEYWOT_NODE_DEG = 353.3
WEYWOT_M0_DEG = 79.0

# Ring pole from Proudfoot et al. (2025), equatorial coordinates
RING_POLE_RA = 259.5
RING_POLE_DEC = 55.0

# Q2R: inner ring
Q2R_RADIUS = 2520.0
Q2R_WIDTH = 10.0

# Q1R: outer ring
Q1R_RADIUS = 4057.2
Q1R_DENSE_FWHM = 5.0

# Q1R is variable in width; show a representative broad component.
# Change this to 100 km, or another value, to explore the range.
Q1R_BROAD_WIDTH = 80.0

# Candidate second satellite, reported in 2025
CANDIDATE_RADIUS = 5676.0
CANDIDATE_RADIUS_UNCERTAINTY = 108.0
CANDIDATE_PERIOD_DAYS = 3.6

# Orbital phase of candidate is unknown.
CANDIDATE_PHASE_DEG = 180.0

# Rotational phases of the ellipsoid long axes are also unknown.
QUAOAR_ROTATION_PHASE_DEG = 0.0
WEYWOT_ROTATION_PHASE_DEG = 0.0

# Rendering resolution
N_ORBIT = 1000
N_LON = 12       # number of curved meridian lines
N_LAT = 6        # number of curved parallel lines
N_CURVE = 200    # points along each line (smoothness, not line density)

# Scale bar shown on the sky-plane plot
SCALE_BAR_KM = 5000.0


# ============================================================
# Vector and coordinate utilities
# ============================================================

def unit(v):
    return v / np.linalg.norm(v)


def pole_vector(ra_deg, dec_deg):
    """Equatorial Cartesian unit vector toward a north pole."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def plane_basis(ra_deg, dec_deg):
    """
    Construct two orthogonal directions in a plane whose
    normal has the specified equatorial RA and Dec.
    """
    pole = pole_vector(ra_deg, dec_deg)

    # Reference direction: equatorial north projected into plane
    reference = np.array([0.0, 0.0, 1.0])
    u = reference - np.dot(reference, pole) * pole

    if np.linalg.norm(u) < 1e-10:
        reference = np.array([1.0, 0.0, 0.0])
        u = reference - np.dot(reference, pole) * pole

    u = unit(u)
    v = unit(np.cross(pole, u))

    return u, v, pole


def rotate_in_plane(u, v, angle_deg):
    """Rotate the reference axes within the same plane."""
    angle = np.deg2rad(angle_deg)

    u_rot = np.cos(angle) * u + np.sin(angle) * v
    v_rot = -np.sin(angle) * u + np.cos(angle) * v

    return u_rot, v_rot


def circle_3d(radius_km, u, v, phase_deg=0.0, n=N_ORBIT):
    """Coordinates of a circular orbit in equatorial Cartesian space."""
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    phase = np.deg2rad(phase_deg)

    xyz = radius_km * (
        np.cos(theta + phase)[:, None] * u
        + np.sin(theta + phase)[:, None] * v
    )

    return xyz


# ============================================================
# Calculate Weywot's orbital phase and position
# ============================================================

# Initialize Weywot's orbital plane
weywot_u, weywot_v, weywot_pole = plane_basis(
    WEYWOT_POLE_RA,
    WEYWOT_POLE_DEC
)

# Propagate mean anomaly to the observation epoch
WEYWOT_MEAN_ANOMALY_DEG = (
    WEYWOT_M0_DEG
    + 360.0 * (JD - WEYWOT_EPOCH_JD) / WEYWOT_PERIOD_DAYS
) % 360.0

# Convert the ascending-node direction from ecliptic
# to equatorial Cartesian coordinates.
eps = np.deg2rad(23.4392911)
Omega = np.deg2rad(WEYWOT_NODE_DEG)

node_equatorial = np.array([
    np.cos(Omega),
    np.sin(Omega) * np.cos(eps),
    np.sin(Omega) * np.sin(eps),
])

# Project the node direction into the existing orbital plane
node_equatorial -= (
    np.dot(node_equatorial, weywot_pole) * weywot_pole
)

weywot_u = unit(node_equatorial)
weywot_v = unit(np.cross(weywot_pole, weywot_u))

# Circular-orbit phase approximation
WEYWOT_PHASE_DEG = (
    WEYWOT_OMEGA_DEG + WEYWOT_MEAN_ANOMALY_DEG
) % 360.0

print(
    f"Weywot mean anomaly at JD {JD}: "
    f"{WEYWOT_MEAN_ANOMALY_DEG:.4f} deg"
)
print(f"Weywot orbital phase: {WEYWOT_PHASE_DEG:.4f} deg")


# ============================================================
# Retrieve Quaoar's geocentric viewing geometry from Horizons
# ============================================================

eph = Horizons(
    id="50000",
    location="500@399",
    epochs=JD,
).ephemerides()

ra_q = np.deg2rad(float(eph["RA"][0]))
dec_q = np.deg2rad(float(eph["DEC"][0]))
delta_au = float(eph["delta"][0])
delta_km = delta_au * AU_KM

# Local sky directions in equatorial Cartesian coordinates
east = np.array([
    -np.sin(ra_q),
    np.cos(ra_q),
    0.0,
])

north = np.array([
    -np.sin(dec_q) * np.cos(ra_q),
    -np.sin(dec_q) * np.sin(ra_q),
    np.cos(dec_q),
])


def project_sky(xyz_km):
    """
    Project Quaoar-centric equatorial Cartesian positions
    into sky offsets in arcseconds.

    Return x = East-positive offset, y = North-positive offset.
    The plotting function will reverse x to put East on the left.
    """
    east_offset = xyz_km @ east / delta_km * ARCSEC_PER_RAD
    north_offset = xyz_km @ north / delta_km * ARCSEC_PER_RAD

    return east_offset, north_offset


def visible_ellipsoid_points(points, axes, u, v, pole, center=None):
    """
    Identify Earth-facing points on a triaxial ellipsoid.

    points: (N, 3) Cartesian surface coordinates in km
    axes:   semiaxes (a, b, c) in km
    u, v:   equatorial body-axis unit vectors
    pole:   polar-axis unit vector
    center: ellipsoid center in the same coordinate frame
    """
    if center is None:
        center = np.zeros(3)

    # IMPORTANT: convert absolute positions to body-centered
    # coordinates before calculating the surface normals.
    relative = points - center

    x = relative @ u
    y = relative @ v
    z = relative @ pole

    # Surface normal: gradient of the triaxial ellipsoid.
    normals = (
        (x / axes[0]**2)[:, None] * u
        + (y / axes[1]**2)[:, None] * v
        + (z / axes[2]**2)[:, None] * pole
    )

    # Direction from Quaoar toward Earth.
    # RA/Dec specify the opposite direction: Earth toward Quaoar.
    observer = -np.array([
        np.cos(dec_q) * np.cos(ra_q),
        np.cos(dec_q) * np.sin(ra_q),
        np.sin(dec_q),
    ])

    # Visible when the outward surface normal faces the observer.
    return (normals @ observer) > 0


# ============================================================
# Construct ring planes and satellite orbits
# ============================================================

ring_u, ring_v, ring_pole = plane_basis(
    RING_POLE_RA, RING_POLE_DEC
)

weywot_u, weywot_v, weywot_pole = plane_basis(
    WEYWOT_POLE_RA, WEYWOT_POLE_DEC
)

# Weywot's long-axis orientation is arbitrary within its equatorial plane.
weywot_u, weywot_v = rotate_in_plane(
    weywot_u, weywot_v, WEYWOT_ROTATION_PHASE_DEG
)

# Quaoar's body axes use the ring pole as its spin-axis proxy.
quaoar_u, quaoar_v, quaoar_pole = plane_basis(
    QUAOAR_POLE_RA, QUAOAR_POLE_DEC
)
quaoar_u, quaoar_v = rotate_in_plane(
    quaoar_u, quaoar_v, QUAOAR_ROTATION_PHASE_DEG
)

# Circular ring centerlines
q2r_center = circle_3d(Q2R_RADIUS, ring_u, ring_v)
q1r_center = circle_3d(Q1R_RADIUS, ring_u, ring_v)

# Inner/outer edges of the Q2R ring
q2r_inner = circle_3d(
    Q2R_RADIUS - Q2R_WIDTH / 2, ring_u, ring_v
)
q2r_outer = circle_3d(
    Q2R_RADIUS + Q2R_WIDTH / 2, ring_u, ring_v
)

# Q1R broad component and dense component.
# The broad component is drawn at a representative 80 km width.
q1r_inner = circle_3d(
    Q1R_RADIUS - Q1R_BROAD_WIDTH / 2, ring_u, ring_v
)
q1r_outer = circle_3d(
    Q1R_RADIUS + Q1R_BROAD_WIDTH / 2, ring_u, ring_v
)

q1r_dense_inner = circle_3d(
    Q1R_RADIUS - Q1R_DENSE_FWHM / 2, ring_u, ring_v
)
q1r_dense_outer = circle_3d(
    Q1R_RADIUS + Q1R_DENSE_FWHM / 2, ring_u, ring_v
)

# Weywot's orbital path and illustrative position
weywot_orbit = circle_3d(
    WEYWOT_ORBIT_RADIUS, weywot_u, weywot_v
)
weywot_position = circle_3d(
    WEYWOT_ORBIT_RADIUS,
    weywot_u,
    weywot_v,
    phase_deg=WEYWOT_PHASE_DEG,
    n=1,
)

# Candidate satellite orbit: provisional plane assumed coplanar with rings.
candidate_orbit = circle_3d(
    CANDIDATE_RADIUS, ring_u, ring_v
)
candidate_position = circle_3d(
    CANDIDATE_RADIUS,
    ring_u,
    ring_v,
    phase_deg=CANDIDATE_PHASE_DEG,
    n=1,
)


# ============================================================
# Wireframe ellipsoid generation
# ============================================================

def ellipsoid_wireframe(axes, u, v, pole, center=None,
                        n_lon=N_LON, n_lat=N_LAT, n_curve=N_CURVE):
    """Make a sparse, smoothly sampled wireframe triaxial ellipsoid.

    n_lon and n_lat control how many meridians and parallels are drawn;
    n_curve controls how smoothly each individual curved line is sampled.
    axes are semiaxes (a, b, c), in km. The c axis follows the specified
    pole, while a and b lie in its equatorial plane.
    """
    if center is None:
        center = np.zeros(3)

    # Sparse coordinates determine how many lines are drawn.
    lon_lines = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    lat_lines = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)[1:-1]

    # Dense coordinates determine how smooth each line appears.
    lon_curve = np.linspace(0, 2 * np.pi, n_curve)
    lat_curve = np.linspace(-np.pi / 2, np.pi / 2, n_curve)

    meridians = []
    for lam in lon_lines:
        points = (
            axes[0] * np.outer(np.cos(lat_curve) * np.cos(lam), u)
            + axes[1] * np.outer(np.cos(lat_curve) * np.sin(lam), v)
            + axes[2] * np.outer(np.sin(lat_curve), pole)
        )
        meridians.append(points + center)

    parallels = []
    for phi in lat_lines:
        points = (
            axes[0] * np.outer(np.cos(lon_curve) * np.cos(phi), u)
            + axes[1] * np.outer(np.sin(lon_curve) * np.cos(phi), v)
            + axes[2] * np.outer(np.ones_like(lon_curve) * np.sin(phi), pole)
        )
        parallels.append(points + center)

    return meridians, parallels


# Quaoar centered at the origin
quaoar_meridians, quaoar_parallels = ellipsoid_wireframe(
    QUAOAR_AXES, quaoar_u, quaoar_v, quaoar_pole
)

# Weywot at its illustrative orbital phase
weywot_center = weywot_position[0]
weywot_meridians, weywot_parallels = ellipsoid_wireframe(
    WEYWOT_AXES,
    weywot_u,
    weywot_v,
    weywot_pole,
    center=weywot_center,
)


# ============================================================
# Plotting helpers
# ============================================================

def plot_curve(ax, xyz, **kwargs):
    east_offset, north_offset = project_sky(xyz)
    ax.plot(-east_offset, north_offset, **kwargs)



def plot_ellipsoid(
    ax, meridians, parallels, axes, u, v, pole,
    center=None, color=color_ellipsoid, lw=0.6, alpha=1.0,
    outline_color=None, outline_lw=1.0,
    outline_ls="-", outline_alpha=1.0,
    n_outline=500, zorder=10
):
    """
    Plot the visible wireframe and smooth projected silhouette
    of a triaxial ellipsoid.

    axes:   semiaxes (a, b, c), in km
    u, v:   equatorial body-axis unit vectors
    pole:   polar-axis unit vector
    center: ellipsoid center in equatorial Cartesian coordinates

    The outline is calculated analytically from the projected
    ellipsoid, so it is independent of wireframe sampling.
    """
    if center is None:
        center = np.zeros(3)

    if outline_color is None:
        outline_color = color

    # --------------------------------------------------------
    # 1. Plot only the Earth-facing wireframe segments
    # --------------------------------------------------------
    for curve in meridians + parallels:
        visible = visible_ellipsoid_points(
            curve, axes, u, v, pole, center=center
        )

        indices = np.flatnonzero(visible)
        if len(indices) == 0:
            continue

        segments = np.split(
            indices,
            np.flatnonzero(np.diff(indices) > 1) + 1
        )

        for segment in segments:
            if len(segment) >= 2:
                plot_curve(
                    ax,
                    curve[segment],
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    zorder=zorder - 1,
                )

    # --------------------------------------------------------
    # 2. Calculate the exact projected ellipse
    # --------------------------------------------------------
    S = (
        axes[0]**2 * np.outer(u, u)
        + axes[1]**2 * np.outer(v, v)
        + axes[2]**2 * np.outer(pole, pole)
    )

    # Projection onto the local East-North sky plane
    sky_basis = np.vstack([east, north])
    C = sky_basis @ S @ sky_basis.T

    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    theta = np.linspace(0.0, 2.0 * np.pi, n_outline)
    circle = np.vstack([np.cos(theta), np.sin(theta)])

    ellipse_en = (
        eigenvectors
        @ np.diag(np.sqrt(eigenvalues))
        @ circle
    )

    # Convert distances to angular offsets
    scale = ARCSEC_PER_RAD / delta_km
    east_center = np.dot(center, east) * scale
    north_center = np.dot(center, north) * scale

    east_outline = east_center + ellipse_en[0] * scale
    north_outline = north_center + ellipse_en[1] * scale

    # East to the left, North up
    ax.plot(
        -east_outline,
        north_outline,
        color=outline_color,
        lw=outline_lw,
        ls=outline_ls,
        alpha=outline_alpha,
        zorder=zorder,
    )


# ============================================================
# Render the Earth-view sky-plane diagram
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Rings: show centerlines and edges
plot_curve(
    ax, q2r_center,
    color=color_ring, lw=lw_Q2R, label="Q2R centerline"
)
plot_curve(ax, q2r_inner, color=color_ring, lw=lw_Q2R, alpha=alpha_Q2R)
plot_curve(ax, q2r_outer, color=color_ring, lw=lw_Q2R, alpha=alpha_Q2R)

plot_curve(
    ax, q1r_center,
    color=color_ring, lw=lw_Q1R, label="Q1R centerline"
)
plot_curve(ax, q1r_inner, color=color_ring, lw=lw_Q1R, alpha=alpha_Q1R)
plot_curve(ax, q1r_outer, color=color_ring, lw=lw_Q1R, alpha=alpha_Q1R)

# Dense Q1R component shown with dashed lines
plot_curve(
    ax, q1r_dense_inner,
    color=color_ring, lw=lw_Q1R, ls=":", alpha=0.9,
    label="Q1R dense component (FWHM)"
)
plot_curve(
    ax, q1r_dense_outer,
    color=color_ring, lw=lw_Q1R, ls=":", alpha=0.9
)

# Satellite orbits
plot_curve(
    ax, weywot_orbit,
    color=color_moon, lw=lw_moon, ls="--", alpha=alpha_moon,
    label="Weywot orbit"
)
plot_curve(
    ax, candidate_orbit,
    color=color_moon, lw=lw_moon, ls="--", alpha=alpha_moon,
    label="Candidate satellite orbit"
)

# Quaoar and Weywot wireframes
plot_ellipsoid(
    ax,
    quaoar_meridians,
    quaoar_parallels,
    QUAOAR_AXES,
    quaoar_u,
    quaoar_v,
    quaoar_pole,
    center=np.zeros(3),
    color=color_ellipsoid,
    lw=0.6,
    alpha=0.75,
    outline_color=color_ellipsoid,
    outline_lw=1.0,
)

# Weywot
plot_ellipsoid(
    ax,
    weywot_meridians,
    weywot_parallels,
    WEYWOT_AXES,
    weywot_u,
    weywot_v,
    weywot_pole,
    center=weywot_center,
    color=color_ellipsoid,
    lw=0.6,
    alpha=0.95,
    outline_color=color_ellipsoid,
    outline_lw=1.0,
)

# Mark Quaoar's center
#ax.scatter(0, 0, color="white", s=10, zorder=5)

# Mark Weywot's illustrative position
wey_e, wey_n = project_sky(weywot_center[None, :])
'''
ax.scatter(
    -wey_e[0], wey_n[0],
    color=color_moon, s=12, zorder=6
)
'''

# Mark candidate's illustrative position (not a predicted ephemeris)
cand_e, cand_n = project_sky(candidate_position)
ax.scatter(
    -cand_e[0], cand_n[0],
    color=color_ellipsoid, s=12/6, zorder=6
)

'''
# Labels
ax.annotate(
    "Quaoar",
    xy=(0, 0),
    xytext=(8, 8),
    textcoords="offset points",
    color=color_ellipsoid
)'''

ax.text(
    x=0.083,
    y=-0.027,
    s="Q2R",
    fontsize=fs_annotation,
    color=color_ring,
    ha="left",
    va="center",
)

ax.text(
    x=0.135,
    y=-0.04,
    s="Q1R",
    fontsize=fs_annotation,
    color=color_ring,
    ha="left",
    va="center",
)

ax.annotate(
    "Weywot",
    xy=(-wey_e[0], wey_n[0]),
    xytext=(8, 8),
    textcoords="offset points",
    color=color_moon,
    fontsize=fs_annotation,
    ha="center"
)

ax.annotate(
    "Candidate satellite",
    xy=(-cand_e[0], cand_n[0]),
    xytext=(8, -20),
    textcoords="offset points",
    color=color_moon,
    fontsize=fs_annotation,
    ha="center"
)

ax.set_xlabel("East ←  Offset (arcsec)  → West", fontsize=fs_axis)
ax.set_ylabel("North ↑  Offset (arcsec)  ↓ South", fontsize=fs_axis)

# Secondary axes: convert angular offsets to projected distances at Quaoar.
# At the geocentric distance delta_km, 1 arcsec corresponds to this many km.
km_per_arcsec = delta_km / ARCSEC_PER_RAD

def arcsec_to_projected_km(offset_arcsec):
    return np.asarray(offset_arcsec) * km_per_arcsec

def projected_km_to_arcsec(distance_km):
    return np.asarray(distance_km) / km_per_arcsec

secax_x = ax.secondary_xaxis(
    "top", functions=(arcsec_to_projected_km, projected_km_to_arcsec)
)
secax_x.set_xlabel("Projected distance at Quaoar (km)", fontsize=fs_axis, labelpad=10)
secax_x.tick_params(labelsize=12)

secax_y = ax.secondary_yaxis(
    "right", functions=(arcsec_to_projected_km, projected_km_to_arcsec)
)
secax_y.set_ylabel("Projected distance at Quaoar (km)", fontsize=fs_axis, labelpad=10)
secax_y.tick_params(labelsize=12)

# Draw a physical scale bar in the lower-left corner of the plotting area.
# The bar length is converted from km to the corresponding angular offset.
fig.canvas.draw()  # ensure the plotted limits have been established
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
bar_length_arcsec = SCALE_BAR_KM / km_per_arcsec
x0 = xmin + 0.07 * (xmax - xmin)
y0 = ymin - 0.03 * (ymax - ymin)
ax.plot(
    [x0, x0 + bar_length_arcsec], [y0, y0],
    color=color_ellipsoid, linewidth=2.0, solid_capstyle="butt", zorder=30
)
ax.text(
    x0 + bar_length_arcsec / 2, y0 + 0.018 * (ymax - ymin),
    f"{SCALE_BAR_KM:g} km", ha="center", va="bottom",
    fontsize=fs_axis, color=color_ellipsoid, zorder=30
)

t = Time(JD, format='jd', scale='utc')
ax.set_title(
    f"Quaoar system as seen from Earth\n{t.datetime} UTC",
    fontsize=fs_title,
    pad=20,
)

ax.set_aspect("equal", adjustable="datalim")
ax.grid(True, alpha=0.25)
#ax.legend(loc="best", fontsize=fs_legend)
plt.tight_layout()
if (DARK_MODE == 'y') or (DARK_MODE == 'Y'):
    plt.savefig("Quaoar_system_sky_view_dark.svg", dpi=200)
else:
    plt.savefig("Quaoar_system_sky_view.svg", dpi=200)
plt.show()


# ============================================================
# Print the assumptions and adopted measurements
# ============================================================

print(f"Observation time: JD {JD} ({t.datetime})")
print(f"Earth-Quaoar distance: {delta_au:.5f} AU")
print(f"Quaoar RA, Dec: {np.rad2deg(ra_q):.5f}, "
      f"{np.rad2deg(dec_q):.5f} deg")
print(f"Quaoar semiaxes (km): {QUAOAR_AXES}")
print(f"Weywot semiaxes (km): {WEYWOT_AXES}")
print(f"Q2R radius/width: {Q2R_RADIUS} / {Q2R_WIDTH} km")
print(f"Q1R radius: {Q1R_RADIUS} km")
print(f"Q1R dense FWHM: {Q1R_DENSE_FWHM} km")
print(f"Q1R representative broad width: {Q1R_BROAD_WIDTH} km")
print(
    f"Candidate satellite radius: {CANDIDATE_RADIUS} "
    f"+/- {CANDIDATE_RADIUS_UNCERTAINTY} km"
)
print("Satellite phases and body rotation phases are illustrative.")

import numpy as np
import matplotlib.pyplot as plt
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

# Light-mode colors
color_moon = 'red'
background_color = 'white'
foreground_color = 'black'
body_line_color = foreground_color
color_ring = 'gray'
color_SOR = 'darkturquoise'
color_MMR = 'magenta'
color_Roche = 'darkorange'

# Opacities
alpha_moon = 0.75
alpha_Q1R = 0.65
alpha_Q2R = 0.65

# Linewidths
lw_moon = 1.5
lw_Q1R = 0.7
lw_Q2R = 0.6


# Optional dark mode: set True for a black background and white body lines.
DARK_MODE = input("Dark mode? (y/n): ")  # False = light background
if (DARK_MODE == 'y') or (DARK_MODE == 'Y'):
    print("Dark mode enabled.")
    DARK_MODE = True

    plt.style.use('dark_background')
    background_color = "black"
    foreground_color = "white"
    body_line_color = foreground_color
    color_moon = 'lightcoral'
    color_ring = 'lightgray'
    color_SOR = 'turquoise'
    color_Roche = 'orange'

else:
    print("No dark mode, moving on...")
    DARK_MODE = False


# ============================================================
# Observation and system parameters
# ============================================================

JD = 2461200.5
AU_KM = 149597870.7
ARCSEC_PER_RAD = 206264.806247

# Quaoar's preferred occultation-derived dimensions
# Full diameters in km (Margoti et al. 2026b)
QUAOAR_AXES = np.array([1241.4, 1093.0, 1023.0]) / 2.0

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

# Scale bar
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
# Render a pole-on view of the Quaoar system
# ============================================================

# Projection onto the ring/equatorial plane. The x/y axes are defined by
# ring_u and ring_v, so all rings in this plane appear as true circles.
def project_pole_on(xyz):
    xyz = np.atleast_2d(xyz)
    return xyz @ ring_u, xyz @ ring_v

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)


def plot_top_curve(xyz, **kwargs):
    x, y = project_pole_on(xyz)
    ax.plot(x, y, **kwargs)


# Rings: centerlines and edges; these are circular in this pole-on view.
plot_top_curve(q2r_center, color=color_ring, lw=1.1, label="Q2R centerline")
plot_top_curve(q2r_inner, color=color_ring, lw=0.7, alpha=alpha_Q2R)
plot_top_curve(q2r_outer, color=color_ring, lw=0.7, alpha=alpha_Q2R)

plot_top_curve(q1r_center, color=color_ring, lw=lw_Q1R, label="Q1R centerline")
plot_top_curve(q1r_inner, color=color_ring, lw=lw_Q1R, alpha=alpha_Q1R)
plot_top_curve(q1r_outer, color=color_ring, lw=lw_Q1R, alpha=alpha_Q1R)
plot_top_curve(q1r_dense_inner, color=color_ring, lw=lw_Q1R, ls=":", alpha=0.9, label="Q1R dense component (FWHM)")
plot_top_curve(q1r_dense_outer, color=color_ring, lw=lw_Q1R, ls=":", alpha=0.9)

# Satellite orbits. Weywot's orbit may appear slightly elliptical because its
# adopted orbital plane is not exactly aligned with the ring plane.
plot_top_curve(weywot_orbit, color=color_moon, lw=lw_moon, ls="--",
               alpha=alpha_moon, label="Weywot orbit")
plot_top_curve(candidate_orbit, color=color_moon, lw=lw_moon, ls="--",
               alpha=alpha_moon, label="Candidate satellite orbit")

# Project the sparse, smooth body wireframes into the ring plane.
# All wireframe curves are shown in this overhead projection.
def plot_top_wireframe(meridians, parallels, color, lw=0.65, alpha=0.9, zorder=10):
    for curve in meridians + parallels:
        plot_top_curve(curve, color=color, lw=lw, alpha=alpha, zorder=zorder)

plot_top_wireframe(quaoar_meridians, quaoar_parallels,
                   color=body_line_color, lw=0.75, alpha=0.9, zorder=12)
plot_top_wireframe(weywot_meridians, weywot_parallels,
                   color=body_line_color, lw=0.6, alpha=0.95, zorder=13)

# Draw a distinct, thicker projected silhouette around each ellipsoid.
# Increase BODY_OUTLINE_LW to thicken only these borders; wireframe widths
# above remain unchanged.
BODY_OUTLINE_LW = 1.5
BODY_OUTLINE_N = 500


def plot_projected_ellipsoid_outline(axes, u, v, pole, center=None,
                                     lw=BODY_OUTLINE_LW, zorder=15):
    """Plot the projected silhouette of a triaxial ellipsoid in the ring plane."""
    if center is None:
        center = np.zeros(3)

    # Covariance of the ellipsoid projected onto the ring-plane X/Y axes.
    projected_axes = np.array([
        [np.dot(u, ring_u), np.dot(v, ring_u), np.dot(pole, ring_u)],
        [np.dot(u, ring_v), np.dot(v, ring_v), np.dot(pole, ring_v)],
    ])
    shape = projected_axes @ np.diag(np.asarray(axes) ** 2) @ projected_axes.T
    eigenvalues, eigenvectors = np.linalg.eigh(shape)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    theta = np.linspace(0.0, 2.0 * np.pi, BODY_OUTLINE_N)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    ellipse = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ circle
    center_x, center_y = project_pole_on(np.asarray(center)[None, :])
    x = ellipse[0] + center_x[0]
    y = ellipse[1] + center_y[0]
    ax.plot(x, y, color=body_line_color, lw=lw, zorder=zorder)


plot_projected_ellipsoid_outline(
    QUAOAR_AXES, quaoar_u, quaoar_v, quaoar_pole, center=np.zeros(3)
)
plot_projected_ellipsoid_outline(
    WEYWOT_AXES, weywot_u, weywot_v, weywot_pole, center=weywot_center
)


# Mark the illustrative satellite positions.
wx, wy = project_pole_on(weywot_center[None, :])
cx, cy = project_pole_on(candidate_position)
#ax.scatter(wx[0], wy[0], color=color_moon, s=14, zorder=20)
ax.scatter(cx, cy, color=foreground_color, s=12/6, zorder=20)

ax.text(
    x=Q2R_RADIUS*1.04,
    y=0,
    s="Q2R",
    fontsize=fs_annotation,
    color=color_ring,
    ha="left",
    va="center",
)
ax.text(
    x=Q1R_RADIUS*1.04,
    y=0,
    s="Q1R",
    fontsize=fs_annotation,
    color=color_ring,
    ha="left",
    va="center",
)
ax.annotate("Weywot", (wx[0], wy[0]), xytext=(7, 0),
            textcoords="offset points", color=color_moon,
            fontsize=fs_annotation, ha="left", va="center")
ax.annotate("Candidate\nsatellite", (cx[0], cy[0]), xytext=(-6, 0),
            textcoords="offset points", color=color_moon,
            fontsize=fs_annotation, ha="right", va="center")

# Physical axes are in kilometres, so no angular secondary axes are needed.
ax.set_xlabel("Equatorial plane X (km)", fontsize=fs_axis, color=foreground_color)
ax.set_ylabel("Equatorial plane Y (km)", fontsize=fs_axis, color=foreground_color)

t = Time(JD, format='jd', scale='utc')
ax.set_title(f"Quaoar system viewed from above the north pole\n{t.datetime} UTC",
             fontsize=fs_title, color=foreground_color)
ax.set_aspect("equal", adjustable="box")
ax.tick_params(axis="both", colors=foreground_color)
for spine in ax.spines.values():
    spine.set_color(foreground_color)
ax.grid(True, alpha=0.25)

# Physical scale bar.
fig.canvas.draw()
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
xrange, yrange = xmax - xmin, ymax - ymin
bar_length = SCALE_BAR_KM
x0 = xmin + 0.07 * xrange
y0 = ymin + 0.08 * yrange
ax.plot([x0, x0 + bar_length], [y0, y0], color=foreground_color,
        linewidth=2.0, solid_capstyle="butt", zorder=30)
ax.text(x0 + bar_length / 2, y0 + 0.018 * yrange,
        f"{SCALE_BAR_KM:g} km", ha="center", va="bottom",
        fontsize=fs_axis, color=foreground_color, zorder=30)


# ============================================================
# Calculate and plot resonances
# ============================================================
QUAOAR_MASS_KG = 1.21227e21
QUAOAR_ROTPER_S = 17.6788 * 3600
WEYWOT_PERIOD_S = WEYWOT_PERIOD_DAYS * 24 * 3600
G = 6.67430e-11

def period_to_sma(T_s):
    a_m = ((G * QUAOAR_MASS_KG * T_s**2) / (4*np.pi**2))**(1/3)
    a_km = a_m / 1000
    return a_km

# Calculate semi-major axes of spin-orbit resonances
SOR_3_1 = period_to_sma(QUAOAR_ROTPER_S * 3)    # 3:1 SOR
SOR_7_5 = period_to_sma(QUAOAR_ROTPER_S * (7/5))    # 7:5 SOR
# Calculate semi-major axes mean-motion resonances w.r.t. Weywot
MMR_6_1 = period_to_sma(WEYWOT_PERIOD_S / 6)    # 6:1 SOR
MMR_7_2 = period_to_sma(WEYWOT_PERIOD_S * (2/7))    # 7:2 SOR

# Plot spin-orbit resonances (SORs; blue)
ax.add_patch(plt.Circle((0, 0), SOR_3_1, color=color_SOR, fill=False, ls='-', lw=lw_moon*2, alpha=alpha_moon/2))
ax.add_patch(plt.Circle((0, 0), SOR_7_5, color=color_SOR, fill=False, ls='-', lw=lw_moon*2, alpha=alpha_moon/2))
# Plot mean-motion resonances (MMRs; purple)
ax.add_patch(plt.Circle((0, 0), MMR_6_1, color=color_MMR, fill=False, ls='-', lw=lw_moon*2, alpha=alpha_moon/2))
ax.add_patch(plt.Circle((0, 0), MMR_7_2, color=color_MMR, fill=False, ls='-', lw=lw_moon*2, alpha=alpha_moon/2))

# Label resonances
ax.text(
    x=0,
    y=SOR_7_5*1.05,
    s="7:5 SOR",
    fontsize=fs_annotation,
    color=color_SOR,
    alpha=alpha_moon,
    ha="center",
    va="bottom",
)
ax.text(
    x=-1500,
    y=SOR_3_1*1.01-400,
    s="3:1 SOR",
    fontsize=fs_annotation,
    color=color_SOR,
    alpha=alpha_moon,
    ha="center",
    va="bottom",
    rotation=15,    # positive rotation = counterclockwise
)
ax.text(
    x=1850,
    y=SOR_3_1*0.97-400, # same y as the 3:1 SOR label for consistency
    s="6:1W MMR",
    fontsize=fs_annotation,
    color=color_MMR,
    alpha=alpha_moon,
    ha="center",
    va="bottom",
    rotation=-15,
)
ax.text(
    x=0,
    y=MMR_7_2*1.05, # same y as the 3:1 SOR label for consistency
    s="7:2W MMR",
    fontsize=fs_annotation,
    color=color_MMR,
    alpha=alpha_moon,
    ha="center",
    va="bottom",
)

# Plot Quaoar's Roche limit for p=0.4 g/cm3 satellites (Morgado et al. 2023)
QUAOAR_ROCHE_LIMIT = 1780   # in km

ax.add_patch(plt.Circle((0, 0), QUAOAR_ROCHE_LIMIT, color=color_Roche, fill=False, ls=':', lw=lw_moon, alpha=alpha_moon))
ax.text(
    x=0,
    y=QUAOAR_ROCHE_LIMIT*0.8, # same y as the 3:1 SOR label for consistency
    s="Roche",
    fontsize=fs_annotation,
    color=color_Roche,
    alpha=alpha_moon,
    ha="center",
    va="top",
)
ax.text(
    x=0,
    y=-QUAOAR_ROCHE_LIMIT*0.8, # same y as the 3:1 SOR label for consistency
    s="limit",
    fontsize=fs_annotation,
    color=color_Roche,
    alpha=alpha_moon,
    ha="center",
    va="bottom",
)

# ============================================================
# Save and show plot
# ============================================================

plt.tight_layout()
if (DARK_MODE == True):
    plt.savefig("Quaoar_system_north_pole_view_dark.svg", dpi=200, facecolor=fig.get_facecolor())
else:
    plt.savefig("Quaoar_system_north_pole_view.svg", dpi=200, facecolor=fig.get_facecolor())

plt.show()

print(f"Pole-on view at: JD {JD} ({t.datetime})")
print(f"Q2R radius/width: {Q2R_RADIUS} / {Q2R_WIDTH} km")
print(f"Q1R radius: {Q1R_RADIUS} km")
print(f"Weywot orbital radius: {WEYWOT_ORBIT_RADIUS} km")
print(f"Candidate satellite radius: {CANDIDATE_RADIUS} +/- {CANDIDATE_RADIUS_UNCERTAINTY} km")
print("Satellite phases and body rotation phases are illustrative.")

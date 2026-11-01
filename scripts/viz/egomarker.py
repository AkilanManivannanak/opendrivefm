"""A top-down car silhouette for bird's-eye panels, in data coordinates.

Every bird's-eye figure used marker="^" for the ego vehicle. A triangle reads
as a generic direction glyph rather than a vehicle, and at the sizes these
figures print it is the first thing a reader's eye lands on in the middle of
the plot. This draws a car instead: body, windscreen, rear glass, all as one
path collection so it scales with the axes like any other artist.
"""
from matplotlib.patches import Polygon
import numpy as np


def ego_car(ax, x=0.0, y=0.0, length=4.6, yaw_deg=90.0, zorder=7,
            body="white", glass="#2b2f3a", edge="#11141a", lw=0.5):
    """Draw the ego vehicle at (x, y). yaw_deg 90 points up the +y axis."""
    L, W = length, length * 0.44
    hl, hw = L / 2, W / 2
    body_pts = np.array([
        (-hw*0.74,  hl), ( hw*0.74,  hl), ( hw,  hl*0.52), ( hw, -hl*0.66),
        ( hw*0.84, -hl), (-hw*0.84, -hl), (-hw, -hl*0.66), (-hw,  hl*0.52)])
    wind = np.array([(-hw*0.66, hl*0.46), (hw*0.66, hl*0.46),
                     ( hw*0.52, hl*0.80), (-hw*0.52, hl*0.80)])
    rear = np.array([(-hw*0.62, -hl*0.30), (hw*0.62, -hl*0.30),
                     ( hw*0.52, -hl*0.66), (-hw*0.52, -hl*0.66)])
    th = np.deg2rad(yaw_deg - 90.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for pts, fc, z in ((body_pts, body, zorder), (wind, glass, zorder + 1),
                       (rear, glass, zorder + 1)):
        p = (R @ pts.T).T + np.array([x, y])
        ax.add_patch(Polygon(p, closed=True, facecolor=fc, edgecolor=edge,
                             linewidth=lw, zorder=z))

"""
3D PyVista renderer for POPGP projection outputs.

Provides interactive and off-screen visualizations of the emergent geometry,
including warped gravity-well surfaces and local metric tensor ellipsoids.

Implements visualization of Section 4.4.4 (Π_geom) and Section 4.4.5 (Π_time)
outputs from docs/framework.md.

Requires: ``pyvista`` (optional dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import Delaunay

if TYPE_CHECKING:
    from popgp.simulator import PiGeomResult, PiTimeResult


def _require_pyvista():
    try:
        import pyvista as pv
        return pv
    except ImportError:
        raise ImportError(
            "pyvista is required for 3D rendering. Install with: uv add pyvista"
        ) from None


class POPGPRenderer:
    """Renders POPGP projection outputs as 3D interactive scenes.

    Parameters
    ----------
    coords : np.ndarray
        Embedded coordinates from Π_geom, shape ``[N, D*]``.
    D_star : int
        Emergent embedding dimension.
    phi : np.ndarray
        Clock-rate potential from Π_time (or a custom source solve),
        shape ``[N]``.
    edges : list[tuple[int, int]] | None
        Graph connectivity edges for wireframe overlay.
    """

    def __init__(
        self,
        coords: np.ndarray,
        D_star: int,
        phi: np.ndarray,
        edges: list[tuple[int, int]] | None = None,
    ) -> None:
        self.coords = np.asarray(coords, dtype=np.float64)
        self.D = D_star
        self.phi = np.asarray(phi, dtype=np.float64)
        self.edges = edges

        if self.D == 2:
            self.points_3d = np.column_stack(
                (self.coords, np.zeros(len(self.coords)))
            )
        elif self.D >= 3:
            self.points_3d = self.coords[:, :3]
        else:
            self.points_3d = np.column_stack(
                (self.coords, np.zeros((len(self.coords), 2)))
            )[:, :3]

        self.tri = Delaunay(self.coords[:, :2] if self.D >= 2 else self.coords)

    @classmethod
    def from_result(
        cls,
        pi_geom: PiGeomResult,
        phi: np.ndarray,
        edges: list[tuple[int, int]] | None = None,
    ) -> POPGPRenderer:
        """Construct from simulator pipeline result objects."""
        coords = pi_geom.coords
        if hasattr(coords, "numpy"):
            coords = coords.numpy()
        return cls(
            coords=coords,
            D_star=pi_geom.D_star,
            phi=phi,
            edges=edges,
        )

    def render_gravity_well(
        self,
        save_path: str | None = None,
        warp_scale: float = 2.0,
        title: str = "POPGP: Emergent Gravity Well from Entropy Deficit",
        cmap: str = "magma",
        show_edges: bool = True,
        window_size: tuple[int, int] = (1400, 900),
    ) -> None:
        """Render a 2.5D warped surface where Z = Φ * warp_scale.

        Uses a Delaunay triangulation of the epistemic MDS coordinates as a
        computationally efficient proxy for the Vietoris-Rips simplicial
        complex (§8.2.1).  The clock-rate potential Φ is mapped to the
        vertical axis, producing a Newtonian gravity-well surface.

        Parameters
        ----------
        save_path : str or None
            If provided, save a screenshot instead of opening a window.
        warp_scale : float
            Vertical exaggeration factor for Φ.
        title : str
            Plot title.
        cmap : str
            Colormap name (``'magma'``, ``'inferno'``, ``'viridis'``).
        show_edges : bool
            Whether to draw triangle edges on the mesh.
        window_size : tuple
            Render window dimensions in pixels.
        """
        pv = _require_pyvista()

        assert self.D >= 2, "Gravity well surface requires D* >= 2"

        warped = self.points_3d.copy()
        warped[:, 2] = self.phi * warp_scale

        faces = np.column_stack(
            (np.full(len(self.tri.simplices), 3), self.tri.simplices)
        ).flatten()

        mesh = pv.PolyData(warped, faces)
        mesh.point_data["Clock Potential (Phi)"] = self.phi

        plotter = pv.Plotter(
            off_screen=(save_path is not None),
            window_size=window_size,
        )

        plotter.add_mesh(
            mesh,
            scalars="Clock Potential (Phi)",
            cmap=cmap,
            show_edges=show_edges,
            edge_color="white",
            line_width=0.8,
            lighting=True,
        )

        plotter.add_text(title, font_size=12, position="upper_left")
        plotter.add_scalar_bar(
            title="Φ (Clock Rate)",
            n_labels=5,
            shadow=True,
        )

        plotter.set_background("black", top="midnightblue")
        plotter.camera.elevation = -30
        plotter.camera.azimuth = 45

        if save_path:
            plotter.screenshot(save_path)
            print(f"Saved 3D gravity well to {save_path}")
        else:
            plotter.show()

        plotter.close()

    def render_flat_embedding(
        self,
        save_path: str | None = None,
        cmap: str = "inferno",
        window_size: tuple[int, int] = (1200, 900),
    ) -> None:
        """Render the flat 2D embedding with Φ as vertex color.

        Useful for comparing against the warped gravity-well view.
        """
        pv = _require_pyvista()

        faces = np.column_stack(
            (np.full(len(self.tri.simplices), 3), self.tri.simplices)
        ).flatten()

        mesh = pv.PolyData(self.points_3d, faces)
        mesh.point_data["Clock Potential (Phi)"] = self.phi

        plotter = pv.Plotter(
            off_screen=(save_path is not None),
            window_size=window_size,
        )

        plotter.add_mesh(
            mesh,
            scalars="Clock Potential (Phi)",
            cmap=cmap,
            show_edges=True,
            edge_color="gray",
            line_width=0.5,
        )

        plotter.add_text(
            "POPGP: Flat Embedding with Clock Potential",
            font_size=12,
            position="upper_left",
        )
        plotter.view_xy()
        plotter.set_background("black")

        if save_path:
            plotter.screenshot(save_path)
            print(f"Saved flat embedding to {save_path}")
        else:
            plotter.show()

        plotter.close()

    def render_tensor_ellipsoids(
        self,
        h_ab: np.ndarray,
        save_path: str | None = None,
        scale_factor: float = 0.15,
        window_size: tuple[int, int] = (1200, 900),
    ) -> None:
        """Render local metric tensors h_ab as oriented ellipsoids.

        Each node gets an ellipsoid whose principal axes are the
        eigenvectors of h_ab(x_i) and whose semi-axis lengths are
        the eigenvalues. This visualizes entanglement shear (Section 8.1).

        Parameters
        ----------
        h_ab : np.ndarray
            Local metric tensors, shape ``[N, D*, D*]``.
        save_path : str or None
            If provided, save a screenshot.
        scale_factor : float
            Global glyph scale.
        window_size : tuple
            Render window dimensions.
        """
        pv = _require_pyvista()

        h_ab = np.asarray(h_ab)
        N = len(self.coords)

        tensors_3x3 = np.zeros((N, 3, 3))
        tensors_3x3[:, :self.D, :self.D] = h_ab
        if self.D == 2:
            tensors_3x3[:, 2, 2] = 1.0

        cloud = pv.PolyData(self.points_3d)
        cloud.point_data["h_ab"] = tensors_3x3.reshape(-1, 9)
        cloud.point_data["Clock Potential (Phi)"] = self.phi

        glyphs = cloud.glyph(
            geom=pv.Sphere(theta_resolution=16, phi_resolution=16),
            orient="h_ab",
            scale="h_ab",
            factor=scale_factor,
        )

        plotter = pv.Plotter(
            off_screen=(save_path is not None),
            window_size=window_size,
        )
        plotter.add_mesh(glyphs, color="cyan", opacity=0.9)
        plotter.add_mesh(cloud, color="white", point_size=6, render_points_as_spheres=True)
        plotter.add_text(
            "POPGP: Local Metric Tensor Ellipsoids (Entanglement Shear)",
            font_size=12,
            position="upper_left",
        )
        plotter.set_background("black", top="midnightblue")

        if save_path:
            plotter.screenshot(save_path)
            print(f"Saved tensor ellipsoids to {save_path}")
        else:
            plotter.show()

        plotter.close()

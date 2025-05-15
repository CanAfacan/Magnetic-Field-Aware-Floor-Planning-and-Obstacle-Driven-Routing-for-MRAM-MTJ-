import numpy as np

#vacuum permeability
mu0 = 4 * np.pi * 1e-7 

class MTJDevice:
    """
    Represents an STT-MTJ stack: free, barrier, reference, fixed (SAF), antiferro.
    Layers: list of dicts {'name', 'thickness' (m), 'Ms' (A/m)}.
    """
    def __init__(self, layers):
        self.layers = layers

    def magnetic_moment(self, area=1e-12):
        """m = Σ (Ms * Volume). area default = 1µm²."""
        m = 0.0
        for layer in self.layers:
            Ms = layer.get('Ms', 0)
            m += Ms * area * layer['thickness']
        return m

    def field_zone_radius(self,
                          area=None,
                          unit=1e-6,
                          B_thresh=1e-3):
        """
        area: block area in floor‐units² (we'll convert to m² via unit²)
        unit: meters per floor‐unit (default 1µm =1e-6m)
        Returns radius in floor‐units
        """
        # to m^2
        if area is None:
            area_m2 = 1e-12
        else:
            area_m2 = area * (unit**2)
        # dipole moment m = sigma Ms·V
        m = 0.0
        for lay in self.layers:
            Ms = lay.get('Ms', 0)
            t = lay['thickness']
            m += Ms * area_m2 * t
        # Bz(r_m) = B_thresh  ⇒  r_m = (mu0*m/(2piB))^(1/3)
        r_m = (mu0 * m / (2 * np.pi * B_thresh)) ** (1/3)
        return r_m / unit
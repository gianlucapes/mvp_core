# src/pinn_heat/domain/2Dplate.py

from dataclasses import dataclass
import jax.numpy as jnp
import jax

@dataclass(frozen=True)
class Plate2d:
    """Piastra quadrata 2D per la conduzione del calore stazionaria.

    Le coordinate sono in millimetri. Internamente i punti vengono
    normalizzati in [0, 1] prima di essere passati alla rete.

    Args:
        width_mm:   larghezza della piastra [mm]
        height_mm:  altezza della piastra [mm]
        t_hot:      temperatura alla base (contatto chip) [°K]
        t_amb:      temperatura sui tre lati esposti all'aria [°K]
    """
    width_mm:  float
    height_mm: float
    t_hot:     float
    t_amb:     float

    def normalize(self, x_mm: jnp.ndarray, y_mm: jnp.ndarray):
        """Converte coordinate fisiche [mm] → [0, 1]."""
        return x_mm / self.width_mm, y_mm / self.height_mm

    def denormalize(self, x: jnp.ndarray, y: jnp.ndarray):
        """Converte coordinate normalizzate [0, 1] → fisiche [mm]."""
        return x * self.width_mm, y * self.height_mm

    def collocation_points(self, n: int, key: jax.Array, normalized: bool = True) -> jnp.ndarray:
        """Campiona n punti interni con Latin Hypercube Sampling (LHS).

        Il dominio [0,1]² viene diviso in n celle per asse; ogni cella
        contribuisce esattamente un punto, garantendo copertura uniforme.

        Returns:
            Array di shape (n, 2). Colonne: [x, y].
            Se normalized=True le coordinate sono in [0, 1],
            altrimenti in mm.
        """
        key_x, key_y, key_px, key_py = jax.random.split(key, 4)

        # Offset casuale dentro ogni cella → punto in [i/n, (i+1)/n]
        cells = jnp.arange(n)
        x = (cells + jax.random.uniform(key_x, (n,))) / n
        y = (cells + jax.random.uniform(key_y, (n,))) / n

        # Permutazione indipendente dei due assi
        x = x[jax.random.permutation(key_px, n)]
        y = y[jax.random.permutation(key_py, n)]

        if not normalized:
            x, y = self.denormalize(x, y)

        return jnp.stack([x, y], axis=-1)


    def boundary_points(
        self, n_per_side: int, normalized: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Genera n_per_side punti equidistanti su ciascuno dei 4 lati.

        Returns:
            points: Array (4 * n_per_side, 2) con le coordinate [x, y].
            t_bc:   Array (4 * n_per_side,)  con la temperatura imposta.

        Layout bordi (coordinate normalizzate):
            base  → y = 0  (T = t_hot)
            top   → y = 1  (T = t_amb)
            left  → x = 0  (T = t_amb)
            right → x = 1  (T = t_amb)
        """
        t = jnp.linspace(0.0, 1.0, n_per_side)

        base  = jnp.stack([t,            jnp.zeros(n_per_side)], axis=-1)
        top   = jnp.stack([t,            jnp.ones(n_per_side)],  axis=-1)
        left  = jnp.stack([jnp.zeros(n_per_side), t],            axis=-1)
        right = jnp.stack([jnp.ones(n_per_side),  t],            axis=-1)

        t_base  = jnp.full(n_per_side, self.t_hot)
        t_sides = jnp.full(n_per_side * 3, self.t_amb)

        points = jnp.concatenate([base, top, left, right], axis=0)
        t_bc   = jnp.concatenate([t_base, t_sides], axis=0)

        if not normalized:
            x, y = self.denormalize(points[:, 0], points[:, 1])
            points = jnp.stack([x, y], axis=-1)

        return points, t_bc
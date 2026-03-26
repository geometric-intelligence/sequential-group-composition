import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class OnlineModularAdditionDataset2D(IterableDataset):
    """
    Online dataset that generates 2D modular addition samples on-the-fly.
    Fully GPU-accelerated for maximum throughput.
    """

    def __init__(
        self,
        p1: int,
        p2: int,
        template: np.ndarray,
        k: int,
        batch_size: int,
        device: str,
        return_all_outputs: bool = False,
    ):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.k = k
        self.batch_size = batch_size
        self.p_flat = p1 * p2
        self.device = device
        self.return_all_outputs = return_all_outputs

        # Store template on GPU for fast rolling
        self.template_gpu = torch.tensor(template, device=device, dtype=torch.float32)

        # Pre-compute coordinate grids on GPU for efficient rolling
        x_coords = torch.arange(p1, device=device)
        y_coords = torch.arange(p2, device=device)
        self.x_grid, self.y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")

    def _roll_2d_batch(self, shifts_x, shifts_y):
        """
        Roll the template by different amounts for each sample in a batch.
        Fully vectorized on GPU.

        Args:
            shifts_x: (batch_size,) or (batch_size, k) tensor of row shifts
            shifts_y: (batch_size,) or (batch_size, k) tensor of col shifts

        Returns:
            Rolled templates: (batch_size, p1, p2) or (batch_size, k, p1, p2)
        """
        # Determine output shape based on input
        if shifts_x.dim() == 1:
            # Single roll per sample: (batch_size,)
            batch_size = shifts_x.shape[0]
            # Broadcast: (1, p1, p2) -> (batch_size, p1, p2)
            x_grid = self.x_grid.unsqueeze(0)  # (1, p1, p2)
            y_grid = self.y_grid.unsqueeze(0)  # (1, p1, p2)
            shifts_x = shifts_x.view(batch_size, 1, 1)  # (batch_size, 1, 1)
            shifts_y = shifts_y.view(batch_size, 1, 1)  # (batch_size, 1, 1)
        else:
            # Multiple rolls per sample: (batch_size, k)
            batch_size, k = shifts_x.shape
            # Broadcast: (1, 1, p1, p2) -> (batch_size, k, p1, p2)
            x_grid = self.x_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, p1, p2)
            y_grid = self.y_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, p1, p2)
            shifts_x = shifts_x.view(batch_size, k, 1, 1)  # (batch_size, k, 1, 1)
            shifts_y = shifts_y.view(batch_size, k, 1, 1)  # (batch_size, k, 1, 1)

        # Compute shifted coordinates with modular arithmetic
        x_shifted = (x_grid - shifts_x) % self.p1
        y_shifted = (y_grid - shifts_y) % self.p2

        # Index into template using advanced indexing
        rolled = self.template_gpu[x_shifted.long(), y_shifted.long()]

        return rolled

    def __iter__(self):
        """Generate batches indefinitely on GPU."""
        while True:
            # Generate random shifts on GPU: (batch_size, k)
            shifts_x = torch.randint(
                0, self.p1, (self.batch_size, self.k), device=self.device, dtype=torch.long
            )
            shifts_y = torch.randint(
                0, self.p2, (self.batch_size, self.k), device=self.device, dtype=torch.long
            )

            # Generate X: roll template for each time step
            # Shape: (batch_size, k, p1, p2)
            X_rolled = self._roll_2d_batch(shifts_x, shifts_y)

            # Reshape to (batch_size, k, p_flat)
            X = X_rolled.reshape(self.batch_size, self.k, self.p_flat)

            if self.return_all_outputs:
                # Generate Y for ALL cumulative sums (intermediate targets)
                # Compute cumulative sum at each timestep
                sx_cumsum = torch.cumsum(shifts_x, dim=1) % self.p1  # (batch_size, k)
                sy_cumsum = torch.cumsum(shifts_y, dim=1) % self.p2  # (batch_size, k)

                # Roll by all cumulative sums: (batch_size, k, p1, p2)
                Y_rolled = self._roll_2d_batch(sx_cumsum, sy_cumsum)

                # Reshape to (batch_size, k, p_flat)
                Y = Y_rolled.reshape(self.batch_size, self.k, self.p_flat)
                Y = Y[:, 1:, :]

            else:
                # Generate Y: only final cumulative sum (current behavior)
                sx_cumsum = shifts_x.sum(dim=1) % self.p1  # (batch_size,)
                sy_cumsum = shifts_y.sum(dim=1) % self.p2  # (batch_size,)

                # Shape: (batch_size, p1, p2)
                Y_rolled = self._roll_2d_batch(sx_cumsum, sy_cumsum)

                # Reshape to (batch_size, p_flat)
                Y = Y_rolled.reshape(self.batch_size, self.p_flat)

            yield X, Y


class OnlineModularAdditionDataset1D(IterableDataset):
    """
    Online dataset that generates 1D modular addition samples on-the-fly.
    Fully GPU-accelerated for maximum throughput.
    """

    def __init__(
        self,
        p: int,
        template: np.ndarray,
        k: int,
        batch_size: int,
        device: str,
        return_all_outputs: bool = False,
    ):
        super().__init__()
        self.p = p
        self.k = k
        self.batch_size = batch_size
        self.device = device
        self.return_all_outputs = return_all_outputs

        # Store template on GPU for fast rolling
        self.template_gpu = torch.tensor(template, device=device, dtype=torch.float32)

    def _roll_1d_batch(self, shifts):
        """
        Roll the 1D template by different amounts for each sample in a batch.
        Fully vectorized on GPU.

        Args:
            shifts: (batch_size,) or (batch_size, k) tensor of shifts

        Returns:
            Rolled templates: (batch_size, p) or (batch_size, k, p)
        """
        if shifts.dim() == 1:
            # Single roll per sample: (batch_size,)
            batch_size = shifts.shape[0]
            # Use advanced indexing
            indices = (
                torch.arange(self.p, device=self.device).unsqueeze(0) - shifts.unsqueeze(1)
            ) % self.p
            rolled = self.template_gpu[indices.long()]
        else:
            # Multiple rolls per sample: (batch_size, k)
            batch_size, k = shifts.shape
            indices = (
                torch.arange(self.p, device=self.device).unsqueeze(0).unsqueeze(0)
                - shifts.unsqueeze(2)
            ) % self.p
            rolled = self.template_gpu[indices.long()]

        return rolled

    def __iter__(self):
        """Generate batches indefinitely on GPU."""
        while True:
            # Generate random shifts on GPU: (batch_size, k)
            shifts = torch.randint(
                0, self.p, (self.batch_size, self.k), device=self.device, dtype=torch.long
            )

            # Generate X: roll template for each time step
            # Shape: (batch_size, k, p)
            X = self._roll_1d_batch(shifts)

            if self.return_all_outputs:
                # Generate Y for ALL cumulative sums (intermediate targets)
                shifts_cumsum = torch.cumsum(shifts, dim=1) % self.p  # (batch_size, k)

                # Roll by all cumulative sums: (batch_size, k, p)
                Y = self._roll_1d_batch(shifts_cumsum)
                Y = Y[:, 1:, :]  # Remove first timestep
            else:
                # Generate Y: only final cumulative sum
                shifts_cumsum = shifts.sum(dim=1) % self.p  # (batch_size,)

                # Shape: (batch_size, p)
                Y = self._roll_1d_batch(shifts_cumsum)

            yield X, Y


class OfflineModularCompositionDataset(Dataset):
    """PyTorch map-style dataset for group composition tasks.

    Stores pre-generated input-output pairs as float32 tensors and supports
    indexing via ``__getitem__`` / ``__len__`` for use with ``DataLoader``.

    Construct instances via the factory classmethods:
      - :meth:`from_group`  -- any escnn group with a regular representation
      - :meth:`from_cn`     -- cyclic group C_p (1D np.roll, no escnn needed)
      - :meth:`from_cnxcn`  -- product group C_{p1} x C_{p2} (2D np.roll, no escnn needed)

    All factories support arbitrary sequence length ``k``, ``"sampled"`` /
    ``"exhaustive"`` mode, and ``return_all_outputs``.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """Create a dataset from numpy arrays.

        Parameters
        ----------
        X : np.ndarray
            Input data (any shape with first axis = N).
        Y : np.ndarray
            Target data (any shape with first axis = N).
        """
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.Y = torch.tensor(np.asarray(Y), dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    @classmethod
    def from_group(
        cls,
        template: np.ndarray,
        k: int,
        group,
        mode: str = "sampled",
        num_samples: int = 65536,
        return_all_outputs: bool = False,
    ) -> tuple["OfflineModularCompositionDataset", np.ndarray]:
        """Build generic group composition dataset for sequence length k.

        Works with any escnn group that has a regular representation.

        Parameters
        ----------
        template : np.ndarray, shape (group_order,)
        k : int
            Sequence length (number of group elements to compose).
        group : escnn group object
        mode : str
            ``"sampled"`` or ``"exhaustive"``.
        num_samples : int
            Number of samples when ``mode="sampled"``.
        return_all_outputs : bool
            If True, return intermediate composition outputs.

        Returns
        -------
        dataset : OfflineModularCompositionDataset
        sequence : np.ndarray, shape (N, k)
            Integer indices of group elements per token.
        """
        group_order = group.order()

        assert template.shape == (group_order,), (
            f"template must be ({group_order},), got {template.shape}"
        )

        regular_rep = group.representations["regular"]
        elements = list(group.elements)
        n_elements = len(elements)

        rep_matrices = np.array([regular_rep(g) for g in elements])

        if mode == "exhaustive":
            total = n_elements**k
            if total > 1_000_000:
                raise ValueError(f"n_elements^k = {total} is huge; use mode='sampled' instead.")
            N = total
            sequence = np.zeros((N, k), dtype=np.int64)
            for idx in range(N):
                for t in range(k):
                    sequence[idx, t] = (idx // (n_elements**t)) % n_elements
        else:
            N = int(num_samples)
            sequence = np.random.randint(0, n_elements, size=(N, k), dtype=np.int64)

        X = np.zeros((N, k, group_order), dtype=np.float32)
        Y = np.zeros((N, k, group_order), dtype=np.float32)

        for i in range(N):
            cumulative_rep = np.eye(group_order)
            for t in range(k):
                elem_idx = sequence[i, t]
                g_rep = rep_matrices[elem_idx]
                X[i, t, :] = g_rep @ template
                cumulative_rep = g_rep @ cumulative_rep
                Y[i, t, :] = cumulative_rep @ template

        if not return_all_outputs:
            Y = Y[:, -1, :]
        else:
            Y = Y[:, 1:, :]

        return cls(X, Y), sequence

    @classmethod
    def from_cn(
        cls,
        p: int,
        template: np.ndarray,
        k: int,
        mode: str = "sampled",
        num_samples: int = 65536,
        return_all_outputs: bool = False,
    ) -> tuple["OfflineModularCompositionDataset", np.ndarray]:
        """Build dataset for cyclic group C_p via 1D ``np.roll``.

        No escnn group object required.

        Parameters
        ----------
        p : int
            Order of the cyclic group.
        template : np.ndarray, shape (p,)
        k : int
            Sequence length (number of group elements to compose).
        mode : str
            ``"sampled"`` or ``"exhaustive"``.
        num_samples : int
            Number of samples when ``mode="sampled"``.
        return_all_outputs : bool
            If True, return intermediate composition outputs.

        Returns
        -------
        dataset : OfflineModularCompositionDataset
        sequence : np.ndarray, shape (N, k)
            Integer shifts per token.
        """
        assert template.shape == (p,), f"template must be ({p},), got {template.shape}"

        if mode == "exhaustive":
            total = p**k
            if total > 1_000_000:
                raise ValueError(f"p^k = {total} is huge; use mode='sampled' instead.")
            N = total
            sequence = np.zeros((N, k), dtype=np.int64)
            for idx in range(N):
                for t in range(k):
                    sequence[idx, t] = (idx // (p**t)) % p
        else:
            N = int(num_samples)
            sequence = np.random.randint(0, p, size=(N, k), dtype=np.int64)

        X = np.zeros((N, k, p), dtype=np.float32)
        Y = np.zeros((N, k, p), dtype=np.float32)

        for i in range(N):
            cumsum = 0
            for t in range(k):
                shift = int(sequence[i, t])
                X[i, t, :] = np.roll(template, shift)
                cumsum = (cumsum + shift) % p
                Y[i, t, :] = np.roll(template, cumsum)

        if not return_all_outputs:
            Y = Y[:, -1, :]
        else:
            Y = Y[:, 1:, :]

        return cls(X, Y), sequence

    @classmethod
    def from_cnxcn(
        cls,
        p1: int,
        p2: int,
        template: np.ndarray,
        k: int,
        mode: str = "sampled",
        num_samples: int = 65536,
        return_all_outputs: bool = False,
    ) -> tuple["OfflineModularCompositionDataset", np.ndarray]:
        r"""Build dataset for product group C_{p1} x C_{p2} via 2D ``np.roll``.

        No escnn group object required.

        Parameters
        ----------
        p1 : int
            Height (rows) dimension.
        p2 : int
            Width (cols) dimension.
        template : np.ndarray, shape (p1, p2)
            2D template image.
        k : int
            Sequence length (number of group elements to compose).
        mode : str
            ``"sampled"`` or ``"exhaustive"``.
        num_samples : int
            Number of samples when ``mode="sampled"``.
        return_all_outputs : bool
            If True, return intermediate composition outputs.

        Returns
        -------
        dataset : OfflineModularCompositionDataset
        sequence_xy : np.ndarray, shape (N, k, 2)
            Integer shifts (ax, ay) per token.
        """
        assert template.shape == (p1, p2), f"template must be ({p1}, {p2}), got {template.shape}"
        p_flat = p1 * p2

        if mode == "exhaustive":
            total = p_flat**k
            if total > 1_000_000:
                raise ValueError(f"(p1*p2)**k = {total} is huge; use mode='sampled' instead.")
            N = total
            sequence_xy = np.zeros((N, k, 2), dtype=np.int64)
            for idx in range(N):
                for t in range(k):
                    flat_idx = (idx // (p_flat**t)) % p_flat
                    sequence_xy[idx, t, 0] = flat_idx // p2
                    sequence_xy[idx, t, 1] = flat_idx % p2
        else:
            N = int(num_samples)
            sequence_xy = np.empty((N, k, 2), dtype=np.int64)
            sequence_xy[:, :, 0] = np.random.randint(0, p1, size=(N, k))
            sequence_xy[:, :, 1] = np.random.randint(0, p2, size=(N, k))

        X = np.zeros((N, k, p_flat), dtype=np.float32)
        Y = np.zeros((N, k, p_flat), dtype=np.float32)

        for i in range(N):
            sx, sy = 0, 0
            for t in range(k):
                ax = int(sequence_xy[i, t, 0])
                ay = int(sequence_xy[i, t, 1])
                rolled = np.roll(np.roll(template, shift=ax, axis=0), shift=ay, axis=1)
                X[i, t, :] = rolled.ravel()
                sx = (sx + ax) % p1
                sy = (sy + ay) % p2
                Y[i, t, :] = np.roll(np.roll(template, shift=sx, axis=0), shift=sy, axis=1).ravel()

        if not return_all_outputs:
            Y = Y[:, -1, :]

        return cls(X, Y), sequence_xy

import numpy as np
import torch
from torch.utils.data import IterableDataset


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

    @staticmethod
    def generate_dataset(
        p1: int,
        p2: int,
        template: np.ndarray,
        k: int,
        mode: str = "sampled",
        num_samples: int = 65536,
        return_all_outputs: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a fixed 2D modular addition dataset.

        Args:
            p1: height (rows) dimension
            p2: width  (cols) dimension
            template: (p1, p2) template array
            k: sequence length
            mode: "sampled" or "exhaustive"
            num_samples: number of samples for "sampled" mode
            return_all_outputs: if True, return intermediate outputs

        Returns:
            X:           (N, k, p1*p2) input sequences (flattened rolled templates)
            Y:           (N, p1*p2) or (N, k, p1*p2) targets
            sequence_xy: (N, k, 2) integer group elements (ax_t, ay_t) per token
        """
        assert template.shape == (p1, p2), f"template must be ({p1}, {p2}), got {template.shape}"
        p_flat = p1 * p2

        if mode == "exhaustive":
            total = (p1 * p2) ** k
            if total > 1_000_000:
                raise ValueError(f"(p1*p2)**k = {total} is huge; use mode='sampled' instead.")
            N = total
            sequence_xy = np.zeros((N, k, 2), dtype=np.int64)
            for idx in range(N):
                for t in range(k):
                    flat_idx = (idx // (p_flat**t)) % p_flat
                    ax = flat_idx // p2
                    ay = flat_idx % p2
                    sequence_xy[idx, t, 0] = ax
                    sequence_xy[idx, t, 1] = ay
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
                ax, ay = int(sequence_xy[i, t, 0]), int(sequence_xy[i, t, 1])
                rolled = np.roll(np.roll(template, shift=ax, axis=0), shift=ay, axis=1)
                X[i, t, :] = rolled.ravel()
                sx = (sx + ax) % p1
                sy = (sy + ay) % p2
                Y[i, t, :] = np.roll(np.roll(template, shift=sx, axis=0), shift=sy, axis=1).ravel()

        if not return_all_outputs:
            Y = Y[:, -1, :]

        return X, Y, sequence_xy


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

    @staticmethod
    def generate_dataset(
        p: int,
        template: np.ndarray,
        k: int,
        mode: str = "sampled",
        num_samples: int = 65536,
        return_all_outputs: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a fixed 1D modular addition dataset for cyclic group C_p.

        Args:
            p: dimension of cyclic group
            template: (p,) template array
            k: sequence length
            mode: "sampled" or "exhaustive"
            num_samples: number of samples for "sampled" mode
            return_all_outputs: if True, return intermediate outputs

        Returns:
            X: (N, k, p) where token t is template rolled by shift_t
            Y: (N, p) or (N, k-1, p) target rolled by cumulative sum
            sequence: (N, k) integer group elements (shifts) per token
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

        return X, Y, sequence


class OfflineModularCompositionDataset:
    """Offline dataset builder for group composition tasks.

    Generates all (or sampled) input-output pairs for composing group elements
    via the regular representation applied to a template.

    Three factory methods cover the supported group families:
      - group_dataset : any escnn group with a regular representation
      - cn_dataset    : cyclic group C_n (uses np.roll, no escnn needed)
      - cnxcn_dataset : product group C_n x C_n (uses 2D np.roll, no escnn needed)
    """

    @staticmethod
    def group_dataset(
        template: np.ndarray,
        k: int,
        group,
        mode: str = "sampled",
        num_samples: int = 65536,
        return_all_outputs: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build generic group composition dataset for sequence length k.

        Works with any escnn group that has a regular representation.
        For a sequence of k group elements (g1, g2, ..., gk), we compute:
        - X[i, t, :] = regular_rep(g_t) @ template
        - Y[i, :]    = regular_rep(g1 * g2 * ... * gk) @ template

        Parameters
        ----------
        template : np.ndarray, shape (group_order,)
            Template array.
        k : int
            Sequence length (number of group elements to compose).
        group : escnn group object
            Any group with a ``"regular"`` representation.
        mode : str
            ``"sampled"`` or ``"exhaustive"``.
        num_samples : int
            Number of samples when ``mode="sampled"``.
        return_all_outputs : bool
            If True, return intermediate composition outputs.

        Returns
        -------
        X : np.ndarray, shape (N, k, group_order)
        Y : np.ndarray, shape (N, group_order) or (N, k-1, group_order)
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
                raise ValueError(
                    f"n_elements^k = {total} is huge; use mode='sampled' instead."
                )
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

        return X, Y, sequence

    @staticmethod
    def cn_dataset(template):
        """Generate exhaustive k=2 dataset for cyclic group C_n.

        Uses ``np.roll`` (no escnn group object required).

        Parameters
        ----------
        template : np.ndarray, shape (n,)

        Returns
        -------
        X : np.ndarray, shape (n**2, 2, n)
        Y : np.ndarray, shape (n**2, n)
        """
        group_size = len(template)
        X = np.zeros((group_size * group_size, 2, group_size))
        Y = np.zeros((group_size * group_size, group_size))

        idx = 0
        for a in range(group_size):
            for b in range(group_size):
                q = (a + b) % group_size
                X[idx, 0, :] = np.roll(template, a)
                X[idx, 1, :] = np.roll(template, b)
                Y[idx, :] = np.roll(template, q)
                idx += 1

        return X, Y

    @staticmethod
    def cnxcn_dataset(template):
        r"""Generate exhaustive k=2 dataset for product group C_n x C_n.

        Uses 2D ``np.roll`` (no escnn group object required).

        Parameters
        ----------
        template : np.ndarray, shape (n, n)
            2D template image.

        Returns
        -------
        X : np.ndarray, shape (n**4, 2, n*n)
        Y : np.ndarray, shape (n**4, n*n)
        """
        image_length, _ = template.shape
        X = np.zeros((image_length**4, 2, image_length * image_length))
        Y = np.zeros((image_length**4, image_length * image_length))

        idx = 0
        for a_x in range(image_length):
            for a_y in range(image_length):
                for b_x in range(image_length):
                    for b_y in range(image_length):
                        q_x = (a_x + b_x) % image_length
                        q_y = (a_y + b_y) % image_length
                        X[idx, 0, :] = np.roll(
                            np.roll(template, a_x, axis=0), a_y, axis=1
                        ).flatten()
                        X[idx, 1, :] = np.roll(
                            np.roll(template, b_x, axis=0), b_y, axis=1
                        ).flatten()
                        Y[idx, :] = np.roll(
                            np.roll(template, q_x, axis=0), q_y, axis=1
                        ).flatten()
                        idx += 1

        return X, Y

    @staticmethod
    def to_device_and_flatten(X, Y, device=None):
        """Flatten X from (N, 2, d) to (N, 2*d), convert to tensors, move to device.

        Parameters
        ----------
        X : np.ndarray, shape (N, 2, d)
        Y : np.ndarray, shape (N, d)
        device : torch.device or str, optional
            If None, auto-detects CUDA availability.

        Returns
        -------
        X_tensor : torch.Tensor, shape (N, 2*d)
        Y_tensor : torch.Tensor, shape (N, d)
        device : torch.device
        """
        num_data_features = len(X[0][0])
        X_flat = X.reshape(X.shape[0], 2 * num_data_features)
        Y_flat = Y.reshape(Y.shape[0], num_data_features)
        X_tensor = torch.tensor(X_flat, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_flat, dtype=torch.float32)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("GPU is available. Using CUDA.")
            else:
                device = torch.device("cpu")
                print("GPU is not available. Using CPU.")

        X_tensor = X_tensor.to(device)
        Y_tensor = Y_tensor.to(device)

        return X_tensor, Y_tensor, device

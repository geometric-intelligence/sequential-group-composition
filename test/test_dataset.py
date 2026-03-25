"""Tests for src.dataset module."""

import numpy as np
import pytest
import torch

import src.dataset as dataset


class TestOnlineModularAdditionDataset1D:
    """Tests for OnlineModularAdditionDataset1D (online __iter__ only)."""

    def test_batch_shape(self):
        """Test that batches have correct shapes."""
        p = 7
        k = 3
        batch_size = 16
        template = np.random.randn(p).astype(np.float32)

        ds = dataset.OnlineModularAdditionDataset1D(
            p=p, template=template, k=k, batch_size=batch_size, device="cpu"
        )

        iterator = iter(ds)
        X, Y = next(iterator)

        assert X.shape == (batch_size, k, p), f"X shape mismatch: {X.shape}"
        assert Y.shape == (batch_size, p), f"Y shape mismatch: {Y.shape}"

    def test_batch_shape_return_all_outputs(self):
        """Test batch shapes with return_all_outputs=True."""
        p = 7
        k = 4
        batch_size = 16
        template = np.random.randn(p).astype(np.float32)

        ds = dataset.OnlineModularAdditionDataset1D(
            p=p,
            template=template,
            k=k,
            batch_size=batch_size,
            device="cpu",
            return_all_outputs=True,
        )

        iterator = iter(ds)
        X, Y = next(iterator)

        assert X.shape == (batch_size, k, p)
        assert Y.shape == (batch_size, k - 1, p)


class TestOnlineModularAdditionDataset2D:
    """Tests for OnlineModularAdditionDataset2D (online __iter__ only)."""

    def test_batch_shape(self):
        """Test that batches have correct shapes."""
        p1, p2 = 5, 5
        k = 3
        batch_size = 16
        template = np.random.randn(p1, p2).astype(np.float32)

        ds = dataset.OnlineModularAdditionDataset2D(
            p1=p1, p2=p2, template=template, k=k, batch_size=batch_size, device="cpu"
        )

        iterator = iter(ds)
        X, Y = next(iterator)

        p_flat = p1 * p2
        assert X.shape == (batch_size, k, p_flat), f"X shape mismatch: {X.shape}"
        assert Y.shape == (batch_size, p_flat), f"Y shape mismatch: {Y.shape}"

    def test_batch_shape_return_all_outputs(self):
        """Test batch shapes with return_all_outputs=True."""
        p1, p2 = 5, 5
        k = 4
        batch_size = 16
        template = np.random.randn(p1, p2).astype(np.float32)

        ds = dataset.OnlineModularAdditionDataset2D(
            p1=p1,
            p2=p2,
            template=template,
            k=k,
            batch_size=batch_size,
            device="cpu",
            return_all_outputs=True,
        )

        iterator = iter(ds)
        X, Y = next(iterator)

        p_flat = p1 * p2
        assert X.shape == (batch_size, k, p_flat)
        assert Y.shape == (batch_size, k - 1, p_flat)


class TestOfflineModularCompositionDataset:
    """Tests for OfflineModularCompositionDataset."""

    @pytest.fixture
    def d3_group(self):
        """Create a DihedralGroup(N=3) for testing."""
        from escnn.group import DihedralGroup

        return DihedralGroup(N=3)

    @pytest.fixture
    def template_d3(self, d3_group):
        """Create a template for D3 group (order 6)."""
        group_order = d3_group.order()
        template = np.random.randn(group_order).astype(np.float32)
        return template

    def test_from_group_sampled(self, template_d3, d3_group):
        """Test from_group output shapes in sampled mode."""
        k = 3
        num_samples = 100
        group_order = len(template_d3)

        ds, sequence = dataset.OfflineModularCompositionDataset.from_group(
            template=template_d3, k=k, group=d3_group, mode="sampled", num_samples=num_samples
        )

        assert len(ds) == num_samples
        assert ds.X.shape == (num_samples, k, group_order), f"X shape mismatch: {ds.X.shape}"
        assert ds.Y.shape == (num_samples, group_order), f"Y shape mismatch: {ds.Y.shape}"
        assert sequence.shape == (num_samples, k), f"sequence shape mismatch: {sequence.shape}"

    def test_from_group_exhaustive(self, template_d3, d3_group):
        """Test from_group output shapes in exhaustive mode."""
        k = 2
        group_order = len(template_d3)
        n_elements = group_order

        ds, sequence = dataset.OfflineModularCompositionDataset.from_group(
            template=template_d3, k=k, group=d3_group, mode="exhaustive"
        )

        expected_n = n_elements**k
        assert len(ds) == expected_n
        assert ds.X.shape == (expected_n, k, group_order)
        assert ds.Y.shape == (expected_n, group_order)
        assert sequence.shape == (expected_n, k)

    def test_from_group_return_all_outputs(self, template_d3, d3_group):
        """Test from_group output shapes with return_all_outputs=True."""
        k = 4
        num_samples = 50
        group_order = len(template_d3)

        ds, sequence = dataset.OfflineModularCompositionDataset.from_group(
            template=template_d3,
            k=k,
            group=d3_group,
            mode="sampled",
            num_samples=num_samples,
            return_all_outputs=True,
        )

        assert len(ds) == num_samples
        assert ds.X.shape == (num_samples, k, group_order)
        assert ds.Y.shape == (num_samples, k - 1, group_order)
        assert sequence.shape == (num_samples, k)

    def test_from_group_template_mismatch_error(self, d3_group):
        """Test that mismatched template length raises error."""
        wrong_size = d3_group.order() + 1
        template = np.random.randn(wrong_size).astype(np.float32)

        with pytest.raises(AssertionError):
            dataset.OfflineModularCompositionDataset.from_group(
                template, k=2, group=d3_group, mode="exhaustive",
            )

    def test_from_group_getitem(self, template_d3, d3_group):
        """Test that __getitem__ returns correct (X_i, Y_i) pair."""
        ds, _ = dataset.OfflineModularCompositionDataset.from_group(
            template=template_d3, k=2, group=d3_group, mode="exhaustive"
        )

        x_i, y_i = ds[0]
        assert x_i.shape == ds.X.shape[1:]
        assert y_i.shape == ds.Y.shape[1:]
        torch.testing.assert_close(x_i, ds.X[0])
        torch.testing.assert_close(y_i, ds.Y[0])

    def test_from_cn_sampled(self):
        """Test from_cn output shapes in sampled mode."""
        p = 7
        template = np.random.randn(p).astype(np.float32)
        k = 3
        num_samples = 100

        ds, sequence = dataset.OfflineModularCompositionDataset.from_cn(
            p=p, template=template, k=k, mode="sampled", num_samples=num_samples
        )

        assert len(ds) == num_samples
        assert ds.X.shape == (num_samples, k, p), f"X shape mismatch: {ds.X.shape}"
        assert ds.Y.shape == (num_samples, p), f"Y shape mismatch: {ds.Y.shape}"
        assert sequence.shape == (num_samples, k)

    def test_from_cn_exhaustive(self):
        """Test from_cn output shapes in exhaustive mode."""
        p = 7
        template = np.random.randn(p).astype(np.float32)
        k = 2

        ds, sequence = dataset.OfflineModularCompositionDataset.from_cn(
            p=p, template=template, k=k, mode="exhaustive"
        )

        expected_n = p**k
        assert len(ds) == expected_n
        assert ds.X.shape == (expected_n, k, p)
        assert ds.Y.shape == (expected_n, p)
        assert sequence.shape == (expected_n, k)

    def test_from_cn_return_all_outputs(self):
        """Test from_cn output shapes with return_all_outputs=True."""
        p = 7
        template = np.random.randn(p).astype(np.float32)
        k = 4
        num_samples = 50

        ds, sequence = dataset.OfflineModularCompositionDataset.from_cn(
            p=p, template=template, k=k, mode="sampled",
            num_samples=num_samples, return_all_outputs=True,
        )

        assert len(ds) == num_samples
        assert ds.X.shape == (num_samples, k, p)
        assert ds.Y.shape == (num_samples, k - 1, p)
        assert sequence.shape == (num_samples, k)

    def test_from_cn_rolling_correctness(self):
        """Test that from_cn X values are rolled versions of template."""
        p = 7
        template = np.random.randn(p).astype(np.float32)
        k = 2

        ds, sequence = dataset.OfflineModularCompositionDataset.from_cn(
            p=p, template=template, k=k, mode="exhaustive"
        )

        shift_0 = int(sequence[0, 0])
        expected_x0 = np.roll(template, shift_0)
        np.testing.assert_allclose(ds.X[0, 0, :].numpy(), expected_x0, rtol=1e-5)

    def test_from_cnxcn_sampled(self):
        """Test from_cnxcn output shapes in sampled mode."""
        p1, p2 = 5, 5
        template = np.random.randn(p1, p2).astype(np.float32)
        k = 3
        num_samples = 100

        ds, sequence_xy = dataset.OfflineModularCompositionDataset.from_cnxcn(
            p1=p1, p2=p2, template=template, k=k, mode="sampled", num_samples=num_samples
        )

        p_flat = p1 * p2
        assert len(ds) == num_samples
        assert ds.X.shape == (num_samples, k, p_flat), f"X shape mismatch: {ds.X.shape}"
        assert ds.Y.shape == (num_samples, p_flat), f"Y shape mismatch: {ds.Y.shape}"
        assert sequence_xy.shape == (num_samples, k, 2)

    def test_from_cnxcn_exhaustive(self):
        """Test from_cnxcn output shapes in exhaustive mode."""
        p1, p2 = 3, 3
        template = np.random.randn(p1, p2).astype(np.float32)
        k = 2

        ds, sequence_xy = dataset.OfflineModularCompositionDataset.from_cnxcn(
            p1=p1, p2=p2, template=template, k=k, mode="exhaustive"
        )

        expected_n = (p1 * p2) ** k
        p_flat = p1 * p2
        assert len(ds) == expected_n
        assert ds.X.shape == (expected_n, k, p_flat)
        assert ds.Y.shape == (expected_n, p_flat)
        assert sequence_xy.shape == (expected_n, k, 2)

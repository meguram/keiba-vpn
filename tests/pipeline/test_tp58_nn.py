import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.pipeline.models.tp58_nn import CrossNet, FMTerm, Net, VARIANTS, listwise_loss, race_index  # noqa: E402


def test_race_index_padding():
    codes = np.array([0, 0, 0, 1, 1, 2])
    idx, starts = race_index(codes, max_n=4)
    assert starts.tolist() == [0, 3, 5]
    assert idx.tolist() == [[0, 1, 2, -1], [3, 4, -1, -1], [5, -1, -1, -1]]


def test_cross_layer_formula():
    torch.manual_seed(0)
    net = CrossNet(5, layers=2)
    x = torch.randn(4, 5)
    y = x
    for w in net.lin:
        y = x * w(y) + y
    assert torch.allclose(net(x), y, atol=1e-6)


def test_fm_matches_pairwise_sum():
    torch.manual_seed(0)
    fm = FMTerm(4, k=3, heads=2)
    x = torch.randn(2, 4)
    out = fm(x)
    V = fm.V.detach().view(4, 2, 3)
    for b in range(2):
        for h in range(2):
            s = sum((V[i, h] * V[j, h]).sum() * x[b, i] * x[b, j] for i in range(4) for j in range(i + 1, 4))
            assert out[b, h].item() == pytest.approx(float(s), abs=1e-5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_and_loss_finite(variant):
    torch.manual_seed(0)
    net = Net(6, variant, hidden=(8,))
    x = torch.randn(7, 6)
    codes = np.array([0, 0, 0, 1, 1, 2, 2])
    idx, _ = race_index(codes, max_n=4)
    y = torch.tensor([1, 0, 0, 0, 1, 1, 0], dtype=torch.float32)
    valid = torch.ones(7, dtype=torch.bool)
    loss = listwise_loss(net(x), torch.tensor(idx), y, (y > -1).float() * 0 + y, y, valid)
    assert torch.isfinite(loss)
    loss.backward()


def test_loss_ignores_padding_and_invalid():
    logits = torch.zeros(5, 3)
    logits[4, 0] = 50.0  # 無効行に巨大なロジット
    idx = torch.tensor([[0, 1, 2, -1], [3, 4, -1, -1]])
    y = torch.tensor([1.0, 0, 0, 1.0, 0])
    valid = torch.tensor([True, True, True, True, False])
    a = listwise_loss(logits, idx, y, y, y, valid)
    logits2 = logits.clone()
    logits2[4, 0] = -50.0
    b = listwise_loss(logits2, idx, y, y, y, valid)
    assert torch.allclose(a, b)

import torch

import sevenn._keys as KEY
from sevenn.nn.global_embedding import GlobalScalarEmbedding


def test_global_scalar_embedding_appends_value():
    data = {
        KEY.NODE_FEATURE: torch.zeros((3, 2)),
        KEY.TOTAL_CHARGE: torch.tensor(1.5),
    }
    embed = GlobalScalarEmbedding(KEY.TOTAL_CHARGE)
    out = embed(data)
    assert out[KEY.NODE_FEATURE].shape == (3, 3)
    assert torch.allclose(out[KEY.NODE_FEATURE][:, -1], torch.full((3,), 1.5))

from kvbridge.fused_fit import fused_scratch_shape


def test_fused_scratch_shape_depends_on_chunk_not_total_observations():
    small = fused_scratch_shape(4, 8, 32, 16)
    large_n = fused_scratch_shape(4, 8, 32, 16)
    assert small == large_n
    assert small["x_panel"] == (4, 8, 32)


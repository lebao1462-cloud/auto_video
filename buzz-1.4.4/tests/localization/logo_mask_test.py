import pytest

from buzz.localization.final_render import FinalRenderOptions, build_bilibili_watermark_filter_graph
from buzz.localization.logo_mask import LogoMaskConfig, LogoMaskRegion


def test_custom_mask_graph_has_stable_per_region_labels():
    config = LogoMaskConfig(mode="custom", regions=(
        LogoMaskRegion("top-left", "Top left", True, .01, .02, .2, .1, 24),
        LogoMaskRegion("bottom-right", "Bottom right", True, .79, .85, .2, .1, 12),
    ))
    graph = build_bilibili_watermark_filter_graph(FinalRenderOptions(logo_mask_config=config))
    assert "[mask_0_base]" in graph and "[mask_1_base]" in graph
    assert "luma_radius=max(0\\,min(24\\,(min(w\\,h)-2)/2))" in graph
    assert "luma_radius=max(0\\,min(12\\,(min(w\\,h)-2)/2))" in graph
    assert "chroma_radius=max(0\\,min(24\\,(min(w\\,h)-2)/4))" in graph
    assert graph.endswith("[vout]")



def test_custom_empty_config_never_falls_back_to_legacy_region():
    graph = build_bilibili_watermark_filter_graph(
        FinalRenderOptions(logo_mask_config=LogoMaskConfig(mode="custom"))
    )
    assert graph == "[0:v]null[vout]"


def test_mask_json_round_trip_and_invalid_regions_are_rejected():
    config = LogoMaskConfig(mode="custom", regions=(
        LogoMaskRegion("one", "One", True, .1, .1, .2, .2, 20),
    ))
    assert LogoMaskConfig.from_json(config.to_json()) == config
    with pytest.raises(ValueError):
        LogoMaskRegion("bad", "Bad", True, .9, .9, .2, .2, 20)
    with pytest.raises(ValueError):
        LogoMaskConfig(mode="custom", regions=tuple(
            LogoMaskRegion(str(index), str(index), True, .1, .1, .1, .1, 10)
            for index in range(17)
        ))

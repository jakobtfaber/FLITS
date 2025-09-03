from config_utils import load_telescope_block, clear_config_cache

def test_load_telescope_block_and_cache(tmp_path):
    cfg = tmp_path / "telescopes.yaml"
    cfg.write_text(
        "telescopes:\n"
        "  testscope:\n"
        "    df_MHz_raw: 1\n"
        "    dt_ms_raw: 2\n"
        "    f_min_GHz: 3\n"
        "    f_max_GHz: 4\n"
    )
    name, params = load_telescope_block(cfg, "testscope")
    assert name == "testscope"
    assert params == {"df_MHz_raw": 1.0, "dt_ms_raw": 2.0, "f_min_GHz": 3.0, "f_max_GHz": 4.0}
    cfg.write_text(
        "telescopes:\n"
        "  testscope:\n"
        "    df_MHz_raw: 10\n"
        "    dt_ms_raw: 20\n"
        "    f_min_GHz: 30\n"
        "    f_max_GHz: 40\n"
    )
    name2, params2 = load_telescope_block(cfg, "testscope")
    assert params2["df_MHz_raw"] == 1.0
    clear_config_cache()
    name3, params3 = load_telescope_block(cfg, "testscope")
    assert params3["df_MHz_raw"] == 10.0

from genesis.config import WAN_MODELS


def test_every_model_has_a_train_run():
    for spec in WAN_MODELS.values():
        assert spec.train_runs
        assert all(run.dit_pattern in spec.dit_patterns for run in spec.train_runs)


def test_resolutions_fit_the_vae():
    for spec in WAN_MODELS.values():
        assert spec.height % 16 == 0 and spec.width % 16 == 0


def test_moe_boundaries_cover_all_timesteps():
    runs = sorted(WAN_MODELS["14B"].train_runs, key=lambda run: run.min_timestep_boundary)
    assert runs[0].min_timestep_boundary == 0.0
    assert runs[-1].max_timestep_boundary == 1.0
    assert runs[0].max_timestep_boundary == runs[1].min_timestep_boundary

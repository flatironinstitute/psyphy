def test_top_level_api_imports():
    import psyphy as p

    for name in [
        "WPPM",
        "Prior",
        "OddityTask",
        "GaussianNoise",
        "StudentTNoise",
        "MAPOptimizer",
        "LangevinSampler",
        "LaplaceApproximation",
        "ParameterPosterior",
        "MAPPosterior",
        "ResponseData",
        "TrialBatch",
        "TrialData",
    ]:
        assert hasattr(p, name)

from ashare_alpha.model_v31 import RIDGE_PARAMETERS_V31


def test_v31_ridge_parameters_are_frozen() -> None:
    assert RIDGE_PARAMETERS_V31 == {
        "alpha": 100.0,
        "solver": "lsqr",
        "tol": 0.000001,
        "max_iter": 5000,
        "fit_intercept": True,
    }

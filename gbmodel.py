params_xgb = {
    "n_estimators": 599,
    "learning_rate": 0.02380041317545529,
    "max_depth": 10,
    "subsample": 0.6350637158086113,
    "colsample_bytree": 0.6824975745925943,
    "min_child_weight": 8.245882345417543,
    "reg_lambda": 2.9622841183300506,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

params_lgb = {
    "n_estimators": 315,
    "learning_rate": 0.01873555586505047,
    "num_leaves": 175,
    "max_depth": 6,
    "min_child_samples": 88,
    "subsample": 0.7549453750367137,
    "colsample_bytree": 0.6204629380130997,
    "reg_lambda": 9.968337723623222,
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
}

params_cb = {
    "iterations": 950,
    "learning_rate": 0.023201843436078066,
    "depth": 9,
    "l2_leaf_reg": 9.377457478846663,
    "bagging_temperature": 1.127113018698872,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "verbose": False,
    "allow_writing_files": False,
}
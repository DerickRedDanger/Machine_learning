results = []

for experiment in EXPERIMENTS:
    result = run_experiment(
        df=train_df,
        target_col="Survived",
        experiment_config=experiment,
        model_registry=MODEL_REGISTRY,
        logger=logger,
    )
    results.append(result)

# convert to DataFrame
# print summary
# plot results
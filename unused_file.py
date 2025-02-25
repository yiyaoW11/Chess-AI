# def process_data():
#     df = pd.read_csv("game_data.csv", usecols=["lines", "result"], dtype={"lines": "str", "result": "category"})
    
#     # Convert to NumPy for vectorized processing
#     lines_array = df["lines"].values  
#     result_array = df["result"].values  

#     # Parallel processing using multiprocessing Pool
#     with mp.Pool(mp.cpu_count()) as pool:
#         results = pool.starmap(process_row, zip(lines_array, result_array))

#     # Vectorized extraction of results using NumPy
#     all_positions, all_evaluations, all_results = map(np.concatenate, zip(*results))
    
#     return all_positions, all_evaluations, all_results


# def process_data():
#     all_positions = []
#     all_evaluations = []
#     all_results = []

#     for chunk in pd.read_csv("game_data.csv", nrows=100, chunksize=20):
#         for _, row in chunk.iterrows():
#             game = create_pgn_game(str(row["lines"]))
#             # print(str(row["lines"]))
#             # print("just printed the row of moves")
#             curr_positions, curr_evaluations, curr_results = extract_game_info(game, row["result"])

#             # print(curr_positions)
#             # print("helllooooo")
#             # break

#             all_positions.extend(curr_positions)
#             all_evaluations.extend(curr_evaluations)
#             all_results.extend(curr_results)

    
#     # engine.quit()
#     return all_positions, all_evaluations, all_results


    # # Define model pipeline with scaling
    # model = make_pipeline(
    #     StandardScaler(),
    #     RandomForestRegressor(random_state=50)
    # )

    # # Hyperparameter tuning using GridSearchCV
    # param_grid = {
    #     'randomforestregressor__n_estimators': [100, 200, 300],
    #     'randomforestregressor__max_depth': [5, 10, 15, None],
    #     'randomforestregressor__min_samples_split': [2, 5, 10],
    # }

    # grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    # grid_search.fit(x_train, y_train)

    # # Best model
    # best_model = grid_search.best_estimator_

    # # Make predictions with the best model
    # y_predictions = best_model.predict(x_test)

    # # Evaluate the model
    # mse = mean_squared_error(y_test, y_predictions)
    # r2 = r2_score(y_test, y_predictions)

    # print(f"Best Model Parameters: {grid_search.best_params_}")
    # print(f"MSE: {mse}")
    # print(f"R2: {r2}")

    # # Cross-validation score (for the entire dataset)
    # cv_score = cross_val_score(best_model, x, y, cv=5, scoring='neg_mean_squared_error')
    # print(f"Cross-Validation MSE: {-cv_score.mean()}")

#     all_positions, all_evaluations, all_results = process_data()

#     x = [extract_features(position) for position in all_positions]
#     y = all_evaluations

#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

#     # # Fitting model
#     # rf_model = RandomForestRegressor(n_estimators=100, random_state=50)
#     # rf_model.fit(x_train, y_train)

#     # y_predictions = rf_model.predict(x_test)

#     # Create a pipeline with scaling and the model
#     model = make_pipeline(
#         StandardScaler(),
#         RandomForestRegressor(n_estimators=100, max_depth=10, random_state=50)
#     )
#     model.fit(x_train, y_train)
#     y_predictions = model.predict(x_test)

#     mse = mean_squared_error(y_test, y_predictions)
#     # accuracy = accuracy_score(y_test, y_predictions)
#     cv_scores = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')

# # Convert negative MSE scores to positive values
#     cv_scores = -cv_scores

# # Print the cross-validation results
#     print(f"Cross-validated MSE: {cv_scores}")
#     print(f"MSE: {mse}")

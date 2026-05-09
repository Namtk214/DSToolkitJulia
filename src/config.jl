# Central training and demo configuration for DSToolkit.

const TRAINING_CONFIG = (
    split = (
        train_ratio = 0.8,
    ),

    model_defaults = (
        ridge_reg = (lambda = 1.0,),
        lasso_reg = (lambda = 1.0,),
        elastic_net_reg = (lambda = 1.0, alpha = 0.5),
        decision_tree_reg = (max_depth = -1,),
        random_forest_reg = (n_trees = 100,),
        xgboost_reg = (num_round = 100, max_depth = 6),
        knn_reg = (K = 5,),

        decision_tree_cls = (max_depth = -1,),
        random_forest_cls = (n_trees = 100,),
        adaboost_cls = (n_iter = 10,),
        xgboost_cls = (num_round = 100, max_depth = 6),
        knn_cls = (K = 5,),

        arima = (order = (1, 1, 1),),
        deep_ts = (
            hidden_dim = 32,
            epochs = 50,
            seq_len = 12,
            learning_rate = 0.01,
        ),
    ),

    demo = (
        random_forest = (n_trees = 50,),
        adaboost = (n_iter = 25,),
        xgboost = (num_round = 50, max_depth = 4),
        ridge = (lambda = 1.0,),
        lasso = (lambda = 0.1,),
        elastic_net = (lambda = 0.1, alpha = 0.5),
        knn = (K = 5,),
        arima = (order = (1, 1, 1),),
        deep_ts = (
            input_dim = 1,
            hidden_dim = 16,
            epochs = 30,
            seq_len = 12,
        ),
    ),
)

training_config(section::Symbol) = getproperty(TRAINING_CONFIG, section)
training_config(section::Symbol, key::Symbol) = getproperty(training_config(section), key)

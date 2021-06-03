#!/usr/bin/env python


from utils import *
import sys


params_lgb = {
        'bagging_freq': 1,
        'bagging_fraction': 1,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.5,
        'learning_rate': 0.1,
        'metric':'rmse',
        #'max_depth': 3,
        'min_data_in_leaf': 15,
        'tree_learner': 'serial',
        'objective': 'regression',
        'verbosity': -1
    }


def train_pipeline(init_sales_df, customers_df, output_model, train_size, use_norm_price=False):
	sales_df, features, price_dict = make_features(init_sales_df, customers_df, use_norm_price=use_norm_price)

	params_lgb['monotone_constraints'] =  [-1 if col == 'price' else 0 for col in features]

	df_train, df_valid, y_train, y_valid, id_train, id_valid = split_train_valid(sales_df, features, train_size=train_size)
	print(f'Training model on {train_size} of data...\n')
	lgb_model = train_model(df_train, df_valid, y_train, y_valid, params_lgb, features)
	print("")
	score_model(lgb_model, sales_df, id_valid, features)
	print("\n\n")

	print("Training model on all data...")
	lgb_model_all = train_all_data_model(sales_df[features], sales_df['volume'], params_lgb, features, ntree=int(lgb_model.best_iteration * 1.5))
	lgb_model_all.save_model(output_model)
	print(f'Model is saved to {output_model}')

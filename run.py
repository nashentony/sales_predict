#!/usr/bin/env python
from utils import *
from train import *
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train, predict or tune_price", required=True)
parser.add_argument("--sales", help="path to sales file", required=True)
parser.add_argument("--customers", help="path to customers file", required=True)
parser.add_argument("--train_size", help="size of train [0-0.9]", type=float)
parser.add_argument("--output_model", help="path to output model")
parser.add_argument("--input_model", help="path to input model")
parser.add_argument("--n_month", help="number of next month to predict", type=int)
parser.add_argument("--report", help="path to report file")
parser.add_argument("--start_r", help="start of range for price tune", type=int)
parser.add_argument("--stop_r", help="stop of range for price tune", type=int)

args = parser.parse_args()


init_sales_df, customers_df = read_data(args.sales, args.customers)

use_norm_price=False

if args.mode == "train":
	train_pipeline(init_sales_df, customers_df, args.output_model, args.train_size, use_norm_price=use_norm_price)
	if args.n_month > 0:
		print("")
		print(f'Predict for next {args.n_month} month')
		predict_n_months(init_sales_df, customers_df, args.n_month, args.output_model, args.report, use_norm_price=use_norm_price)

elif args.mode == "predict":
	if args.n_month > 0:
		print(f'Predict for next {args.n_month} month')
		predict_n_months(init_sales_df, customers_df, args.n_month, args.input_model, args.report, use_norm_price=use_norm_price)

elif args.mode == "tune_price":
	if args.n_month > 0:
		print(f'Predict for next {args.n_month} month')
		predict_n_months(init_sales_df, customers_df, args.n_month, args.input_model, use_norm_price=use_norm_price, check_tune=True, start_r=args.start_r, stop_r=args.stop_r)

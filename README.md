
##Training an predict
```
python3.6 run.py --mode=train --sales=data/sales.tsv --customers=data/customers.tsv --train_size=0.65 --output_model=models/5.model --n_month=3
```

##Only predict
```
python3.6 run.py --mode=predict --sales=data/sales.tsv --customers=data/customers.tsv --input_model=models/5.model --n_month=3 --report=report.csv
```

##Predict sales changing depending on price
```
python3.6 run.py --mode=tune_price --sales=data/sales.tsv --customers=data/customers.tsv --input_model=models/5.model --n_month=3 --start_r=-15 --stop_r=16
```

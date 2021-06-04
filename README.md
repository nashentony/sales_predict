
# Python pipeline
### Training an predict
```
python run.py --mode=train --sales=data/sales.tsv --customers=data/customers.tsv --train_size=0.65 --output_model=models/5.model --n_month=3
```

### Only predict
```
python run.py --mode=predict --sales=data/sales.tsv --customers=data/customers.tsv --input_model=models/5.model --n_month=3 --report=report.csv
```

### Predict sales changing depending on price
```
python run.py --mode=tune_price --sales=data/sales.tsv --customers=data/customers.tsv --input_model=models/5.model --n_month=3 --start_r=-15 --stop_r=16
```

# Docker pipeline

### To make image 
```
sudo docker build --tag predict_sales .
```

### Training an predict
```

 sudo docker run -v $dir/:$dir/  -i --rm predict_sales --mode=train --sales=$dir/data/sales.tsv --customers=$dir/data/customers.tsv --train_size=0.65 --output_model=$dir/models/5.model --n_month=3
```

### Only predict
```
 sudo docker run -v $dir/:$dir/  -i --rm predict_sales --mode=predict --sales=$dir/data/sales.tsv --customers=$dir/data/customers.tsv --input_model=$dir/models/5.model --n_month=3 --report=$dir/report.csv
```

### Predict sales changing depending on price
```
 sudo docker run -v $dir/:$dir/  -i --rm predict_sales --mode=tune_price --sales=$dir/data/sales.tsv --customers=$dir/data/customers.tsv --input_model=$dir/models/5.model --n_month=3 --start_r=-15 --stop_r=16
```


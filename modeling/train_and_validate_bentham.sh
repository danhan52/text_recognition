# train model for one epoch
python run_model.py train BenthamDataset 1 16 False 0 $1 new
# prediction
python run_model.py pred BenthamDataset 1 16 False 0 $1 new $1 0
python end_batch.py 0

for i in {1000..4000..1000}
  do
	# train model for one epoch
	python run_model.py train BenthamDataset 1 16 False $i $1 new $1 $(($i-1000))
	# prediction
	python run_model.py pred BenthamTest 1 16 False $i $1 new $1 $i

	# delete old files
	python end_batch.py $i $1 2000
  done
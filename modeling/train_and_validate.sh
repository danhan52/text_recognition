if [[ $# -ne 4 ]]; then
	echo "Arguments are train_location, test_location, model_location, epochs"
	exit 1
fi

# train model for one epoch
python run_model.py train $1 1 16 True 0 $3 new
# prediction
python run_model.py pred $2 1 16 True 0 $3 new $3 0
python end_batch.py 0

for i in $(seq 1 $(($4-1)))
  do
	# train model for one epoch
	python run_model.py train $1 1 16 True $i $3 new $3 $(($i-1))
	# prediction
	python run_model.py pred $2 1 16 True $i $3 new $3 $i

	# delete old files
	python end_batch.py $i $3 2
  done

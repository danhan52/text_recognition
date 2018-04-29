# train model for one epoch
python run_model.py train combined_train 10 16 True 0 $1 new
# prediction
python run_model.py pred combined_test 1 16 True 1000 $1 new $1 0
# python end_batch.py 0

# for i in {1..9..1}
#   do
# 	# train model for one epoch
# 	python run_model.py train combined_train 1 16 True $i $1 new $1 $(($i-1))
# 	# prediction
# 	python run_model.py pred combined_test 1 16 True $i $1 new $1 $i

# 	# delete old files
# 	python end_batch.py $i $1 2
#   done
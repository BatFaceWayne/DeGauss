python train_change_prob.py -s /media/ray/data_volume/data_baselines/dynerf/coffee_martini --port 6452 --expname coffee_martini_sparse  --configs arguments/hypernerf/hyper_2.py
wait
python calc_metric.py -m sear_steak_sparse
wait
python train_change_prob.py -s /media/ray/data_volume/data_baselines/dynerf/flame_salmon --port 6469 --expname flame_salmon_sparse --configs arguments/hypernerf/hyper_2.py
wait
python calc_metric.py -m flame_salmon_sparse
wait
python train_change_prob.py -s /media/ray/data_volume/data_baselines/dynerf/cook_spinach --port 6468 --expname cook_spinach_sparse  --configs arguments/hypernerf/hyper_2.py
wait
python calc_metric.py -m cook_spinach_sparse
wait
python train_change_prob.py -s /media/ray/data_volume/data_baselines/dynerf/cut_roasted_beef --port 6450 --expname cut_roasted_beef_sparse  --configs arguments/hypernerf/hyper_2.py
wait
python calc_metric.py -m cut_roasted_beef_sparse
wait
python train_change_prob.py -s /media/ray/data_volume/data_baselines/dynerf/flame_steak --port 6451 --expname flame_steak_sparse  --configs arguments/hypernerf/hyper_2.py
wait
python calc_metric.py -m flame_steak_sparse
wait
python train_change_prob.py -s /media/ray/data_volume/data_baselines/dynerf/sear_steak --port 6452 --expname sear_steak_sparse  --configs arguments/hypernerf/hyper_2.py
wait
python calc_metric.py -m sear_steak_sparse
wait
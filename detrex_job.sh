idx=$1
srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=10 -p A100-IML --mem=40000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/detrex_05:/home/iqbal/detrex_05 \
  --container-image=/netscratch/naeem/detrex-torch1.13-detectron2-sourced.sqsh \
  --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=aug_${idx} \
  --container-workdir=/home/iqbal/detrex_05 \
  --time=00-04:00 \
  bash train_aug.sh $idx

# srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=20 -p A100-IML --mem=60000 \
#   --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/detrex_05:/home/iqbal/detrex \
#   --container-image=/netscratch/naeem/detrex-torch1.13-detectron2-sourced.sqsh \
#   --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=def_detr_pheno \
#   --container-workdir=/home/iqbal/detrex \
#   --time=00-02:00 \
#   bash train_detrex.sh
srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=20 -p H100 --mem=60000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/detrex_orig:/home/iqbal/detrex \
  --container-image=/netscratch/naeem/detrex-torch1.13-detectron2-sourced.sqsh \
  --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=def_detr_pheno \
  --container-workdir=/home/iqbal/detrex \
  --time=00-01:00 \
  bash train_detrex.sh

srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=20 -p A100-IML --mem=40000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/detrex_orig:/home/iqbal/detrex \
  --container-image=/netscratch/naeem/detrex-torch1.13-detectron2-sourced.sqsh \
  --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=pheno_syn_v6_flis \
  --container-workdir=/home/iqbal/detrex \
  --time=02-00:00 \
  --pty /bin/bash
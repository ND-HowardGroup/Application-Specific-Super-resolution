#!/bin/csh

#$ -M vmannam@nd.edu	 # Email address for job notification
#$ -m abe		 # Send mail when job begins, ends and aborts
#$ -q gpu		 # Specify queue
#$ -l gpu_card=1
#$ -N gans_DEd_config63        # Specify job name

#$ -pe smp 4                # Specify parallel environment and legal core size
setenv OMP_NUM_THREADS 4	         # Required modules

module load pytorch/1.1.0
pip install "scikit_image==0.16.2" --user
nvidia-smi
python main_config63.py


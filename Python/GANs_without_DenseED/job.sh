#!/bin/csh

#$ -M vmannam@nd.edu	 # Email address for job notification
#$ -m abe		 # Send mail when job begins, ends and aborts
#$ -q gpu		 # Specify queue
#$ -l gpu_card=1
#$ -N gans_config9433         # Specify job name

#$ -pe smp 4                # Specify parallel environment and legal core size
setenv OMP_NUM_THREADS 4	         # Required modules

module load pytorch/1.1.0
nvidia-smi
python main_config9433.py


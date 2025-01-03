#!/bin/bash

# list of config files
CONFIG_FILES=(
              # "gaot/abla/multiscale/heat_l_sines.json"
              # "gaot/abla/multiscale/wave_c_sines.json"
              # "gaot/airfoil/airfoil_li/geoemb/lano.json"
              # "gaot/airfoil/airfoil_li_large/geoemb/lano.json"
              "gaot/airfoil/airfoil_li/gaot.json"
              "gaot/airfoil/airfoil_li_large/gaot.json"
              # "gaot/abla/data_size/wave_c_sines/128.json"
              # "gaot/abla/data_size/wave_c_sines/256.json"
              # "gaot/abla/data_size/wave_c_sines/512.json"
              # "gaot/abla/data_size/wave_gauss/128.json"
              # "gaot/abla/data_size/wave_gauss/256.json"
              # "gaot/abla/data_size/wave_gauss/512.json"
              # "gaot/fd/finetune/wave_c_sines/fd_128.json"
              # "gaot/fd/finetune/wave_c_sines/fd_256.json"
              # "gaot/fd/finetune/wave_c_sines/fd_512.json"
              # "gaot/fd/finetune/wave_c_sines/fd_1024.json"
              # "gaot/fd/finetune/wave_gauss/fd_128.json"
              # "gaot/fd/finetune/wave_gauss/fd_256.json"
              # "gaot/fd/finetune/wave_gauss/fd_512.json"
              # "gaot/fd/finetune/wave_gauss/fd_1024.json"
              )
for CONFIG_FILE in "${CONFIG_FILES[@]}"
do
  
  JOB_NAME=$(basename "$CONFIG_FILE" .json)
  JOB_NAME=${JOB_NAME#config_}  
  echo "Submitting job: $JOB_NAME with config file: $CONFIG_FILE"

  # submit the job
  sbatch --job-name=$JOB_NAME \
         --output=${CONFIG_FILE%.json}_%j.out \
         --error=${CONFIG_FILE%.json}_%j.err \
         --export=CONFIG_FILE=$CONFIG_FILE \
         submit_job.sh
done

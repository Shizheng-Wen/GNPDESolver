#!/bin/bash

# list of config files
CONFIG_FILES=(
              # "gaot/static/poisson_c_sines.json"
              # "gaot/abla/data_size/poisson_c_sines/256.json"
              # "gaot/abla/data_size/poisson_c_sines/512.json"
              # "gaot/abla/data_size/poisson_c_sines/1024.json"

              # "gaot/abla/data_size/airfoil_grid/128.json"
              # "gaot/abla/data_size/airfoil_grid/256.json"
              # "gaot/abla/data_size/airfoil_grid/512.json"
              # "gaot/abla/data_size/airfoil_grid/1024.json"

              # "gaot/abla/data_size/poisson_gauss/128.json"
              # "gaot/abla/data_size/poisson_gauss/256.json"
              # "gaot/abla/data_size/poisson_gauss/512.json"
              # "gaot/abla/data_size/poisson_gauss/1024.json"

              # "gaot/abla/data_size/ns_gauss/128.json"
              # "gaot/abla/data_size/ns_gauss/256.json"
              # "gaot/abla/data_size/ns_gauss/512.json"

              "gaot/abla/data_size/ce_crp/128.json"
              "gaot/abla/data_size/ce_crp/256.json"
              "gaot/abla/data_size/ce_crp/512.json"
              "gaot/abla/data_size/ce_crp/1024.json"

              "gaot/abla/data_size/ce_kh/128.json"
              "gaot/abla/data_size/ce_kh/256.json"
              "gaot/abla/data_size/ce_kh/512.json"
              "gaot/abla/data_size/ce_kh/1024.json"

              "gaot/abla/data_size/ce_rpui/128.json"
              "gaot/abla/data_size/ce_rpui/256.json"
              "gaot/abla/data_size/ce_rpui/512.json"
              "gaot/abla/data_size/ce_rpui/1024.json"

              "gaot/abla/data_size/ns_sines/128.json"
              "gaot/abla/data_size/ns_sines/256.json"
              "gaot/abla/data_size/ns_sines/512.json"
              "gaot/abla/data_size/ns_sines/1024.json"


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

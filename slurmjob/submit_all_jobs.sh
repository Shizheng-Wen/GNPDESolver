#!/bin/bash

# list of config files
CONFIG_FILES=(
              "gaot/fluid_data/model_size/lift_channel/ce_gauss/32.json"
              "gaot/fluid_data/model_size/lift_channel/ce_gauss/64.json"
              "gaot/fluid_data/model_size/lift_channel/ce_gauss/128.json"
              "gaot/fluid_data/model_size/lift_channel/ce_rp/32.json"
              "gaot/fluid_data/model_size/lift_channel/ce_rp/64.json"
              "gaot/fluid_data/model_size/lift_channel/ce_rp/128.json"

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

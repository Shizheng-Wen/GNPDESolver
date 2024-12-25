#!/bin/bash

# list of config files
CONFIG_FILES=(
              "gaot/abla/multiscale/heat_l_sines.json"
              "gaot/abla/multiscale/wave_c_sines.json"
              "gaot/abla/data_size/ce_gauss/128.json"
              "gaot/abla/data_size/ce_gauss/256.json"
              "gaot/abla/data_size/ce_gauss/512.json"
      
              "gaot/abla/data_size/ce_rp/128.json"
              "gaot/abla/data_size/ce_rp/256.json"
              "gaot/abla/data_size/ce_rp/512.json"

              "gaot/abla/data_size/ns_pwc/128.json"
              "gaot/abla/data_size/ns_pwc/256.json"
              "gaot/abla/data_size/ns_pwc/512.json"

              "gaot/abla/data_size/ns_sl/128.json"
              "gaot/abla/data_size/ns_sl/256.json"
              "gaot/abla/data_size/ns_sl/512.json"

              "gaot/abla/data_size/ns_svs/128.json"
              "gaot/abla/data_size/ns_svs/256.json"
              "gaot/abla/data_size/ns_svs/512.json"
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

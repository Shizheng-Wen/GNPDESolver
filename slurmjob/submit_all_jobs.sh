#!/bin/bash

# list of config files
              # LANO


              # RANO
              # "rano/config_rano_poisson_c_sines.json"
              # "rano/config_rano_airfoil_li.json"
            # "lano/fluid_data/uvit/ce_gauss.json"
            # "lano/fluid_data/uvit/ce_rp.json"
            # "lano/fluid_data/uvit/ns_gauss.json"
            # "lano/fluid_data/uvit/ns_pwc.json"
            # "lano/fluid_data/uvit/ns_sl.json"
            # "lano/fluid_data/uvit/ns_svs.json"

              # GINO
              # "gino/config_gino_poisson_c.json"
              # "gino/config_gino_airfoil_li.json"
              # "gino/config_gino_airfoil_li_large.json"

              # "lscot/fluid_data/pretrain_T/test/ce_gauss.json"
              # "lscot/fluid_data/pretrain_T/test/ce_rp.json"
              # "lscot/fluid_data/pretrain_T/test/ns_gauss.json"
              # "lscot/fluid_data/pretrain_T/test/ns_pwc.json"
              # "lscot/fluid_data/pretrain_T/test/ns_sl.json"
              # "lano/fluid_data/stepper/ce_rp.json"
              # "lano/fluid_data/stepper/ns_gauss.json"
              # "lano/fluid_data/stepper/ns_pwc.json"
              # "lano/fluid_data/stepper/ns_sl.json"
              # "lano/fluid_data/stepper/ns_svs.json"
CONFIG_FILES=(
              "gaot/fluid_data/full/multiscale/ce_gauss.json"
              "gaot/fluid_data/full/multiscale/ce_rp.json"
              # "gaot/fluid_data/full/multiscale/ns_gauss.json"
              # "gaot/fluid_data/full/multiscale/ns_pwc.json"
              # "gaot/fluid_data/full/multiscale/ns_sl.json"
              # "gaot/fluid_data/full/multiscale/ns_svs.json"
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

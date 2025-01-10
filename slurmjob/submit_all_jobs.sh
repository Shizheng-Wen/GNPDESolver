#!/bin/bash

# list of config files
CONFIG_FILES=(
              # "gaot/hyperbolic/full/wave_layer.json"
              # "gaot/hyperbolic/full/wave_gauss.json"

              # "gaot/fluid_data/full/multiscale/ns_gauss.json"
              # "gaot/fluid_data/full/stepper/derivate/ns_pwc.json"
              # "gaot/fluid_data/full/multiscale/ns_sl.json"
              # "gaot/fluid_data/full/stepper/derivate/ns_svs.json"
              # "gaot/fluid_data/model_size/num_trans/ce_gauss/10.json"
              # "gaot/fluid_data/model_size/lift_channel/ce_rp/64.json"
              # "gaot/parabolic/full/ace.json"
              # "gaot/static/poisson_gauss.json"
              # "gaot/static/airfoil_grid.json"
              # "gaot/abla/data_size/ce_crp/1024.json"
              # "gaot/abla/data_size/ce_kh/1024.json"
              # "gaot/abla/data_size/ce_rpui/1024.json"
              # "gaot/abla/data_size/ns_sines/1024.json"
              
              # "gaot/fluid_data/sparse/ns_gauss.json"
              # "gaot/fluid_data/sparse/ns_pwc.json"
              # "gaot/fluid_data/sparse/ns_sl.json"
              # "gaot/fluid_data/sparse/ns_svs.json"
              #"gaot/fluid_data/sparse/ce_gauss_10blocks.json"
              # "gaot/fluid_data/sparse/geoemb/stats/ce_rp.json"
              # "gaot/parabolic/sparse/ace.json"
              # "gaot/hyperbolic/sparse/wave_layer.json"
              # "gaot/abla/multiscale/wave_c_sines.json"
              # "gaot/abla/multiscale/heat_l_sines.json"
              # "gaot/static/sparse/poisson_gauss.json"
              # "gaot/abla/multiscale/poisson_c_sines.json"
              # "gaot/airfoil/airfoil_li/geoemb/lano.json"
              # "gaot/static/elasticity.json"
# ----------------------Foundation Model---------------------------------
              # "gaot/abla/data_size/wave_l_sines/128.json"
              # "gaot/abla/data_size/wave_l_sines/256.json"
              # "gaot/abla/data_size/wave_l_sines/512.json"
              # "gaot/abla/data_size/wave_l_sines/1024.json"
              # "gaot/fd/finetune/wave_l_sines/fd_128.json"
              "gaot/fd/finetune/wave_l_sines/fd_256.json"
              "gaot/fd/finetune/wave_l_sines/fd_512.json"
              "gaot/fd/finetune/wave_l_sines/fd_1024.json"
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

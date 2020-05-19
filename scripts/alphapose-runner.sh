#!/bin/bash

FLAG_KWARGS="|verbose|"

if [ ! -n "$FLAG_KWARGS" ]; then
    FLAG_KWARGS=""
fi

declare -a ARGS
declare -A KWARGS

echo "=============================================================================="
echo "debug info"
echo "=============================================================================="
echo "--raw args--------------------------------------------------------------------"
echo "$@"
echo "=============================================================================="

while (( "$#" )); do
  case "$1" in
    -*|--*)
      KEY=${1##*-}
      echo "found labeled arg '$KEY'"
      if [[ "$FLAG_KWARGS" =~ "|${KEY}|" ]]; then
          echo "  is flag"
          KWARGS["$KEY"]="true"
          shift 1
      else
          if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            echo "  has value '$2'"
            KWARGS["$KEY"]="$2"
            shift 2
          else
            echo "  is flag, but not labeled as such"
            KWARGS["$KEY"]="true"
            shift 1
          fi
      fi
      ;;
    *) # preserve positional arguments
      ARGS+=("$1")
      echo "found positional arg '$1'"
      shift 1
      ;;
  esac
done

REDIS="redis-cli -h localhost"
POSER="producer video"

echo "=============================================================================="
echo "debug info"
echo "=============================================================================="
echo "--KWARGS----------------------------------------------------------------------"
echo $KWARGS
echo "--ARGS------------------------------------------------------------------------"
echo $ARGS
echo "=============================================================================="

environment_id=${KWARGS["environment_id"]:-"724fe65b-f925-48a1-9ae0-ee1b85443d64"}  # 724fe65b-f925-48a1-9ae0-ee1b85443d64
assignment_id=${KWARGS["assignment_id"]:?"assignment_id is required"}
start_date=${KWARGS["start"]}  # 2020-03-10T00:00:00+0000
duration=${KWARGS["dur"]:-"1d"}
slot=${KWARGS["slot"]}
state_id=$(producer hash $start_date $duration)
date="${start_date: 0:4}/${start_date: 5:2}/${start_date: 8:2}"
available_gpus=$($REDIS lrange "airflow.gpu.slots.available" 0 $($REDIS llen "airflow.gpu.slots.available"))

echo $available_gpus

if [ ${KWARGS["verbose"]} = "true" ]; then
    echo "available_gpus:: $available_gpus"
    echo "environment_id:: $environment_id"
    echo "assignment_id:: $assignment_id"
fi

function log_verbose() {
    if [ ${KWARGS["verbose"]} = "true" ]; then
        echo $1
    fi
}

start=`date +%s`

if [ -d /data/prepared/$environment_id/$assignment_id/$date/ ]
then
    log_verbose /data/prepared/$environment_id/$assignment_id/$state_id.$slot.json
    for f in $(cat /data/prepared/$environment_id/$assignment_id/$state_id.$slot.json | jq -r '.[].video')
    do
        if [ ! -d /data/prepared/$environment_id/$assignment_id/$date/${f: -12:-4}/*.json ]; then
            echo "allocating GPU"
            selected_gpu=""
            iterations=0
            while [[ "$selected_gpu" -eq "" ]]
            do
                for gpu in $available_gpus
                do
                    key="airflow.gpu.slots.$gpu"
                    aquired=$($REDIS setnx $key holding)
                    if [ $aquired -eq 1 ]; then
                        selected_gpu=$gpu
                        break 2
                    fi
                done
                if [[ ! "$selected_gpu" == "" ]]; then
                    break
                fi
                iterations=$(( iterations++ ))
                if [ $iterations -gt 69 ]
                then
                    exit 22
                fi
                sleep 1
            done
            if [[ ! "$selected_gpu" == "" ]]; then
                echo "GPU aquired - executing inference $selected_gpu"
                GPU=$selected_gpu $POSER $f
                key="airflow.gpu.slots.$selected_gpu"
                free=$($REDIS del $key)
            fi            # ./build/examples/openpose/openpose.bin --video \
            #     $f \
            #     --model_folder /openpose/models/ --num_gpu "$num_gpu" \
            #     --num_gpu_start "$num_gpu_start" --model_pose BODY_25 \
            #     --write_json /data/prepared/$environment_id/$assignment_id/$date/ --display 0 --render_pose 0
        fi
    done
else
    echo "nothing to do"
fi

date

end=`date +%s`
runtime=$((end-start))

if [ ${KWARGS["verbose"]} = "true" ]; then
    echo "runtime:: $runtime"
fi

exit 0

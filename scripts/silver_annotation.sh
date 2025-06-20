#!/usr/bin/bash 

# read config parameter: if no present, stick to default
CONFIG=${1:-"config/compmix/quasar.yml"}

# adjust name to log
IFS='/' read -ra NAME <<< "$CONFIG"
DATA=${NAME[1]}
IFS='.' read -ra NAME <<< "${NAME[2]}"
NAME=${NAME[0]}
OUT=out/${DATA}/silver_annotation_${NAME}.out
mkdir -p out/${DATA}
echo "Writing log to ${OUT}..."

# start script
export CONFIG OUT
nohup sh -c 'python -u quasar/distant_supervision/silver_annotation.py --inference ${CONFIG}'  > $OUT 2>&1 &
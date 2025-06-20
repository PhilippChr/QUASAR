#!/usr/bin/bash

## check argument length
if [[ $# -lt 1 ]]
then
	echo "Error: Invalid number of options: Please specify at least the pipeline-function."
	echo "Usage: bash scripts/gnn.sh\\
		--train\\
		[<PATH_TO_CONFIG>] [<SOURCES_STR>] [<ADDITIONAL_OPTION>]"
	exit 0
fi

## read config parameter: if no present, stick to default
FUNCTION=$1
CONFIG=${2:-"config/compmix/quasar.yml"}
SOURCES=${3:-"kb_text_table_info"}

## set path for output
# get function name
FUNCTION_NAME=${FUNCTION#"--"}
# get data name
IFS='/' read -ra NAME <<< "$CONFIG"
DATA=${NAME[1]}
# get config name
if [[ ${#NAME[@]} -gt 3 ]]
then
  CFG_FOLDER="${NAME[2]}/"
  CFG_NAME=${NAME[3]%".yml"}
else
  CFG_NAME=${NAME[2]%".yml"}
fi

# set output path (include sources only if not default value)
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
if [[ $# -lt 3 ]]
then
	OUT="out/${DATA}/${DATE}_${CFG_FOLDER}gnn-${FUNCTION_NAME}-${CFG_NAME}.out"
else
	OUT="out/${DATA}/${DATE}_${CFG_FOLDER}gnn-${FUNCTION_NAME}-${CFG_NAME}-${SOURCES}.out"
fi
echo "Writing log to ${OUT}..."

# load conda
eval "$(conda shell.bash hook)"
conda activate quasar

## start script
if ! command -v sbatch &> /dev/null
then
	# no slurm setup: run via nohup
	export FUNCTION CONFIG SOURCES OUT
  nohup sh -c 'python -u quasar/heterogeneous_answering/graph_neural_network/iterative_gnns.py ${FUNCTION} ${CONFIG} ${SOURCES}' > $OUT 2>&1 &
else
	# run with sbatch
	sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$OUT
#SBATCH -p gpu20
#SBATCH --gres gpu:2
#SBATCH -t 0-05:00:00
#SBATCH -o $OUT
#SBATCH -d singleton
#SBATCH --mem 256G

python -u quasar/heterogeneous_answering/graph_neural_network/iterative_gnns.py $FUNCTION $CONFIG $SOURCES 
EOT
fi

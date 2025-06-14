#!/usr/bin/bash 

## check argument length
if [[ $# -lt 1 ]]
then
	echo "Error: Invalid number of options: Please specify the data which should be downloaded."
	echo "Usage: bash scripts/download.sh <DATA_FOR_DOWNLOAD>"
	exit 0
fi

case "$1" in
"quasar")
    echo "Downloading QUASAR data..."
    wget http://qa.mpi-inf.mpg.de/quasar/data/quasar.zip
    mkdir -p _data/compmix/
    unzip quasar.zip -d _data/compmix/
    rm quasar.zip
    echo "Successfully downloaded QUASAR data!"
    ;;
"compmix")
    echo "Downloading ConvMix dataset..."
    mkdir -p _benchmarks/compmix
    cd _benchmarks/compmix
    wget http://qa.mpi-inf.mpg.de/quasar/data/compmix.zip
    unzip compmix.zip
    echo "Successfully downloaded ConvMix dataset!"
    ;;
*)
    echo "Error: Invalid specification of the data. Data $1 could not be found."
	exit 0
    ;;
esac

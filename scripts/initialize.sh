#!/usr/bin/bash 

# create directories
mkdir -p _benchmarks
mkdir -p _data
mkdir -p _intermediate_representations
mkdir -p _results
mkdir -p _results/compmix
mkdir -p out
mkdir -p out/compmix
mkdir -p out/slurm

# download 
bash scripts/download.sh compmix
bash scripts/download.sh quasar
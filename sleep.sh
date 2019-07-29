#!/bin/bash
#PBS -N one_line_21961
#PBS -o $PBS_JOBID.out
#PBS -e $PBS_JOBID.err
#PBS -W umask=022
#PBS -q mem64
#PBS -l nodes=1:ppn=16

source ~/.bashrc

echo "Starting on $(hostname) at $(date)"

cd $PBS_O_WORKDIR

sleep 3600

echo "Job Ended at $(date)"
echo '======================================================='

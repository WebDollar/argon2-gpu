#!/bin/bash

machines="$1"
branches="$2"
batch_size="$3"
samples="$4"
duration="$5"
queue="$6"
run_tests="$7"

if [ -z "$machines" ]; then
    echo "ERROR: Machines must be specified!" 1>&2
    exit 1
fi

if [ -z "$branches" ]; then
    echo "ERROR: Branches must be specified!" 1>&2
    exit 1
fi

if [ -z "$batch_size" ]; then
    batch_size=256
fi

if [ -z "$samples" ]; then
    samples=5
fi

if [ -z "$duration" ]; then
    duration=1d
fi

if [ -z "$queue" ]; then
    queue=gpu
fi

REPO_URL='https://gitlab.com/omos/argon2-gpu.git'

dest_dir="$(pwd)"

task_file="$(mktemp)"

qsub_old="$(which qsub)"

module add pbspro-client

for machine in $machines; do
    spec="#PBS -l nodes=1:ppn=1:gpu=1:mem=16gb:cl_$machine"
    qsub=qsub
    case "$machine" in pro:*)
        machine="${machine#pro:}"
        spec="#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb:cl_$machine=True"
        qsub="$qsub_old"
    esac
    
    cat >$task_file <<EOF
#!/bin/bash
#PBS -N argon2-gpu-$machine-($branches)
#PBS -l walltime=$duration
$spec
#PBS -q $queue

module add cmake-3.6.1
module add cuda-8.0

mkdir -p "$dest_dir/\$PBS_JOBID" || exit 1

cd "$dest_dir/\$PBS_JOBID" || exit 1

mkdir include || exit 1

git clone 'https://github.com/KhronosGroup/OpenCL-Headers.git' include/CL || exit 1
(cd include/CL && git checkout opencl11) || exit 1

export C_INCLUDE_PATH="\$C_INCLUDE_PATH:\$(readlink -f include)"
export CPLUS_INCLUDE_PATH="\$CPLUS_INCLUDE_PATH:\$(readlink -f include)"

git clone "$REPO_URL" argon2-gpu || exit 1

cd argon2-gpu || exit 1

(cmake . && make) || exit 1

if [ "$run_tests" == "yes" ]; then
    ./argon2-gpu-test
fi

bash scripts/benchmark-commits.sh "$machine" . .. "$batch_size" "$samples" '' '' '' '' '' $branches
EOF

    "$qsub" "$task_file"
done

rm -f "$task_file"

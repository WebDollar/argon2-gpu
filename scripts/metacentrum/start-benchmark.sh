#!/bin/bash

machine="$1"
max_memory="$2"
batch_size="$3"
samples="$4"
branch="$5"
duration="$6"
queue="$7"
run_tests="$8"

if [ -z "$machine" ]; then
    echo "ERROR: Machine must be specified!" 1>&2
    exit 1
fi

if [ -z "$max_memory" ]; then
    echo "ERROR: Maximum memory must be specified!" 1>&2
    exit 1
fi

if [ -z "$batch_size" ]; then
    batch_size=256
fi

if [ -z "$samples" ]; then
    samples=5
fi

if [ -z "$branch" ]; then
    branch='master'
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

cat >$task_file <<EOF
#!/bin/bash
#PBS -N argon2-gpu-$machine-$branch
#PBS -l walltime=$duration
#PBS -l nodes=1:ppn=1:cl_$machine
#PBS -l gpu=1
#PBS -q $queue
#PBS -l mem=16gb

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

git checkout "$branch" || exit 1

(cmake . && make) || exit 1

if [ "$run_tests" == "yes" ]; then
    ./argon2-gpu-test
fi

bash scripts/run-benchmark.sh "$max_memory" "$batch_size" "$samples" \
    >"$dest_dir/\$PBS_JOBID/benchmark-$machine-$branch.csv"
EOF

qsub "$task_file"

rm -f "$task_file"

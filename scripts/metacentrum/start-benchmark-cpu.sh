#!/bin/bash

machine="$1"
machine_spec="$2"
branches="$3"
ncpus="$4"
samples="$5"
duration="$6"
queue="$7"

if [ -z "$machine" ]; then
    echo "ERROR: Machine must be specified!" 1>&2
    exit 1
fi

if [ -z "$machine_spec" ]; then
    echo "ERROR: Machine spec must be specified!" 1>&2
    exit 1
fi

if [ -z "$branches" ]; then
    echo "ERROR: Branches must be specified!" 1>&2
    exit 1
fi

if [ -z "$ncpus" ]; then
    echo "ERROR: Number of CPU cores must be specified!" 1>&2
    exit 1
fi

if [ -z "$samples" ]; then
    samples=32
fi

if [ -z "$duration" ]; then
    duration=24:00:00
fi

REPO_URL='https://gitlab.com/omos/argon2-gpu.git'

dest_dir="$(pwd)"

task_file="$(mktemp)"

module add pbspro-client

case "$machine_spec" in
    cluster)
        machine_spec=":cl_$machine=True"
        ;;
    node:*)
        machine_spec=":vnode=${machine_spec#node:}"
        ;;
    none)
        machine_spec=""
        ;;
    *)
        echo "ERROR: Invalid machine spec!" 1>&2
        exit 1
        ;;
esac

cat >$task_file <<EOF
#!/bin/bash
#PBS -N argon2-cpu-$machine-${branches// /:}
#PBS -l select=1:ncpus=$ncpus:mem=32gb$machine_spec
#PBS -l walltime=$duration
$(if [ -n "$queue" ]; then echo "#PBS -q $queue"; fi)

module add gcc-5.3.0
module add cmake-3.6.1
module add cuda-8.0

mkdir -p "$dest_dir/\$PBS_JOBID" || exit 1

cd "$dest_dir/\$PBS_JOBID" || exit 1

mkdir include || exit 1

git clone 'https://github.com/KhronosGroup/OpenCL-Headers.git' include/CL || exit 1
(cd include/CL && git checkout opencl11) || exit 1

export C_INCLUDE_PATH="\$C_INCLUDE_PATH:\$(readlink -f include)"
export CPLUS_INCLUDE_PATH="\$CPLUS_INCLUDE_PATH:\$(readlink -f include)"

git clone --recursive "$REPO_URL" argon2-gpu || exit 1

cd argon2-gpu || exit 1

(cmake -DCMAKE_BUILD_TYPE=Release . && make) 1>../build.log 2>&1 || exit 1

bash scripts/benchmark-commits.sh "cpu-$machine" . .. $((2*$ncpus)) "$samples" cpu '' '' '' '' $branches 1>../bench.log 2>&1
EOF

qsub "$task_file"
rm -f "$task_file"

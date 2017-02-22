#!/bin/bash

max_memory="$1"
batch_size="$2"
samples="$3"

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

echo "mode,kernel,version,type,t_cost,m_cost,lanes,ns_per_hash"
for mode in opencl cuda; do
    for kernel in by-segment oneshot; do
        for version in 1.3 1.0; do
            for type in i d; do
                for (( m_cost = 64; m_cost <= $(( $max_memory / $batch_size )); m_cost *= 4 )); do
                    for (( t_cost = 1; t_cost <= 16; t_cost *= 2 )); do
                        for (( lanes = 1; lanes <= 8; lanes *= 2 )); do
                            ns_per_hash=$(./argon2-gpu-bench -m $mode -k $kernel -b $batch_size -T $t_cost -M $m_cost -L $lanes -o ns-per-hash --output-mode mean -s $samples)
                            
                            echo "$mode,$kernel,v$version,Argon2$type,$t_cost,$m_cost,$lanes,$ns_per_hash"
                        done
                    done
                done
            done
        done
    done
done

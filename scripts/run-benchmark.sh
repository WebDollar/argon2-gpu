#!/bin/bash

batch_size="$1"
samples="$2"
modes="$3"
kernels="$4"
versions="$5"
types="$6"
precomputes="$7"

if [ -z "$batch_size" ]; then
    batch_size=64
fi

if [ -z "$samples" ]; then
    samples=5
fi

if [ -z "$modes" ]; then
    modes='opencl cuda'
fi

if [ -z "$kernels" ]; then
    kernels='by-segment oneshot'
fi

if [ -z "$versions" ]; then
    versions='1.3 1.0'
fi

if [ -z "$types" ]; then
    types='i d'
fi

if [ -z "$precomputes" ]; then
    precomputes='no yes'
fi

max_m_cost=1024

echo "[INFO] Benchmarking max memory cost..." 1>&2
while true; do
    next_m_cost=$(( $max_m_cost * 2 ))
    if ! ./argon2-gpu-bench -t i -v 1.3 -p -m cuda -b $batch_size -s 1 -T 1 -M $next_m_cost -L 16 >/dev/null 2>/dev/null; then
        break
    fi
    max_m_cost=$next_m_cost
done
echo "[INFO] Max memory cost: $max_m_cost" 1>&2

echo "mode,kernel,version,type,precompute,t_cost,m_cost,lanes,ns_per_hash"
for mode in $modes; do
    for kernel in $kernels; do
        for version in $versions; do
            for type in $types; do
                if [ $type == 'i' ]; then
                    precomputes2="$precomputes"
                else
                    precomputes2='no'
                fi
                for precompute in $precomputes2; do
                    for (( m_cost = 64; m_cost <= $max_m_cost; m_cost *= 4 )); do
                        for (( t_cost = 1; t_cost <= 16; t_cost *= 2 )); do
                            for (( lanes = 1; lanes <= 8; lanes *= 2 )); do
                                if [ $precompute == 'yes' ]; then
                                    precompute_flag='-p'
                                else
                                    precompute_flag=''
                                fi
                                ns_per_hash=$(./argon2-gpu-bench \
                                    -t $type -v $version \
                                    $precompute_flag \
                                    -m $mode -k $kernel \
                                    -b $batch_size -s $samples \
                                    -T $t_cost -M $m_cost -L $lanes \
                                    -o ns-per-hash --output-mode mean)
                                
                                echo "$mode,$kernel,v$version,Argon2$type,$precompute,$t_cost,$m_cost,$lanes,$ns_per_hash"
                            done
                        done
                    done
                done
            done
        done
    done
done

#!/bin/bash

max_batch_size="$1"
samples="$2"
modes="$3"
kernels="$4"
versions="$5"
types="$6"
precomputes="$7"

if [ -z "$max_batch_size" ]; then
    max_batch_size=64
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
    types='i d id'
fi

if [ -z "$precomputes" ]; then
    precomputes='no yes'
fi

MAX_WORK=32768

echo "mode,kernel,version,type,precompute,t_cost,m_cost,lanes,ns_per_hash,batch_size"
for mode in $modes; do
    for kernel in $kernels; do
        for version in $versions; do
            for type in $types; do
                if [ $type != 'd' ]; then
                    precomputes2="$precomputes"
                else
                    precomputes2='no'
                fi
                for precompute in $precomputes2; do
                    for t_cost in 1 2 4 6 8; do
                        for (( lanes = 1; lanes <= 8; lanes *= 2 )); do
                            batch_size=$max_batch_size
                            for (( m_cost = 64; ; m_cost *= 2 )); do
                                if [ $precompute == 'yes' ]; then
                                    precompute_flag='-p'
                                else
                                    precompute_flag=''
                                fi
                                
                                ret=1
                                while [ $(( $m_cost / $batch_size * $lanes )) -le $MAX_WORK ]; do
                                    ns_per_hash=$(./argon2-gpu-bench \
                                        -t $type -v $version \
                                        $precompute_flag \
                                        -m $mode -k $kernel \
                                        -b $batch_size -s $samples \
                                        -T $t_cost -M $m_cost -L $lanes \
                                        -o ns-per-hash --output-mode mean)
                                    ret=$?
                                    if [ $ret == 0 ]; then
                                        break
                                    fi
                                    
                                    (( batch_size /= 2 ))
                                done
                                
                                if [ $ret != 0 ]; then
                                    break
                                fi
                                
                                echo "$mode,$kernel,v$version,Argon2$type,$precompute,$t_cost,$m_cost,$lanes,$ns_per_hash,$batch_size"
                            done
                        done
                    done
                done
            done
        done
    done
done

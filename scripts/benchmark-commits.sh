#!/bin/bash

dirname="$(dirname "$0")"

bench_id="$1"
src_dir="$2"

if [ -z "$bench_id" ]; then
    echo "ERROR: Bench ID not specified!" 1>&2
    exit 1
fi

shift 2

max_memory="$1"
batch_size="$2"
samples="$3"

if [ -z "$max_memory" ]; then
    echo "ERROR: Max memory not specified!" 1>&2
    exit 1
fi

if [ -z "$batch_size" ]; then
    echo "ERROR: Batch size not specified!" 1>&2
    exit 1
fi

if [ -z "$samples" ]; then
    echo "ERROR: Sample count not specified!" 1>&2
    exit 1
fi

shift 3

for commit in $@; do
    (cd "$src_dir" && git checkout "$commit") || exit 1
    
    make || exit 1
    
    "$dirname/run-benchmark.sh" "$max_memory" "$batch_size" "$samples" \
        | tee "bench-$bench_id-$commit.csv" || exit 1
done

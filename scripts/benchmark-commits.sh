#!/bin/bash

dirname="$(dirname "$0")"

bench_id="$1"
src_dir="$2"
dst_dir="$3"

if [ -z "$bench_id" ]; then
    echo "ERROR: Bench ID not specified!" 1>&2
    exit 1
fi

if [ -z "$src_dir" ]; then
    echo "ERROR: Source directory not specified!" 1>&2
    exit 1
fi

if [ -z "$dst_dir" ]; then
    echo "ERROR: Destination directory not specified!" 1>&2
    exit 1
fi

shift 3

max_batch_size="$1"
samples="$2"
modes="$3"
kernels="$4"
versions="$5"
types="$6"
precomputes="$7"

if [ -z "$max_batch_size" ]; then
    echo "ERROR: Batch size not specified!" 1>&2
    exit 1
fi

if [ -z "$samples" ]; then
    echo "ERROR: Sample count not specified!" 1>&2
    exit 1
fi

shift 7

for commit in $@; do
    (cd "$src_dir" && git rev-parse --verify "$commit") >"$dst_dir/hash-$bench_id-$commit.txt" || exit 1
    (cd "$src_dir" && git checkout "$commit") || exit 1
    
    make || exit 1
    
    "$dirname/run-benchmark.sh" "$max_batch_size" "$samples" \
        "$modes" "$kernels" "$versions" "$types" "$precomputes" \
        | tee "$dst_dir/bench-$bench_id-$commit.csv" || exit 1
done

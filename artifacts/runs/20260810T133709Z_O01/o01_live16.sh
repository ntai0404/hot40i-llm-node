#!/system/bin/sh
set -u

cd /data/local/tmp/h40m || exit 90
binary=./minimal_decoder_cache_android
source_dir=./source
catalog=./h40m/tensor_catalog.tsv
arena=./h40m/expert_arena.bin
tokens=12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194

rm -f o01_live16.done
for policy in lru per_layer_hotset; do
    echo "$policy $$" > o01_live16.active
    H40_THREADS=6 H40_IO_OVERLAP=1 H40_CACHE_POLICY="$policy" \
        taskset ff "$binary" "$source_dir" "$catalog" "$arena" "$tokens" \
        "o01_${policy}_16.json" > "o01_${policy}_16.stdout" 2> "o01_${policy}_16.stderr"
    status=$?
    echo "$status" > "o01_${policy}_16.exit"
    if [ "$status" -ne 0 ]; then
        echo "failed $policy $status" > o01_live16.done
        exit "$status"
    fi
done
echo "complete 0" > o01_live16.done
rm -f o01_live16.active

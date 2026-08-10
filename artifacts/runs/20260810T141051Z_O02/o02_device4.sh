#!/system/bin/sh
set -u

cd /data/local/tmp/h40m || exit 90
binary=./minimal_decoder_window_android
source_dir=./source
catalog=./h40m/tensor_catalog.tsv
arena=./h40m/expert_arena.bin
tokens=12194,12194,12194,12194

rm -f o02_device4.done
for candidate in off exact_w1 approximate_w1 approximate_w4; do
    mode=off
    window=0
    case "$candidate" in
        exact_w1) mode=exact; window=1 ;;
        approximate_w1) mode=approximate; window=1 ;;
        approximate_w4) mode=approximate; window=4 ;;
    esac
    echo "$candidate $$" > o02_device4.active
    H40_THREADS=6 H40_IO_OVERLAP=1 H40_EXPERT_REUSE="$mode" H40_REUSE_WINDOW="$window" \
        taskset ff "$binary" "$source_dir" "$catalog" "$arena" "$tokens" \
        "o02_${candidate}_4.json" > "o02_${candidate}_4.stdout" 2> "o02_${candidate}_4.stderr"
    status=$?
    echo "$status" > "o02_${candidate}_4.exit"
    if [ "$status" -ne 0 ]; then
        echo "failed $candidate $status" > o02_device4.done
        exit "$status"
    fi
done
echo "complete 0" > o02_device4.done
rm -f o02_device4.active

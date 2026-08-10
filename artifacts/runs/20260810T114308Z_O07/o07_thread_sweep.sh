#!/system/bin/sh
set -u

root=/data/local/tmp/h40m
binary="$root/minimal_decoder_o07_parallel"
tokens=12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194,12194

rm -f "$root/o07_sweep.done" "$root/o07_sweep.active"
for threads in 1 2 4 6 8; do
  prefix="$root/o07_t${threads}_16"
  rm -f "$prefix.json" "$prefix.stdout" "$prefix.stderr" "$prefix.exit" "$prefix.top" "$prefix.state"
  H40_THREADS="$threads" H40_IO_OVERLAP=1 "$binary" \
    "$root/source" \
    "$root/h40m/tensor_catalog.tsv" \
    "$root/h40m/expert_arena.bin" \
    "$tokens" \
    "$prefix.json" \
    >"$prefix.stdout" 2>"$prefix.stderr" &
  pid=$!
  echo "$threads $pid" > "$root/o07_sweep.active"
  sleep 30
  {
    echo "threads=$threads pid=$pid"
    top -H -b -n 1 -m 40
    echo "===STATUS==="
    cat "/proc/$pid/status" 2>/dev/null
    echo "===AFFINITY==="
    taskset -pc "$pid" 2>/dev/null
    echo "===FREQUENCIES==="
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
      printf '%s ' "$cpu"
      cat "$cpu/cpufreq/scaling_cur_freq" 2>/dev/null
    done
  } > "$prefix.top" 2>&1
  wait "$pid"
  code=$?
  echo "$code" > "$prefix.exit"
  {
    cat /proc/meminfo | grep -E 'MemAvailable|SwapTotal|SwapFree'
    grep -E 'pswpin |pswpout |pgmajfault ' /proc/vmstat
    dumpsys thermalservice | grep 'Thermal Status'
  } > "$prefix.state" 2>&1
  if [ "$code" -ne 0 ]; then
    echo "$threads $code" > "$root/o07_sweep.done"
    exit "$code"
  fi
done
echo "complete 0" > "$root/o07_sweep.done"
rm -f "$root/o07_sweep.active"

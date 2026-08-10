#!/system/bin/sh
set -u

root=/data/local/tmp/h40m
binary="$root/minimal_decoder_o07_final"
tokens=12194
index=1
while [ "$index" -lt 40 ]; do
  tokens="$tokens,12194"
  index=$((index + 1))
done

rm -f "$root/o06_sustained.active" "$root/o06_sustained.done"
for threads in 6 8; do
  prefix="$root/o06_sustained_t${threads}"
  rm -f "$prefix.json" "$prefix.stdout" "$prefix.stderr" "$prefix.exit" "$prefix.telemetry" "$prefix.top"
  H40_THREADS="$threads" H40_IO_OVERLAP=1 taskset ff "$binary" \
    "$root/source" \
    "$root/h40m/tensor_catalog.tsv" \
    "$root/h40m/expert_arena.bin" \
    "$tokens" \
    "$prefix.json" \
    >"$prefix.stdout" 2>"$prefix.stderr" &
  pid=$!
  echo "$threads $pid" > "$root/o06_sustained.active"
  sample=0
  while kill -0 "$pid" 2>/dev/null; do
    {
      echo "=== sample=$sample epoch=$(date +%s) threads=$threads pid=$pid ==="
      grep -E '^cpu[0-7] ' /proc/stat
      grep -E 'MemAvailable|SwapTotal|SwapFree' /proc/meminfo
      grep -E 'pswpin |pswpout |pgmajfault ' /proc/vmstat
      printf 'freq_little_khz='; cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
      printf 'freq_big_khz='; cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq
      dumpsys thermalservice | grep 'Thermal Status'
      cat "/proc/$pid/stat" 2>/dev/null
    } >> "$prefix.telemetry" 2>&1
    if [ $((sample % 5)) -eq 0 ]; then
      top -H -b -n 1 -m 30 >> "$prefix.top" 2>&1
    fi
    sample=$((sample + 1))
    sleep 60
  done
  wait "$pid"
  code=$?
  echo "$code" > "$prefix.exit"
  if [ "$code" -ne 0 ]; then
    echo "$threads $code" > "$root/o06_sustained.done"
    exit "$code"
  fi
done
echo "complete 0" > "$root/o06_sustained.done"
rm -f "$root/o06_sustained.active"

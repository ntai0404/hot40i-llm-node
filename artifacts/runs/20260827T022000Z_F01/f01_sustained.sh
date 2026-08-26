#!/system/bin/sh

set -u

ROOT=/data/local/tmp/h40m
SERVICE_PID="$(cat "$ROOT/f01_service.pid")"
DURATION_SECONDS=1860
START_EPOCH="$(date +%s)"
DEADLINE=$((START_EPOCH + DURATION_SECONDS))

rm -f "$ROOT/f01_requests.log" "$ROOT/f01_thermal.jsonl" "$ROOT/f01.done" "$ROOT/f01.active" "$ROOT/f01_failure.log"
printf '%s\n' "pid=$SERVICE_PID" "started_epoch=$START_EPOCH" "duration_seconds=$DURATION_SECONDS" > "$ROOT/f01.active"

request_infer() {
    printf '%b' 'POST /infer HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\nConnection: close\r\n\r\n12194' | nc -w 900 127.0.0.1 8080
}

request_metrics() {
    printf '%b' 'GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n' | nc -w 20 127.0.0.1 8080
}

request_health() {
    printf '%b' 'GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n' | nc -w 20 127.0.0.1 8080
}

snapshot() {
    label="$1"
    now="$(date +%s)"
    {
        printf '{"schema_version":1,"captured_at_epoch":%s,"label":"%s"' "$now" "$label"
        if [ -r "/proc/$SERVICE_PID/status" ]; then
            rss="$(grep '^VmRSS:' "/proc/$SERVICE_PID/status" | awk '{print $2}')"
            hwm="$(grep '^VmHWM:' "/proc/$SERVICE_PID/status" | awk '{print $2}')"
            threads="$(grep '^Threads:' "/proc/$SERVICE_PID/status" | awk '{print $2}')"
            printf ',"service_pid":%s,"service_alive":true,"service_rss_kib":%s,"service_hwm_kib":%s,"service_threads":%s' "$SERVICE_PID" "${rss:-0}" "${hwm:-0}" "${threads:-0}"
        else
            printf ',"service_pid":%s,"service_alive":false' "$SERVICE_PID"
        fi
        mem_available="$(grep '^MemAvailable:' /proc/meminfo | awk '{print $2}')"
        swap_total="$(grep '^SwapTotal:' /proc/meminfo | awk '{print $2}')"
        swap_free="$(grep '^SwapFree:' /proc/meminfo | awk '{print $2}')"
        printf ',"mem_available_kib":%s,"swap_total_kib":%s,"swap_free_kib":%s' "${mem_available:-0}" "${swap_total:-0}" "${swap_free:-0}"
        vmstat="$(grep -E '^(pswpin|pswpout|pgmajfault) ' /proc/vmstat | awk '{printf \"%s=%s;\",$1,$2}')"
        printf ',"vmstat":"%s"' "$vmstat"
        printf ',"cpu_freq_khz":{'
        first=1
        for policy in /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq; do
            [ -r "$policy" ] || continue
            name="$(dirname "$policy")"
            name="${name##*/}"
            value="$(cat "$policy")"
            if [ "$first" -eq 0 ]; then printf ','; fi
            first=0
            printf '"%s":%s' "$name" "${value:-0}"
        done
        printf '}'
        thermal="$(dumpsys thermalservice 2>/dev/null | grep 'Thermal Status' | head -n 1 | tr '\n' ' ' | sed 's/"/\\"/g')"
        battery="$(dumpsys battery 2>/dev/null | grep -E 'level:|temperature:|current now:|status:' | tr '\n' ';' | sed 's/"/\\"/g')"
        printf ',"thermal":"%s","battery":"%s"}\n' "$thermal" "$battery"
    } >> "$ROOT/f01_thermal.jsonl"
}

snapshot start
request_health > "$ROOT/f01_health_start.http" 2> "$ROOT/f01_health_start.stderr"

count=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    count=$((count + 1))
    request_start="$(date +%s)"
    printf 'request=%s start_epoch=%s\n' "$count" "$request_start" >> "$ROOT/f01_requests.log"
    request_infer > "$ROOT/f01_request_${count}.http" 2> "$ROOT/f01_request_${count}.stderr"
    request_rc=$?
    request_end="$(date +%s)"
    printf 'request=%s end_epoch=%s rc=%s elapsed_seconds=%s\n' "$count" "$request_end" "$request_rc" "$((request_end - request_start))" >> "$ROOT/f01_requests.log"
    request_metrics > "$ROOT/f01_metrics_${count}.http" 2> "$ROOT/f01_metrics_${count}.stderr"
    if [ "$request_rc" -ne 0 ]; then
        printf 'request=%s rc=%s\n' "$count" "$request_rc" >> "$ROOT/f01_failure.log"
    fi
    if [ ! -d "/proc/$SERVICE_PID" ]; then
        printf 'service_pid=%s disappeared after request=%s\n' "$SERVICE_PID" "$count" >> "$ROOT/f01_failure.log"
        break
    fi
    snapshot "after_request_${count}"
done

snapshot finish
request_metrics > "$ROOT/f01_metrics_final.http" 2> "$ROOT/f01_metrics_final.stderr"
request_health > "$ROOT/f01_health_final.http" 2> "$ROOT/f01_health_final.stderr"
printf 'requests_completed=%s\nfinished_epoch=%s\n' "$count" "$(date +%s)" > "$ROOT/f01.done"
rm -f "$ROOT/f01.active"

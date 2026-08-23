#!/system/bin/sh
set -eu

cd /data/local/tmp/h40m
service_pid="$(cat a00_service.pid)"

request() {
    printf '%b' "$1" | nc -w 900 127.0.0.1 8080
}

snapshot() {
    label="$1"
    {
        printf 'snapshot=%s\n' "$label"
        date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
        grep -E '^(VmRSS|VmHWM|Threads):' "/proc/$service_pid/status"
        grep -E '^(MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
        grep -E '^(pswpin|pswpout|pgmajfault) ' /proc/vmstat
        if [ -r /sys/block/zram0/mm_stat ]; then
            printf 'zram_mm_stat='
            cat /sys/block/zram0/mm_stat
        fi
    } >> a00_telemetry.log
}

rm -f a00_infer_*.http a00_metrics_*.http a00_telemetry.log a00_repeat.done
request 'GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n' > a00_metrics_0.http
snapshot before

i=1
while [ "$i" -le 3 ]; do
    printf 'request_%s_started ' "$i"
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    request 'POST /infer HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\nConnection: close\r\n\r\n12194' > "a00_infer_${i}.http"
    request 'GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n' > "a00_metrics_${i}.http"
    snapshot "after_${i}"
    printf 'request_%s_completed ' "$i"
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    i=$((i + 1))
done

printf 'pass\n' > a00_repeat.done

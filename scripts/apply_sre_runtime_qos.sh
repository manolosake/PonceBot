#!/usr/bin/env bash
set -euo pipefail

# Balanced profile for mixed workloads:
# - leaves ample CPU/RAM for LLMs and backend services
# - keeps Android emulators responsive

MEM_LIMIT="${SRE_QOS_MEM_LIMIT:-6g}"
MEM_RESERVATION="${SRE_QOS_MEM_RESERVATION:-3g}"
MEM_SWAP="${SRE_QOS_MEM_SWAP:-6g}"
CPUS_PER_CONTAINER="${SRE_QOS_CPUS:-4}"
CPU_SHARES="${SRE_QOS_CPU_SHARES:-768}"
BLKIO_WEIGHT="${SRE_QOS_BLKIO_WEIGHT:-300}"

containers=(
  "sre-runtime-gate-909be0d4"
  "sre-runtime-auth-a544"
  "sre-runtime-auth-a544-hostnet"
)

cpusets=(
  "0-3"
  "4-7"
  "8-11"
)

echo "[qos] applying profile: mem=$MEM_LIMIT mem_swap=$MEM_SWAP mem_res=$MEM_RESERVATION cpus=$CPUS_PER_CONTAINER cpu_shares=$CPU_SHARES blkio=$BLKIO_WEIGHT"

for i in "${!containers[@]}"; do
  c="${containers[$i]}"
  set_cpu="${cpusets[$i]}"

  if ! docker inspect "$c" >/dev/null 2>&1; then
    echo "[qos] skip: $c (container not found)"
    continue
  fi

  docker update \
    --cpuset-cpus "$set_cpu" \
    --cpus "$CPUS_PER_CONTAINER" \
    --memory "$MEM_LIMIT" \
    --memory-swap "$MEM_SWAP" \
    --memory-reservation "$MEM_RESERVATION" \
    --cpu-shares "$CPU_SHARES" \
    --blkio-weight "$BLKIO_WEIGHT" \
    "$c" >/dev/null

  echo "[qos] updated: $c cpuset=$set_cpu"

  if ! docker exec "$c" sh -lc 'test -c /dev/kvm' >/dev/null 2>&1; then
    echo "[qos] warning: $c does not expose /dev/kvm (emulator may fall back to software acceleration)"
  fi
done

echo
echo "[qos] resulting limits:"
docker inspect -f 'Name={{.Name}} Cpuset={{.HostConfig.CpusetCpus}} NanoCpus={{.HostConfig.NanoCpus}} Memory={{.HostConfig.Memory}} MemSwap={{.HostConfig.MemorySwap}} MemRes={{.HostConfig.MemoryReservation}} CpuShares={{.HostConfig.CpuShares}} BlkioWeight={{.HostConfig.BlkioWeight}}' "${containers[@]}" 2>/dev/null || true

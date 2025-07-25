#!/bin/bash

echo "===== CPU Info ====="

# CPU Model & Vendor
cpu_model=$(lscpu | grep "Model name" | sed 's/Model name:\s*//')
vendor_id=$(lscpu | grep "Vendor ID" | awk '{print $3}')

echo "CPU Model: $cpu_model"
echo "Vendor: $vendor_id"

# Number of physical cores
physical_cores=$(lscpu | grep "^Core(s) per socket" | awk '{print $4}')
sockets=$(lscpu | grep "^Socket(s):" | awk '{print $2}')
total_cores=$((physical_cores * sockets))
echo "Physical Cores: $total_cores"

# Logical cores
logical_cores=$(nproc)
echo "Logical Cores: $logical_cores"

echo ""
echo "===== Cache Sizes ====="
lscpu | grep "L1d cache"
lscpu | grep "L1i cache"
lscpu | grep "L2 cache"
lscpu | grep "L3 cache"

echo ""
echo "===== Prefetchers ====="

if [[ "$vendor_id" == "GenuineIntel" ]]; then
    # Check if rdmsr exists
    if ! command -v rdmsr &>/dev/null; then
        echo "⚠️  msr-tools not installed. Run: sudo apt install msr-tools"
        exit 0
    fi

    # Load MSR module
    sudo modprobe msr 2>/dev/null

    # Read IA32_MISC_ENABLE MSR (0x1A0) from CPU 0
    if sudo rdmsr -p0 0x1a0 &>/dev/null; then
        val=$(sudo rdmsr -p0 0x1a0)
        bin=$(printf "%064d\n" "$(echo "obase=2; ibase=16; $val" | bc)")

        echo "IA32_MISC_ENABLE (0x1A0) = $val (binary: $bin)"

        # Decode common prefetcher bits
        echo "Hardware Prefetcher (bit 9):            $([[ "${bin:54:1}" == "0" ]] && echo "Enabled" || echo "Disabled")"
        echo "Adjacent Cache Line Prefetcher (bit 19): $([[ "${bin:44:1}" == "0" ]] && echo "Enabled" || echo "Disabled")"
        echo "L2 Hardware Prefetcher (bit 37):         $([[ "${bin:26:1}" == "0" ]] && echo "Enabled" || echo "Disabled")"
        echo "L2 Streamer Prefetcher (bit 38):         $([[ "${bin:25:1}" == "0" ]] && echo "Enabled" || echo "Disabled")"
    else
        echo "⚠️  Could not read MSR 0x1A0 — prefetcher info unavailable."
    fi

else
    echo "Prefetcher check not supported for vendor: $vendor_id (likely AMD)"
    echo "Hint: Check BIOS/UEFI settings for prefetcher controls on AMD CPUs."
fi


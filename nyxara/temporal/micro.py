"""NYXARA · temporal/micro.py — Layer 1 (milliseconds): the body's nervous system.

The fastest layer of the fractal mind is a sub-agent that does nothing but *feel the
machine*: it samples real CPU, memory, load, process count, and network packet/byte
counters as fast as it is ticked, holds a short rolling window, and rolls that window up
into a :class:`~nyxara.temporal.signals.MicroAggregate` with spike-based **anomaly
flags** (a CPU pin, a memory wall, a sudden packet storm).

This is genuinely real on Linux — it reads ``/proc/stat`` (CPU jiffies), ``/proc/meminfo``
(memory), ``/proc/net/dev`` (per-interface packet/byte counters), and ``os.getloadavg``.
On a platform without ``/proc`` it degrades gracefully (load average where available, the
rest zeroed and marked ``source="fallback"``) rather than crashing — the loop above it
must never die because the body is on an unusual host.

When an :class:`~nyxara.kernel.bus.EventBus` is supplied, each aggregate is emitted on
``temporal.micro.aggregate`` (and anomalies on ``temporal.micro.anomaly``) so the rest of
the system can react in real time. Pure standard library.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Deque, List, Optional, Tuple

from nyxara.temporal.clock import MILLISECOND, RealClock, TemporalClock
from nyxara.temporal.signals import HardwareSample, MicroAggregate, NetworkSample

__all__ = ["MicroLayer"]


class MicroLayer:
    """Layer 1 — a millisecond-cadence monitor of real hardware and network packets."""

    scale_s: float = MILLISECOND

    def __init__(self, *, clock: Optional[TemporalClock] = None, window: int = 128,
                 bus: Any = None, cpu_alert: float = 90.0, mem_alert: float = 90.0,
                 packet_spike_factor: float = 8.0, packet_spike_floor: float = 50.0) -> None:
        self.clock: TemporalClock = clock or RealClock()
        self.bus = bus
        self.cpu_alert = float(cpu_alert)
        self.mem_alert = float(mem_alert)
        self.packet_spike_factor = float(packet_spike_factor)
        self.packet_spike_floor = float(packet_spike_floor)
        self._hw: Deque[HardwareSample] = deque(maxlen=max(2, window))
        self._net: Deque[NetworkSample] = deque(maxlen=max(2, window))
        self._prev_cpu: Optional[Tuple[float, float]] = None   # (busy, total) jiffies
        self.ticks = 0
        self.aggregates = 0
        self._has_proc = os.path.isdir("/proc")

    # ------------------------------------------------------------------ #
    # real readings
    # ------------------------------------------------------------------ #
    def _read_cpu_percent(self) -> Tuple[float, str]:
        """Busy fraction since the previous read, from ``/proc/stat`` jiffies."""
        try:
            with open("/proc/stat", "r", encoding="utf-8") as fh:
                parts = fh.readline().split()
            # cpu user nice system idle iowait irq softirq steal guest guest_nice
            vals = [float(x) for x in parts[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)   # idle + iowait
            total = sum(vals)
            busy = total - idle
            pct = 0.0
            if self._prev_cpu is not None:
                d_busy = busy - self._prev_cpu[0]
                d_total = total - self._prev_cpu[1]
                pct = (100.0 * d_busy / d_total) if d_total > 0 else 0.0
            self._prev_cpu = (busy, total)
            return max(0.0, min(100.0, pct)), "proc"
        except Exception:  # noqa: BLE001 — never let the body fall over the loop
            return 0.0, "fallback"

    def _read_mem(self) -> Tuple[float, float, float, str]:
        """(percent_used, used_mb, total_mb, source) from ``/proc/meminfo``."""
        try:
            info = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    k, _, rest = line.partition(":")
                    info[k.strip()] = float(rest.split()[0])   # value is in kB
            total_kb = info.get("MemTotal", 0.0)
            avail_kb = info.get("MemAvailable",
                                info.get("MemFree", 0.0) + info.get("Cached", 0.0))
            used_kb = max(0.0, total_kb - avail_kb)
            pct = (100.0 * used_kb / total_kb) if total_kb > 0 else 0.0
            return pct, used_kb / 1024.0, total_kb / 1024.0, "proc"
        except Exception:  # noqa: BLE001
            return 0.0, 0.0, 0.0, "fallback"

    def _read_load(self) -> float:
        try:
            return float(os.getloadavg()[0])
        except Exception:  # noqa: BLE001 — not available on every platform
            return 0.0

    def _read_nprocs(self) -> int:
        if not self._has_proc:
            return 0
        try:
            return sum(1 for p in os.listdir("/proc") if p.isdigit())
        except Exception:  # noqa: BLE001
            return 0

    def _read_net(self) -> NetworkSample:
        """Cumulative packet/byte counters, summed over real interfaces (skipping lo)."""
        now = self.clock.now()
        try:
            rxb = txb = rxp = txp = 0
            with open("/proc/net/dev", "r", encoding="utf-8") as fh:
                lines = fh.readlines()[2:]   # skip the two header rows
            for line in lines:
                iface, _, data = line.partition(":")
                if iface.strip() == "lo":
                    continue
                cols = data.split()
                if len(cols) < 16:
                    continue
                rxb += int(cols[0]); rxp += int(cols[1])     # rx bytes, rx packets
                txb += int(cols[8]); txp += int(cols[9])     # tx bytes, tx packets
            return NetworkSample(at=now, rx_bytes=rxb, tx_bytes=txb,
                                 rx_packets=rxp, tx_packets=txp, source="proc")
        except Exception:  # noqa: BLE001
            return NetworkSample(at=now, source="fallback")

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def sample(self) -> HardwareSample:
        """Take a single hardware reading right now (also updates the CPU baseline)."""
        cpu, csrc = self._read_cpu_percent()
        mem_pct, used, total, msrc = self._read_mem()
        return HardwareSample(
            at=self.clock.now(), cpu_percent=cpu, mem_percent=mem_pct,
            mem_used_mb=used, mem_total_mb=total, load_avg_1m=self._read_load(),
            n_procs=self._read_nprocs(),
            source="proc" if "proc" in (csrc, msrc) else "fallback")

    def tick(self) -> HardwareSample:
        """One millisecond-scale beat: sample hardware + network into the rolling window."""
        hw = self.sample()
        self._hw.append(hw)
        self._net.append(self._read_net())
        self.ticks += 1
        return hw

    # ------------------------------------------------------------------ #
    # rolling aggregate + anomaly detection
    # ------------------------------------------------------------------ #
    def _packet_anomaly(self, rates: List[float]) -> bool:
        """A packet storm: the latest interval far exceeds the mean of the earlier ones."""
        if len(rates) < 3:
            return False
        last = rates[-1]
        prior = rates[:-1]
        mean = sum(prior) / len(prior)
        return last >= self.packet_spike_floor and last >= self.packet_spike_factor * max(mean, 1e-9)

    def aggregate(self) -> MicroAggregate:
        """Roll the current window up into one Layer-1 → Layer-2 message."""
        agg = MicroAggregate(at=self.clock.now())
        if not self._hw:
            return agg
        cpus = [s.cpu_percent for s in self._hw]
        mems = [s.mem_percent for s in self._hw]
        agg.n_samples = len(self._hw)
        agg.span_s = max(0.0, self._hw[-1].at - self._hw[0].at)
        agg.cpu_mean = sum(cpus) / len(cpus)
        agg.cpu_max = max(cpus)
        agg.mem_mean = sum(mems) / len(mems)
        agg.mem_max = max(mems)

        # network rates from the first→last cumulative counters over the window span
        packet_rates: List[float] = []
        if len(self._net) >= 2:
            first, last = self._net[0], self._net[-1]
            dt = max(1e-6, last.at - first.at)
            agg.rx_bytes_per_s = max(0.0, (last.rx_bytes - first.rx_bytes) / dt)
            agg.tx_bytes_per_s = max(0.0, (last.tx_bytes - first.tx_bytes) / dt)
            agg.rx_packets_per_s = max(0.0, (last.rx_packets - first.rx_packets) / dt)
            agg.tx_packets_per_s = max(0.0, (last.tx_packets - first.tx_packets) / dt)
            # per-interval packet rates for spike detection
            for a, b in zip(list(self._net)[:-1], list(self._net)[1:]):
                d = max(1e-6, b.at - a.at)
                dp = (b.rx_packets - a.rx_packets) + (b.tx_packets - a.tx_packets)
                packet_rates.append(max(0.0, dp / d))

        # anomalies — real, threshold-based health flags
        if agg.cpu_max >= self.cpu_alert:
            agg.anomalies.append(f"cpu spike {agg.cpu_max:.0f}% >= {self.cpu_alert:.0f}%")
        if agg.mem_max >= self.mem_alert:
            agg.anomalies.append(f"memory pressure {agg.mem_max:.0f}% >= {self.mem_alert:.0f}%")
        if self._packet_anomaly(packet_rates):
            agg.anomalies.append(
                f"packet storm {packet_rates[-1]:.0f} pkt/s (~{self.packet_spike_factor:.0f}x baseline)")

        self.aggregates += 1
        self._emit(agg)
        return agg

    def _emit(self, agg: MicroAggregate) -> None:
        """Best-effort: publish the aggregate (and any anomaly) on the event bus."""
        if self.bus is None:
            return
        try:
            emit = getattr(self.bus, "emit", None)
            if emit is None:
                return
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            payload = agg.to_dict()
            if loop is not None:
                loop.create_task(emit("temporal.micro.aggregate", payload))
                for a in agg.anomalies:
                    loop.create_task(emit("temporal.micro.anomaly", {"note": a, **payload}))
        except Exception:  # noqa: BLE001 — telemetry is never allowed to break the loop
            pass

    # ------------------------------------------------------------------ #
    def report(self) -> dict:
        last = self.aggregate()
        return {"name": "micro", "scale_s": self.scale_s, "ticks": self.ticks,
                "aggregates": self.aggregates, "window": len(self._hw),
                "has_proc": self._has_proc, "last": last.to_dict()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA temporal · MicroLayer self-test")
    print("=" * 70)
    m = MicroLayer(window=16)
    for _ in range(8):
        m.tick()
    agg = m.aggregate()
    print(f"samples taken       : {m.ticks}")
    print(f"sample              : {m.sample().to_dict()}")
    print(f"aggregate           : {agg.to_dict()}")
    assert m.ticks == 8
    assert agg.n_samples > 0
    print("\nALL SELF-TESTS PASSED ✓")

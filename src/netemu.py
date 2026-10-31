"""Application-layer, trace-driven link emulation over real TCP sockets.

The analytic model in network.py turns logged bits into time with a formula.
This module checks that formula against real transport: for every round of a
finished run it opens one loopback TCP connection per transmitting client and
pushes exactly the number of bytes that client uploaded in that round, with
the sender paced by a token bucket at the profile's uplink rate and delayed by
the profile's one-way latency. Clients transmit concurrently on independent
links (one paced connection each), and the round ends when the server has
received the last byte from the slowest client. Wall-clock time is measured
with a monotonic clock, so TCP handshakes, segmentation, socket buffering and
OS scheduling are included rather than modelled.

What it is NOT: a kernel-level emulator (tc/netem), a radio model (no
scheduling, fading, HARQ or loss), or a downlink model. Uplink only, exactly the
quantity the analytic model costs, so the two can be compared row by row.
Combined with each run's per-round compute time and AUROC trajectory it gives a
wall-clock time-to-target-accuracy.
"""
from __future__ import annotations

import socket
import threading
import time
from typing import Dict, List, Mapping, Sequence

CHUNK = 64 * 1024


def _recv_all(conn: socket.socket, n: int) -> None:
    got = 0
    while got < n:
        b = conn.recv(min(1 << 20, n - got))
        if not b:
            raise ConnectionError("peer closed early")
        got += len(b)


def _paced_send(port: int, nbytes: int, rate_bps: float, delay_s: float,
                start: float, errors: List[str]) -> None:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=600) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            payload = b"\x5a" * CHUNK
            # one-way propagation delay before the first byte leaves
            while time.perf_counter() < start + delay_s:
                time.sleep(max(0.0, start + delay_s - time.perf_counter()))
            t0 = time.perf_counter()
            sent = 0
            while sent < nbytes:
                k = min(CHUNK, nbytes - sent)
                s.sendall(payload[:k])
                sent += k
                # token bucket: never ahead of rate * elapsed
                due = t0 + sent * 8.0 / rate_bps
                while True:
                    lag = due - time.perf_counter()
                    if lag <= 0:
                        break
                    time.sleep(lag)
            s.shutdown(socket.SHUT_WR)
    except Exception as e:                           # surfaced by the caller
        errors.append(repr(e))


def emulate_round(client_bytes: Sequence[int], rate_bps: float, delay_s: float) -> float:
    """Real seconds until the server holds every byte of every client's upload."""
    sizes = [int(b) for b in client_bytes if b > 0]
    if not sizes:
        return 0.0
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0)); srv.listen(len(sizes))
    port = srv.getsockname()[1]
    errors: List[str] = []
    done: List[float] = []
    lock = threading.Lock()

    def serve(conn):
        with conn:
            # the sender announces nothing: the receiver reads to EOF
            total = 0
            while True:
                b = conn.recv(1 << 20)
                if not b:
                    break
                total += len(b)
            with lock:
                done.append(time.perf_counter())

    start = time.perf_counter()
    senders = [threading.Thread(target=_paced_send, args=(port, n, rate_bps, delay_s, start, errors), daemon=True)
               for n in sizes]
    for t in senders:
        t.start()
    servers = []
    for _ in sizes:
        conn, _ = srv.accept()
        th = threading.Thread(target=serve, args=(conn,), daemon=True); th.start(); servers.append(th)
    for t in senders + servers:
        t.join()
    srv.close()
    if errors:
        raise RuntimeError("; ".join(errors))
    return max(done) - start


def emulate_run(round_bits: List[Mapping], rate_mbps: float, rtt_ms: float) -> List[float]:
    """Per-round emulated upload time for a run's logged `round_bits`.
    Uses the per-client bit counts when logged, else n_senders equal uploads."""
    out = []
    for rb in round_bits:
        if rb.get("upload_bits", 0) <= 0:
            out.append(0.0); continue
        cb = rb.get("client_bits")
        if cb is None:
            n = int(rb.get("n_senders", 1)) or 1
            cb = [rb["upload_bits"] / n] * n
        out.append(emulate_round([b / 8.0 for b in cb], rate_mbps * 1e6, rtt_ms / 2e3))
    return out


def time_to_target(history: List[Mapping], round_bits: List[Mapping], upload_s: Sequence[float],
                   target: float, metric: str = "fed_auroc") -> Dict[str, float]:
    """Wall-clock (compute of the slowest client + emulated upload) until the
    federated metric first reaches `target`; inf if never."""
    by_round = {int(h["round"]): h.get(metric) for h in history}
    t = 0.0
    for rb, up in zip(round_bits, upload_s):
        t += float(rb.get("compute_s_max", 0.0)) + float(up)
        v = by_round.get(int(rb["round"]))
        if v is not None and v == v and v >= target:
            return {"time_to_target_s": t, "round_to_target": float(rb["round"])}
    return {"time_to_target_s": float("inf"), "round_to_target": float("inf")}

const WORKER_SCRIPT = `
'use strict';

// Standard SHA-256 round constants
const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]);

// Pre-allocated buffers per worker (zero GC during stress)
const F64_LEN = 131072; // 1 MB Float64Array (L2/L3 cache pressure)
const F64_MASK = F64_LEN - 1;
const f64Buf = new Float64Array(F64_LEN);

const U32_LEN = 16384; // 64 KB Uint32Array (L1/L2 integer pressure)
const U32_MASK = U32_LEN - 1;
const u32Buf = new Uint32Array(U32_LEN);

const MAT_N = 64;
const matA = new Float64Array(MAT_N * MAT_N);
const matB = new Float64Array(MAT_N * MAT_N);
const matC = new Float64Array(MAT_N * MAT_N);

const shaW = new Uint32Array(64);
const shaH = new Uint32Array([
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]);

let workerId = 0;
let running = false;
let workload = 'extreme';
let duty = 1.0;

// Seed buffers deterministically per worker
function initBuffers(seed) {
  let s = (seed + 1) * 0x9e3779b9;
  for (let i = 0; i < F64_LEN; i++) {
    s = (Math.imul(s ^ (s >>> 15), 0x85ebca6b) + 0x1234567) >>> 0;
    f64Buf[i] = 0.5 + (s & 0xffff) * 1.52587890625e-5;
  }
  for (let i = 0; i < U32_LEN; i++) {
    s = (Math.imul(s ^ (s >>> 13), 0xc2b2ae35) + i) >>> 0;
    u32Buf[i] = s;
  }
  for (let i = 0; i < MAT_N * MAT_N; i++) {
    matA[i] = 0.25 + ((i * 17 + seed) % 97) * 0.01;
    matB[i] = 0.25 + ((i * 31 + seed) % 89) * 0.01;
    matC[i] = 0.1;
  }
  shaH[0] = (0x6a09e667 ^ (seed * 0x9e3779b1)) >>> 0;
}

// Kernel 1: Extreme Mix — saturates FPU FMA, transcendental units, Integer ALU, and L1/L2/L3 cache ports simultaneously
let extIdx = 0;
let sink = 0;

function runExtreme(rounds) {
  let a0 = 1.1234567, a1 = 1.2345678, a2 = 1.3456789, a3 = 1.4567891;
  let a4 = 0.9876543, a5 = 0.8765432, a6 = 0.7654321, a7 = 0.6543219;
  let i0 = (0x13579bdf ^ extIdx) >>> 0;
  let i1 = 0x2468ace1 >>> 0;
  let i2 = 0x9e3779b9 >>> 0;
  let i3 = 0x85ebca6b >>> 0;
  let idx = extIdx;
  const fBuf = f64Buf;
  const uBuf = u32Buf;

  for (let r = 0; r < rounds; r++) {
    // 8-way independent unrolled FP64 FMA + cache read/write (strictly bounded contraction)
    const p0 = idx & F64_MASK;
    const p1 = (idx + 17) & F64_MASK;
    const p2 = (idx + 67) & F64_MASK;
    const p3 = (idx + 257) & F64_MASK;

    const m0 = fBuf[p0];
    const m1 = fBuf[p1];
    const m2 = fBuf[p2];
    const m3 = fBuf[p3];

    a0 = a0 * 0.4999993 + m0 * 0.483121 + 0.17;
    a1 = a1 * 0.4999989 + m1 * 0.471983 + 0.23;
    a2 = a2 * 0.4999997 + m2 * 0.491237 + 0.19;
    a3 = a3 * 0.4999981 + m3 * 0.468719 + 0.21;

    a4 = a4 * 0.4123 + a0 * m0 * 0.3511 + 0.13;
    a5 = a5 * 0.4271 + a1 * m1 * 0.3419 + 0.17;
    a6 = a6 * 0.4319 + a2 * m2 * 0.3347 + 0.11;
    a7 = a7 * 0.4193 + a3 * m3 * 0.3483 + 0.19;

    // Transcendental / sqrt stress (keeps SFU/div/sqrt pipes busy)
    a0 = Math.sqrt(a0 * a4 + 1.0001);
    a1 = Math.sin(a1) + Math.cos(a5) + 2.15;
    a2 = Math.sqrt(a2 * a6 + 1.0002);
    a3 = Math.cos(a3) + Math.sin(a7) + 2.15;

    // Integer 32-bit ALU / multiplier / bit-rotate mix
    const uIdx = idx & U32_MASK;
    const uv = uBuf[uIdx];
    i0 = (Math.imul(i0 ^ uv, 0x85ebca6b) + ((i1 << 13) | (i1 >>> 19))) >>> 0;
    i1 = (Math.imul(i1 ^ i0, 0xc2b2ae35) + ((i2 << 17) | (i2 >>> 15))) >>> 0;
    i2 = (Math.imul(i2 ^ i1, 0x27d4eb2d) + ((i3 << 7) | (i3 >>> 25))) >>> 0;
    i3 = (Math.imul(i3 ^ i2, 0x165667b1) + ((i0 << 11) | (i0 >>> 21))) >>> 0;

    // Write back bounded values to cache lines (dirty writeback + prevents dead-code elimination)
    fBuf[p0] = (a0 + a4) * 0.22;
    fBuf[p2] = (a2 + a6) * 0.22;
    uBuf[(uIdx + 31) & U32_MASK] = i0 ^ i3;

    idx = (idx + 73) & F64_MASK;
  }

  extIdx = idx;
  sink = (sink + a0 + a1 + a2 + a3 + (i0 & 0xff) * 1e-4) % 10000;
  // ~48 effective ALU/FPU ops per round
  return rounds * 48;
}

// Kernel 2: Dense FP64 Blocked Matrix Multiply + FMA Heater
let fpPhase = 0.0;
function runFp64(blocks) {
  const N = MAT_N;
  const A = matA;
  const B = matB;
  const C = matC;
  let totalOps = 0;

  for (let rep = 0; rep < blocks; rep++) {
    // 4x4 unrolled inner kernel across a 64x64 matrix multiply
    for (let i = 0; i < N; i += 4) {
      const i0 = i * N;
      const i1 = (i + 1) * N;
      const i2 = (i + 2) * N;
      const i3 = (i + 3) * N;

      for (let j = 0; j < N; j += 4) {
        let c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
        let c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
        let c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
        let c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;

        for (let k = 0; k < N; k++) {
          const a0 = A[i0 + k];
          const a1 = A[i1 + k];
          const a2 = A[i2 + k];
          const a3 = A[i3 + k];

          const kRow = k * N + j;
          const b0 = B[kRow];
          const b1 = B[kRow + 1];
          const b2 = B[kRow + 2];
          const b3 = B[kRow + 3];

          c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
          c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
          c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
          c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        }

        const s = 0.015625; // 1/64 keeps values strictly normalized
        C[i0 + j] = c00 * s; C[i0 + j + 1] = c01 * s; C[i0 + j + 2] = c02 * s; C[i0 + j + 3] = c03 * s;
        C[i1 + j] = c10 * s; C[i1 + j + 1] = c11 * s; C[i1 + j + 2] = c12 * s; C[i1 + j + 3] = c13 * s;
        C[i2 + j] = c20 * s; C[i2 + j + 1] = c21 * s; C[i2 + j + 2] = c22 * s; C[i2 + j + 3] = c23 * s;
        C[i3 + j] = c30 * s; C[i3 + j + 1] = c31 * s; C[i3 + j + 2] = c32 * s; C[i3 + j + 3] = c33 * s;
      }
    }

    // Slight non-linear feedback into A to prevent identical cache patterns
    fpPhase = (fpPhase + 0.013) % 628.3185;
    const perturb = Math.sin(fpPhase) * 0.05 + 0.5;
    for (let i = 0; i < N; i++) {
      A[i * N + ((i + rep) & 63)] = Math.sqrt(Math.abs(C[i * N + i]) + 0.25) * perturb;
    }
    totalOps += 2 * N * N * N; // 2*64^3 = 524,288 FLOPs per GEMM
  }

  sink = (sink + C[0] + C[MAT_N * MAT_N - 1]) % 10000;
  return totalOps;
}

// Kernel 3: Real 64-Round SHA-256 Compression Loop (Pure Integer Bitwise ALU Heater)
let shaNonce = 0;
function runSha256(hashes) {
  const W = shaW;
  const K = SHA256_K;
  let h0 = shaH[0] | 0, h1 = shaH[1] | 0, h2 = shaH[2] | 0, h3 = shaH[3] | 0;
  let h4 = shaH[4] | 0, h5 = shaH[5] | 0, h6 = shaH[6] | 0, h7 = shaH[7] | 0;
  let nonce = shaNonce | 0;

  for (let n = 0; n < hashes; n++) {
    nonce = (nonce + 1) | 0;
    W[0] = h0 ^ nonce;
    W[1] = h1; W[2] = h2; W[3] = h3;
    W[4] = h4; W[5] = h5; W[6] = h6; W[7] = h7;
    W[8] = 0x80000000 | 0;
    W[9] = nonce ^ 0x5a827999;
    W[10] = 0x6ed9eba1; W[11] = 0x8f1bbcdc;
    W[12] = 0xca62c1d6; W[13] = h0 + h4;
    W[14] = 0; W[15] = 0x00000100;

    for (let i = 16; i < 64; i++) {
      const w15 = W[i - 15];
      const s0 = ((w15 >>> 7) | (w15 << 25)) ^ ((w15 >>> 18) | (w15 << 14)) ^ (w15 >>> 3);
      const w2 = W[i - 2];
      const s1 = ((w2 >>> 17) | (w2 << 15)) ^ ((w2 >>> 19) | (w2 << 13)) ^ (w2 >>> 10);
      W[i] = (W[i - 16] + s0 + W[i - 7] + s1) | 0;
    }

    let a = h0, b = h1, c = h2, d = h3;
    let e = h4, f = h5, g = h6, h = h7;

    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + K[i] + W[i]) | 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) | 0;

      h = g; g = f; f = e; e = (d + temp1) | 0;
      d = c; c = b; b = a; a = (temp1 + temp2) | 0;
    }

    h0 = (h0 + a) | 0; h1 = (h1 + b) | 0; h2 = (h2 + c) | 0; h3 = (h3 + d) | 0;
    h4 = (h4 + e) | 0; h5 = (h5 + f) | 0; h6 = (h6 + g) | 0; h7 = (h7 + h) | 0;
  }

  shaH[0] = h0 >>> 0; shaH[1] = h1 >>> 0; shaH[2] = h2 >>> 0; shaH[3] = h3 >>> 0;
  shaH[4] = h4 >>> 0; shaH[5] = h5 >>> 0; shaH[6] = h6 >>> 0; shaH[7] = h7 >>> 0;
  shaNonce = nonce;
  sink = (sink + (h0 & 0xff) * 1e-4) % 10000;
  return hashes; // exact SHA-256 block hashes
}

// Adaptive chunk sizing (~75ms per slice so yield overhead is <0.02% while staying stoppable)
let chunkUnits = { extreme: 180000, fp64: 24, crypto: 18000 };
let reportOps = 0;
let reportWorkMs = 0;
let lastReportTs = 0;

const channel = new MessageChannel();
channel.port1.onmessage = step;

function scheduleNext(workMs) {
  if (!running) return;
  if (duty >= 0.99) {
    // Sub-10-microsecond yield — zero 4ms setTimeout clamp -> 99.9%+ CPU saturation
    channel.port2.postMessage(0);
  } else {
    const clampedDuty = Math.max(0.1, Math.min(0.95, duty));
    const sleepMs = Math.max(4, workMs * ((1 - clampedDuty) / clampedDuty));
    setTimeout(step, sleepMs);
  }
}

function step() {
  if (!running) return;
  const t0 = performance.now();
  const mode = workload;
  const units = chunkUnits[mode] || 10000;

  let ops = 0;
  if (mode === 'extreme') {
    ops = runExtreme(units);
  } else if (mode === 'fp64') {
    ops = runFp64(units);
  } else {
    ops = runSha256(units);
  }

  const t1 = performance.now();
  const workMs = Math.max(0.1, t1 - t0);

  // Calibrate chunk size toward ~75ms target slice
  const targetMs = duty >= 0.99 ? 80 : 45;
  if (workMs < targetMs * 0.55) {
    chunkUnits[mode] = Math.min(units * 2, mode === 'fp64' ? 4096 : 20000000);
  } else if (workMs > targetMs * 1.7 && units > 2) {
    chunkUnits[mode] = Math.max(mode === 'fp64' ? 1 : 256, Math.floor(units * 0.65));
  }

  reportOps += ops;
  reportWorkMs += workMs;

  if (t1 - lastReportTs >= 420) {
    const wallMs = Math.max(1, t1 - lastReportTs);
    self.postMessage({
      type: 'stats',
      workerId,
      mode,
      ops: reportOps,
      workMs: reportWorkMs,
      wallMs,
      checksum: sink
    });
    reportOps = 0;
    reportWorkMs = 0;
    lastReportTs = t1;
  }

  scheduleNext(workMs);
}

self.onmessage = function (e) {
  const msg = e.data;
  if (!msg) return;
  if (msg.type === 'start') {
    workerId = msg.workerId || 0;
    workload = msg.workload || 'extreme';
    duty = typeof msg.duty === 'number' ? msg.duty : 1.0;
    initBuffers(workerId);
    running = true;
    reportOps = 0;
    reportWorkMs = 0;
    lastReportTs = performance.now();
    channel.port2.postMessage(0);
  } else if (msg.type === 'config') {
    if (msg.workload) workload = msg.workload;
    if (typeof msg.duty === 'number') duty = msg.duty;
  } else if (msg.type === 'stop') {
    running = false;
  }
};
`;

export class CpuStressEngine {
  constructor(onTelemetry) {
    this.onTelemetry = onTelemetry;
    this.workers = [];
    this.workerStats = new Map();
    this.running = false;
    this.blobUrl = null;
    this.reportTimer = null;

    this.config = {
      threads: Math.max(1, navigator.hardwareConcurrency || 8),
      workload: 'extreme',
      duty: 1.0
    };

    // Session aggregates
    this.peakRate = 0;
    this.samples = [];
  }

  _ensureBlobUrl() {
    if (!this.blobUrl) {
      const blob = new Blob([WORKER_SCRIPT], { type: 'application/javascript' });
      this.blobUrl = URL.createObjectURL(blob);
    }
    return this.blobUrl;
  }

  start(config = {}) {
    this.config = { ...this.config, ...config };
    if (this.running) {
      this.updateConfig(config);
      return;
    }

    this.running = true;
    this.peakRate = 0;
    this.samples = [];
    this.workerStats.clear();

    this._syncWorkerCount(this.config.threads);

    this.reportTimer = setInterval(() => {
      this._emitTelemetry();
    }, 500);
  }

  updateConfig(partial = {}) {
    const prevThreads = this.config.threads;
    const prevWorkload = this.config.workload;
    const prevDuty = this.config.duty;
    this.config = { ...this.config, ...partial };

    if (!this.running) return;

    if (this.config.workload !== prevWorkload) {
      // Reset peak when switching unit/kernel
      this.peakRate = 0;
      this.samples = [];
    }

    if (this.config.threads !== prevThreads) {
      this._syncWorkerCount(this.config.threads);
    }

    if (this.config.workload !== prevWorkload || this.config.duty !== prevDuty) {
      for (const w of this.workers) {
        w.postMessage({
          type: 'config',
          workload: this.config.workload,
          duty: this.config.duty
        });
      }
    }
  }

  _syncWorkerCount(targetCount) {
    const clamped = Math.max(1, Math.min(256, Math.round(targetCount)));
    const url = this._ensureBlobUrl();

    // Terminate excess workers immediately
    while (this.workers.length > clamped) {
      const w = this.workers.pop();
      this.workerStats.delete(w._kiiId);
      try {
        w.terminate();
      } catch (_) {}
    }

    // Spawn needed workers
    while (this.workers.length < clamped) {
      const id = this.workers.length;
      const worker = new Worker(url);
      worker._kiiId = id;
      worker.onmessage = (e) => {
        const data = e.data;
        if (!data || data.type !== 'stats') return;
        const ratePerSec = data.ops / (data.wallMs / 1000);
        const saturation = Math.min(100, (data.workMs / data.wallMs) * 100);
        this.workerStats.set(data.workerId, {
          ratePerSec,
          saturation,
          mode: data.mode,
          ts: performance.now()
        });
      };
      worker.onerror = () => {
        // Ignore terminated/error worker cleanly
      };
      worker.postMessage({
        type: 'start',
        workerId: id,
        workload: this.config.workload,
        duty: this.config.duty
      });
      this.workers.push(worker);
    }
  }

  _emitTelemetry() {
    if (!this.running) return;
    const now = performance.now();
    let totalRate = 0;
    let totalSat = 0;
    let reportingWorkers = 0;

    for (const [, stat] of this.workerStats) {
      // Only include fresh reports (<2.5s old) and matching workload
      if (now - stat.ts < 2500 && stat.mode === this.config.workload) {
        totalRate += stat.ratePerSec;
        totalSat += stat.saturation;
        reportingWorkers++;
      }
    }

    // Convert ops/sec to MOPS (Million ops/sec) or MH/s (Mega hashes/sec)
    const rateMillions = totalRate / 1e6;
    if (rateMillions > this.peakRate) {
      this.peakRate = rateMillions;
    }
    if (reportingWorkers > 0) {
      this.samples.push(rateMillions);
      if (this.samples.length > 7200) this.samples.shift();
    }

    const avgSat = reportingWorkers > 0 ? totalSat / reportingWorkers : 0;

    if (this.onTelemetry) {
      this.onTelemetry({
        running: this.running,
        workload: this.config.workload,
        rateMillions,
        peakMillions: this.peakRate,
        activeWorkers: this.workers.length,
        reportingWorkers,
        saturationPct: Math.min(100, avgSat)
      });
    }
  }

  getSummary() {
    const avg =
      this.samples.length > 0
        ? this.samples.reduce((a, b) => a + b, 0) / this.samples.length
        : 0;
    return {
      workload: this.config.workload,
      avgMillions: avg,
      peakMillions: this.peakRate
    };
  }

  stop() {
    this.running = false;
    if (this.reportTimer) {
      clearInterval(this.reportTimer);
      this.reportTimer = null;
    }
    for (const w of this.workers) {
      try {
        w.terminate();
      } catch (_) {}
    }
    this.workers = [];
    this.workerStats.clear();
  }

  destroy() {
    this.stop();
    if (this.blobUrl) {
      URL.revokeObjectURL(this.blobUrl);
      this.blobUrl = null;
    }
  }
}

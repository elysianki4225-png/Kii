// Multi-Worker Physical Memory (RAM) Stress, Bandwidth & MemTest Verification Engine
// Distributes 32 MB Uint32Array blocks across Web Workers to bypass per-isolate limits
// and saturate multi-channel DDR4/DDR5/LPDDR5X memory controllers.

const RAM_WORKER_SCRIPT = `
'use strict';

const CHUNK_BYTES = 32 * 1024 * 1024; // 32 MB per block
const CHUNK_WORDS = CHUNK_BYTES >>> 2; // 8,388,608 uint32 words

let workerId = 0;
let running = false;
let mode = 'bandwidth'; // 'bandwidth' | 'verify' | 'hammer'
let targetMB = 256;
let blocks = [];
let allocatedMB = 0;
let oomHit = false;

let passNum = 0;
let verifyPatternIdx = 0;
let blockCursor = 0;
let totalErrors = 0;
let reportBytes = 0;
let reportVerifiedBytes = 0;
let reportPasses = 0;
let lastReportTs = 0;
let sink = 0;

const channel = new MessageChannel();
channel.port1.onmessage = step;

function commitPhysicalPages(arr, seed) {
  // Touch every 4 KB OS memory page (1024 uint32s) so OS truly commits physical RAM
  let v = (seed * 0x9e3779b1 + 0x5a5a5a5a) >>> 0;
  for (let i = 0; i < CHUNK_WORDS; i += 1024) {
    v = (Math.imul(v ^ i, 0x85ebca6b) + 0x1234567) >>> 0;
    arr[i] = v;
  }
}

function allocateBlocks(mb) {
  blocks = [];
  allocatedMB = 0;
  oomHit = false;
  const needed = Math.max(1, Math.ceil(mb / 32));

  for (let b = 0; b < needed; b++) {
    if (!running) return;
    try {
      const arr = new Uint32Array(CHUNK_WORDS);
      commitPhysicalPages(arr, (workerId << 12) + b + 1);
      blocks.push(arr);
      allocatedMB += 32;
      if ((b & 3) === 3 || b === needed - 1) {
        self.postMessage({
          type: 'alloc',
          workerId,
          allocatedMB,
          oomHit: false
        });
      }
    } catch (_) {
      oomHit = true;
      break;
    }
  }

  self.postMessage({
    type: 'alloc',
    workerId,
    allocatedMB,
    oomHit
  });
}

// Mode 1: Max Bandwidth — native memcpy (TypedArray.set) + unrolled cache-line RMW
function runBandwidthStep() {
  const n = blocks.length;
  if (n === 0) return;

  const idxA = blockCursor % n;
  const idxB = (blockCursor + 1) % n;
  const src = blocks[idxA];
  const dst = blocks[idxB];
  blockCursor++;

  if (n >= 2) {
    // Native SIMD memcpy inside V8: reads 32 MB + writes 32 MB = 64 MB bus traffic
    dst.set(src);
    reportBytes += CHUNK_BYTES * 2;
  }

  // Unrolled cache-line read-modify-write pass (16 words = 64-byte cache line)
  // Forces dirty cache line writeback & full read+write DRAM channel saturation
  const mask = (passNum * 0x9e3779b9 + workerId * 17) >>> 0;
  let acc = 0;
  for (let i = 0; i < CHUNK_WORDS; i += 16) {
    const v0 = (dst[i] ^ mask) >>> 0;
    const v4 = (dst[i + 4] + 0x1010101) >>> 0;
    const v8 = (dst[i + 8] ^ v0) >>> 0;
    const v12 = (dst[i + 12] + v4) >>> 0;
    dst[i] = v0;
    dst[i + 4] = v4;
    dst[i + 8] = v8;
    dst[i + 12] = v12;
    acc = (acc + v8 + v12) >>> 0;
  }
  // Every 64-byte cache line read & dirtied = 32 MB read + 32 MB write = 64 MB
  reportBytes += CHUNK_BYTES * 2;
  sink = (sink + acc) >>> 0;

  if (blockCursor >= n) {
    blockCursor = 0;
    passNum++;
    reportPasses++;
  }
}

// Mode 2: MemTest Pattern Verification — monomorphic V8-optimal static & PRNG verifiers
const STATIC_PATTERNS = [
  0x55555555,
  0xAAAAAAAA,
  0x00000000,
  0xFFFFFFFF,
  0x0F0F0F0F,
  0xF0F0F0F0
];

function verifyStaticBlock(arr, pat) {
  const len = arr.length | 0;
  arr.fill(pat);
  let errs = 0;
  for (let i = 0; i < len; i += 8) {
    if (
      arr[i] !== pat || arr[i + 1] !== pat || arr[i + 2] !== pat || arr[i + 3] !== pat ||
      arr[i + 4] !== pat || arr[i + 5] !== pat || arr[i + 6] !== pat || arr[i + 7] !== pat
    ) {
      for (let j = 0; j < 8; j++) {
        if (arr[i + j] !== pat) errs++;
      }
    }
  }
  return errs;
}

function verifyPrngBlock(arr, seedIn) {
  const len = arr.length | 0;
  const seed = seedIn | 0;
  const imul = Math.imul;
  for (let i = 0; i < len; i += 4) {
    arr[i] = imul(i ^ seed, 0x85ebca6b) >>> 0;
    arr[i + 1] = imul((i + 1) ^ seed, 0x85ebca6b) >>> 0;
    arr[i + 2] = imul((i + 2) ^ seed, 0x85ebca6b) >>> 0;
    arr[i + 3] = imul((i + 3) ^ seed, 0x85ebca6b) >>> 0;
  }
  let errs = 0;
  for (let i = 0; i < len; i += 4) {
    if (arr[i] !== (imul(i ^ seed, 0x85ebca6b) >>> 0)) errs++;
    if (arr[i + 1] !== (imul((i + 1) ^ seed, 0x85ebca6b) >>> 0)) errs++;
    if (arr[i + 2] !== (imul((i + 2) ^ seed, 0x85ebca6b) >>> 0)) errs++;
    if (arr[i + 3] !== (imul((i + 3) ^ seed, 0x85ebca6b) >>> 0)) errs++;
  }
  return errs;
}

function runVerifyStep() {
  const n = blocks.length;
  if (n === 0) return;

  const bIdx = blockCursor % n;
  const arr = blocks[bIdx];
  const patType = verifyPatternIdx % 7;

  let errs = 0;
  if (patType < 6) {
    errs = verifyStaticBlock(arr, STATIC_PATTERNS[patType] >>> 0);
  } else {
    const seed = (Math.imul(passNum + 1, 0x9e3779b1) + workerId * 131 + bIdx * 17) | 0;
    errs = verifyPrngBlock(arr, seed);
  }

  totalErrors += errs;
  reportBytes += CHUNK_BYTES * 2;
  reportVerifiedBytes += CHUNK_BYTES;

  blockCursor++;
  if (blockCursor >= n) {
    blockCursor = 0;
    verifyPatternIdx++;
    passNum++;
    reportPasses++;
  }
}

// Mode 3: Random Latency / Row-Hammer Cache-Thrash — defeats hardware prefetchers completely
let hammerRng = 0x13579bdf;
function runHammerStep() {
  const n = blocks.length;
  if (n === 0) return;

  const wordMask = CHUNK_WORDS - 1;
  let rng = (hammerRng ^ (workerId * 0x9e3779b9)) >>> 0;
  let acc = 0;
  const ops = 655360; // ~655k random cache-line RMW touches across ALL allocated GB

  for (let i = 0; i < ops; i++) {
    // xorshift32 PRNG — unpredictable page/row jumps across all blocks
    rng ^= rng << 13;
    rng ^= rng >>> 17;
    rng ^= rng << 5;
    const u = rng >>> 0;
    const bIdx = u % n;
    // Align to 16-word (64-byte) cache line boundary
    const wIdx = ((u >>> 4) << 4) & wordMask;
    const blk = blocks[bIdx];
    const oldVal = blk[wIdx];
    const newVal = (oldVal ^ u) >>> 0;
    blk[wIdx] = newVal;
    blk[(wIdx + 8) & wordMask] = (blk[(wIdx + 8) & wordMask] + newVal) >>> 0;
    acc = (acc + newVal) >>> 0;
  }

  hammerRng = rng;
  sink = (sink + acc) >>> 0;
  // Each random miss pulls & dirties a full 64-byte DRAM cache line (read 64B + writeback 64B = 128B)
  reportBytes += ops * 128;
  blockCursor++;
  if (blockCursor >= Math.max(4, n)) {
    blockCursor = 0;
    passNum++;
    reportPasses++;
  }
}

function step() {
  if (!running) return;
  if (blocks.length === 0) return;

  if (mode === 'bandwidth') {
    runBandwidthStep();
  } else if (mode === 'verify') {
    runVerifyStep();
  } else {
    runHammerStep();
  }

  const now = performance.now();
  if (now - lastReportTs >= 420) {
    const wallMs = Math.max(1, now - lastReportTs);
    self.postMessage({
      type: 'stats',
      workerId,
      mode,
      allocatedMB,
      oomHit,
      bytes: reportBytes,
      verifiedBytes: reportVerifiedBytes,
      passesDelta: reportPasses,
      errors: totalErrors,
      wallMs,
      checksum: sink
    });
    reportBytes = 0;
    reportVerifiedBytes = 0;
    reportPasses = 0;
    lastReportTs = now;
  }

  if (running) {
    channel.port2.postMessage(0);
  }
}

self.onmessage = function (e) {
  const msg = e.data;
  if (!msg) return;
  if (msg.type === 'start') {
    workerId = msg.workerId || 0;
    mode = msg.mode || 'bandwidth';
    targetMB = Math.max(32, msg.targetMB || 256);
    running = true;
    totalErrors = 0;
    passNum = 0;
    verifyPatternIdx = 0;
    blockCursor = 0;
    reportBytes = 0;
    reportVerifiedBytes = 0;
    reportPasses = 0;

    allocateBlocks(targetMB);
    if (!running || blocks.length === 0) return;
    lastReportTs = performance.now();
    channel.port2.postMessage(0);
  } else if (msg.type === 'config') {
    if (msg.mode) mode = msg.mode;
  } else if (msg.type === 'stop') {
    running = false;
    blocks = [];
    allocatedMB = 0;
  }
};
`;

export class RamStressEngine {
  constructor(onTelemetry) {
    this.onTelemetry = onTelemetry;
    this.workers = [];
    this.workerStats = new Map();
    this.workerAllocMB = new Map();
    this.running = false;
    this.allocating = false;
    this.blobUrl = null;
    this.reportTimer = null;

    const cores = Math.max(1, navigator.hardwareConcurrency || 8);
    this.config = {
      targetMB: 2048, // Default 2 GB — instantly noticeable in Task Manager & safe on 8GB+ machines
      mode: 'bandwidth', // 'bandwidth' | 'verify' | 'hammer'
      threads: Math.max(2, Math.min(16, cores))
    };

    // Session totals
    this.peakGBs = 0;
    this.totalVerifiedGB = 0;
    this.totalPasses = 0;
    this.totalErrors = 0;
    this.oomHit = false;
    this.samplesGBs = [];
  }

  _ensureBlobUrl() {
    if (!this.blobUrl) {
      const blob = new Blob([RAM_WORKER_SCRIPT], { type: 'application/javascript' });
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
    this.allocating = true;
    this.peakGBs = 0;
    this.totalVerifiedGB = 0;
    this.totalPasses = 0;
    this.totalErrors = 0;
    this.oomHit = false;
    this.samplesGBs = [];
    this.workerStats.clear();
    this.workerAllocMB.clear();

    this._spawnWorkers();

    this.reportTimer = setInterval(() => {
      this._emitTelemetry();
    }, 500);
  }

  updateConfig(partial = {}) {
    const prevTarget = this.config.targetMB;
    const prevThreads = this.config.threads;
    const prevMode = this.config.mode;
    this.config = { ...this.config, ...partial };

    if (!this.running) return;

    // Reallocate if capacity or thread count changes
    if (this.config.targetMB !== prevTarget || this.config.threads !== prevThreads) {
      this.stop();
      this.start(this.config);
      return;
    }

    if (this.config.mode !== prevMode) {
      this.peakGBs = 0;
      this.samplesGBs = [];
      for (const w of this.workers) {
        w.postMessage({ type: 'config', mode: this.config.mode });
      }
    }
  }

  _spawnWorkers() {
    const url = this._ensureBlobUrl();
    const maxPossibleWorkers = Math.max(1, Math.floor(this.config.targetMB / 32));
    const count = Math.max(1, Math.min(32, this.config.threads, maxPossibleWorkers));
    const basePerWorkerMB = Math.floor((this.config.targetMB / count) / 32) * 32;
    let remainderChunks = Math.max(0, Math.round((this.config.targetMB - basePerWorkerMB * count) / 32));

    for (let i = 0; i < count; i++) {
      let workerMB = Math.max(32, basePerWorkerMB);
      if (remainderChunks > 0) {
        workerMB += 32;
        remainderChunks--;
      }

      const worker = new Worker(url);
      worker._kiiId = i;
      worker.onmessage = (e) => {
        const data = e.data;
        if (!data) return;

        if (data.type === 'alloc') {
          this.workerAllocMB.set(data.workerId, data.allocatedMB || 0);
          if (data.oomHit) this.oomHit = true;
          this._emitTelemetry();
        } else if (data.type === 'stats') {
          this.allocating = false;
          this.workerAllocMB.set(data.workerId, data.allocatedMB || 0);
          if (data.oomHit) this.oomHit = true;

          const bytesPerSec = data.bytes / (data.wallMs / 1000);
          this.totalVerifiedGB += (data.verifiedBytes || 0) / (1024 * 1024 * 1024);
          this.totalPasses += data.passesDelta || 0;

          this.workerStats.set(data.workerId, {
            bytesPerSec,
            errors: data.errors || 0,
            mode: data.mode,
            ts: performance.now()
          });
        }
      };
      worker.onerror = () => {
        this.oomHit = true;
      };
      worker.postMessage({
        type: 'start',
        workerId: i,
        mode: this.config.mode,
        targetMB: workerMB
      });
      this.workers.push(worker);
    }
  }

  _emitTelemetry() {
    if (!this.running) return;
    const now = performance.now();
    let totalBytesPerSec = 0;
    let totalErrors = 0;
    let reporting = 0;

    for (const [, stat] of this.workerStats) {
      totalErrors += stat.errors || 0;
      if (now - stat.ts < 2500 && stat.mode === this.config.mode) {
        totalBytesPerSec += stat.bytesPerSec;
        reporting++;
      }
    }

    let allocatedMB = 0;
    for (const [, mb] of this.workerAllocMB) {
      allocatedMB += mb;
    }

    const gbs = totalBytesPerSec / (1024 * 1024 * 1024);
    if (gbs > this.peakGBs) this.peakGBs = gbs;
    if (reporting > 0) {
      this.samplesGBs.push(gbs);
      if (this.samplesGBs.length > 7200) this.samplesGBs.shift();
    }
    this.totalErrors = totalErrors;

    if (this.onTelemetry) {
      this.onTelemetry({
        running: this.running,
        allocating: this.allocating && reporting === 0,
        mode: this.config.mode,
        gbs,
        peakGBs: this.peakGBs,
        allocatedMB,
        targetMB: this.config.targetMB,
        activeWorkers: this.workers.length,
        verifiedGB: this.totalVerifiedGB,
        passes: Math.floor(this.totalPasses / Math.max(1, this.workers.length)),
        errors: totalErrors,
        oomHit: this.oomHit
      });
    }
  }

  getSummary() {
    const avgGBs =
      this.samplesGBs.length > 0
        ? this.samplesGBs.reduce((a, b) => a + b, 0) / this.samplesGBs.length
        : 0;
    return {
      mode: this.config.mode,
      avgGBs,
      peakGBs: this.peakGBs,
      verifiedGB: this.totalVerifiedGB,
      errors: this.totalErrors
    };
  }

  stop() {
    this.running = false;
    this.allocating = false;
    if (this.reportTimer) {
      clearInterval(this.reportTimer);
      this.reportTimer = null;
    }
    // Terminate workers immediately -> OS reclaims 100% of allocated physical RAM instantly
    for (const w of this.workers) {
      try {
        w.terminate();
      } catch (_) {}
    }
    this.workers = [];
    this.workerStats.clear();
    this.workerAllocMB.clear();
  }

  destroy() {
    this.stop();
    if (this.blobUrl) {
      URL.revokeObjectURL(this.blobUrl);
      this.blobUrl = null;
    }
  }
}

// High-Intensity Multi-Core CPU Stress Engine
// Features inline-compiled WebAssembly SIMD128 (v128) Power Virus + FP64 GEMM + Mixed + SHA-256

const WORKER_SCRIPT = `
'use strict';

// ==================== Inline WebAssembly SIMD128 Binary Generator ====================
function uleb(v) {
  const out = [];
  do {
    let b = v & 0x7f;
    v >>>= 7;
    if (v !== 0) b |= 0x80;
    out.push(b);
  } while (v !== 0);
  return out;
}

function sleb(v) {
  const out = [];
  let more = true;
  while (more) {
    let b = v & 0x7f;
    v >>= 7;
    if ((v === 0 && (b & 0x40) === 0) || (v === -1 && (b & 0x40) !== 0)) more = false;
    else b |= 0x80;
    out.push(b);
  }
  return out;
}

function wasmSection(id, body) {
  return [id, ...uleb(body.length), ...body];
}

function f32x4Bytes(a, b, c, d) {
  const buf = new ArrayBuffer(16);
  new Float32Array(buf).set([a, b, c, d]);
  return Array.from(new Uint8Array(buf));
}

function f64x2Bytes(a, b) {
  const buf = new ArrayBuffer(16);
  new Float64Array(buf).set([a, b]);
  return Array.from(new Uint8Array(buf));
}

function i32x4Bytes(a, b, c, d) {
  const buf = new ArrayBuffer(16);
  new Uint32Array(buf).set([a, b, c, d]);
  return Array.from(new Uint8Array(buf));
}

function simdOp(op) {
  return [0xfd, ...uleb(op)];
}

function buildSimdWasmModule() {
  const V128 = 0x7b, I32 = 0x7f, F32 = 0x7d;
  const LOCAL_GET = 0x20, LOCAL_SET = 0x21, LOCAL_TEE = 0x22;
  const I32_CONST = 0x41, I32_ADD = 0x6a, I32_SUB = 0x6b, I32_AND = 0x71, I32_GT_S = 0x4a;
  const LOOP = 0x03, BR_IF = 0x0d, END = 0x0b;

  // burn(rounds: i32, byteMask: i32, startPtr: i32) -> f32
  const typeSec = wasmSection(1, [1, 0x60, 3, I32, I32, I32, 1, F32]);
  const funcSec = wasmSection(3, [1, 0]);
  // 64 pages = 4 MB linear memory per worker (L2/L3 cache saturation)
  const memSec = wasmSection(5, [1, 0x00, ...uleb(64)]);
  const expSec = wasmSection(7, [
    2,
    3, 0x6d, 0x65, 0x6d, 0x02, 0, // 'mem'
    4, 0x62, 0x75, 0x72, 0x6e, 0x00, 0 // 'burn'
  ]);

  // Params: 0=rounds, 1=byteMask, 2=ptr (startPtr)
  // Locals: 3..25 (23 v128 registers: v0..v7, d0..d3, k0..k1, constants, m0)
  const body = [];
  body.push(1, 23, V128);

  const setV128Const = (idx, bytes) => { body.push(...simdOp(12), ...bytes, LOCAL_SET, idx); };
  setV128Const(17, f32x4Bytes(0.461, 0.457, 0.463, 0.453));
  setV128Const(18, f32x4Bytes(0.449, 0.459, 0.451, 0.467));
  setV128Const(19, f32x4Bytes(0.31, 0.37, 0.41, 0.43));
  setV128Const(20, f64x2Bytes(0.471, 0.467));
  setV128Const(21, f64x2Bytes(0.459, 0.463));
  setV128Const(22, f64x2Bytes(0.33, 0.39));
  setV128Const(23, i32x4Bytes(0x85ebca6b, 0xc2b2ae35, 0x27d4eb2d, 0x165667b1));
  setV128Const(24, i32x4Bytes(0x9e3779b9, 0x7f4a7c15, 0x517cc1b7, 0x243f6a88));

  for (let i = 0; i < 8; i++) {
    setV128Const(3 + i, f32x4Bytes(0.7 + i * 0.07, 0.8 + i * 0.05, 0.9 + i * 0.03, 0.6 + i * 0.09));
  }
  for (let i = 0; i < 4; i++) {
    setV128Const(11 + i, f64x2Bytes(0.75 + i * 0.11, 0.85 + i * 0.09));
  }
  setV128Const(15, i32x4Bytes(0x13579bdf, 0x2468ace1, 0xdeadbeef, 0x12345678));
  setV128Const(16, i32x4Bytes(0x87654321, 0xfedcba98, 0x99887766, 0x55443322));

  // Ensure startPtr aligned & masked
  body.push(LOCAL_GET, 2, LOCAL_GET, 1, I32_AND, LOCAL_SET, 2);

  // Main tight SIMD loop
  body.push(LOOP, 0x40);

  // 4x unrolled multi-port vector furnace (800 effective parallel scalar ops/iter)
  for (let u = 0; u < 4; u++) {
    // Load cache line slice into m0
    body.push(LOCAL_GET, 2, ...simdOp(0), 0x04, ...uleb(u * 16), LOCAL_SET, 25);

    // 8 independent f32x4 vector chains (32 FP32 lanes in flight)
    for (let i = 0; i < 8; i++) {
      const dst = 3 + i;
      const other = i < 4 ? 25 : (3 + ((i + 3) & 7));
      body.push(
        LOCAL_GET, dst, LOCAL_GET, 17, ...simdOp(230), // f32x4.mul
        LOCAL_GET, other, LOCAL_GET, 18, ...simdOp(230), // f32x4.mul
        ...simdOp(228), // f32x4.add
        LOCAL_GET, 19, ...simdOp(228), // + cAdd
        LOCAL_SET, dst
      );
    }

    // 2x f32x4.sqrt (vector sqrt unit stress)
    body.push(LOCAL_GET, 3, ...simdOp(227), LOCAL_GET, 19, ...simdOp(228), LOCAL_SET, 3);
    body.push(LOCAL_GET, 7, ...simdOp(227), LOCAL_GET, 19, ...simdOp(228), LOCAL_SET, 7);

    // 4 independent f64x2 vector chains (8 FP64 lanes in flight)
    for (let i = 0; i < 4; i++) {
      const dst = 11 + i;
      const other = 11 + ((i + 1) & 3);
      body.push(
        LOCAL_GET, dst, LOCAL_GET, 20, ...simdOp(242), // f64x2.mul
        LOCAL_GET, other, LOCAL_GET, 21, ...simdOp(242), // f64x2.mul
        ...simdOp(240), // f64x2.add
        LOCAL_GET, 22, ...simdOp(240), // + dAdd
        LOCAL_SET, dst
      );
    }

    // 2 independent i32x4 integer vector chains (8 Int32 SIMD lanes)
    body.push(
      LOCAL_GET, 15, LOCAL_GET, 23, ...simdOp(181),
      LOCAL_GET, 16, ...simdOp(174),
      LOCAL_GET, 24, ...simdOp(81),
      LOCAL_SET, 15
    );
    body.push(
      LOCAL_GET, 16, LOCAL_GET, 23, ...simdOp(181),
      LOCAL_GET, 15, ...simdOp(174),
      LOCAL_GET, 24, ...simdOp(81),
      LOCAL_SET, 16
    );
  }

  // Dirty writeback to L2/L3 cache lines
  body.push(LOCAL_GET, 2, LOCAL_GET, 3, ...simdOp(11), 0x04, 0x00);
  body.push(LOCAL_GET, 2, LOCAL_GET, 7, ...simdOp(11), 0x04, 0x10);

  // Advance pointer by 256 bytes (4 cache lines)
  body.push(LOCAL_GET, 2, I32_CONST, ...sleb(256), I32_ADD, LOCAL_GET, 1, I32_AND, LOCAL_SET, 2);

  // rounds--; br_if 0 (rounds > 0)
  body.push(LOCAL_GET, 0, I32_CONST, 1, I32_SUB, LOCAL_TEE, 0, I32_CONST, 0, I32_GT_S, BR_IF, 0);
  body.push(END);

  // Fold f32x4 + f64x2 + i32x4 into return f32 (prevents any dead-code elimination)
  body.push(
    LOCAL_GET, 3, LOCAL_GET, 7, ...simdOp(228),
    LOCAL_GET, 10, ...simdOp(228),
    ...simdOp(31), 0x00,
    LOCAL_GET, 11, ...simdOp(33), 0x00, 0xb6,
    0x92,
    LOCAL_GET, 15, ...simdOp(27), 0x00, I32_CONST, ...sleb(255), I32_AND, 0xb2,
    0x92,
    END
  );

  const codeSec = wasmSection(10, [1, ...uleb(body.length), ...body]);
  return new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, ...typeSec, ...funcSec, ...memSec, ...expSec, ...codeSec]);
}

let simdBurnFn = null;
let simdMemF32 = null;
let simdReady = false;
const SIMD_BYTE_MASK = (4 * 1024 * 1024) - 64; // 0x3FFFC0

function initSimd(seed) {
  if (simdReady) return true;
  try {
    if (typeof WebAssembly !== 'object') return false;
    const bytes = buildSimdWasmModule();
    if (!WebAssembly.validate(bytes)) return false;
    const mod = new WebAssembly.Module(bytes);
    const inst = new WebAssembly.Instance(mod);
    simdBurnFn = inst.exports.burn;
    simdMemF32 = new Float32Array(inst.exports.mem.buffer);
    const len = simdMemF32.length;
    for (let i = 0; i < len; i++) {
      simdMemF32[i] = 0.35 + ((i * 19 + seed * 7) % 100) * 0.0055;
    }
    simdReady = true;
    return true;
  } catch (_) {
    simdReady = false;
    return false;
  }
}

// ==================== JS Fallback & Specialized Kernels ====================
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

const F64_LEN = 131072; // 1 MB
const F64_MASK = F64_LEN - 1;
const f64Buf = new Float64Array(F64_LEN);

const U32_LEN = 16384; // 64 KB
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
let workload = 'simd';
let duty = 1.0;
let simdPtr = 0;
let extIdx = 0;
let fpPhase = 0.0;
let shaNonce = 0;
let sink = 0;

function initBuffers(seed) {
  initSimd(seed);
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
  simdPtr = (seed * 4096) & SIMD_BYTE_MASK;
}

function runSimd(rounds) {
  if (!simdReady || !simdBurnFn) {
    return runExtreme(Math.max(50000, Math.floor(rounds * 0.25)));
  }
  const out = simdBurnFn(rounds, SIMD_BYTE_MASK, simdPtr);
  simdPtr = (simdPtr + 16384) & SIMD_BYTE_MASK;
  sink = (sink + out) % 10000;
  // 800 effective parallel SIMD scalar ops per round
  return rounds * 800;
}

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
  const imul = Math.imul;
  const sqrt = Math.sqrt;
  const sin = Math.sin;
  const cos = Math.cos;

  for (let r = 0; r < rounds; r++) {
    const p0 = idx & F64_MASK;
    const p1 = (idx + 17) & F64_MASK;
    const p2 = (idx + 67) & F64_MASK;
    const p3 = (idx + 257) & F64_MASK;

    const m0 = fBuf[p0], m1 = fBuf[p1], m2 = fBuf[p2], m3 = fBuf[p3];

    a0 = a0 * 0.4999993 + m0 * 0.483121 + 0.17;
    a1 = a1 * 0.4999989 + m1 * 0.471983 + 0.23;
    a2 = a2 * 0.4999997 + m2 * 0.491237 + 0.19;
    a3 = a3 * 0.4999981 + m3 * 0.468719 + 0.21;

    a4 = a4 * 0.4123 + a0 * m0 * 0.3511 + 0.13;
    a5 = a5 * 0.4271 + a1 * m1 * 0.3419 + 0.17;
    a6 = a6 * 0.4319 + a2 * m2 * 0.3347 + 0.11;
    a7 = a7 * 0.4193 + a3 * m3 * 0.3483 + 0.19;

    a0 = sqrt(a0 * a4 + 1.0001);
    a1 = sin(a1) + cos(a5) + 2.15;
    a2 = sqrt(a2 * a6 + 1.0002);
    a3 = cos(a3) + sin(a7) + 2.15;

    const uIdx = idx & U32_MASK;
    const uv = uBuf[uIdx];
    i0 = (imul(i0 ^ uv, 0x85ebca6b) + ((i1 << 13) | (i1 >>> 19))) >>> 0;
    i1 = (imul(i1 ^ i0, 0xc2b2ae35) + ((i2 << 17) | (i2 >>> 15))) >>> 0;
    i2 = (imul(i2 ^ i1, 0x27d4eb2d) + ((i3 << 7) | (i3 >>> 25))) >>> 0;
    i3 = (imul(i3 ^ i2, 0x165667b1) + ((i0 << 11) | (i0 >>> 21))) >>> 0;

    fBuf[p0] = (a0 + a4) * 0.22;
    fBuf[p2] = (a2 + a6) * 0.22;
    uBuf[(uIdx + 31) & U32_MASK] = i0 ^ i3;

    idx = (idx + 73) & F64_MASK;
  }

  extIdx = idx;
  sink = (sink + a0 + a1 + a2 + a3 + (i0 & 0xff) * 1e-4) % 10000;
  return rounds * 48;
}

function runFp64(blocks) {
  const N = MAT_N;
  const A = matA, B = matB, C = matC;
  let totalOps = 0;

  for (let rep = 0; rep < blocks; rep++) {
    for (let i = 0; i < N; i += 4) {
      const i0 = i * N, i1 = (i + 1) * N, i2 = (i + 2) * N, i3 = (i + 3) * N;
      for (let j = 0; j < N; j += 4) {
        let c00 = 0, c01 = 0, c02 = 0, c03 = 0;
        let c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        let c20 = 0, c21 = 0, c22 = 0, c23 = 0;
        let c30 = 0, c31 = 0, c32 = 0, c33 = 0;
        for (let k = 0; k < N; k++) {
          const a0 = A[i0 + k], a1 = A[i1 + k], a2 = A[i2 + k], a3 = A[i3 + k];
          const kRow = k * N + j;
          const b0 = B[kRow], b1 = B[kRow + 1], b2 = B[kRow + 2], b3 = B[kRow + 3];
          c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
          c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
          c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
          c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        }
        const s = 0.015625;
        C[i0 + j] = c00 * s; C[i0 + j + 1] = c01 * s; C[i0 + j + 2] = c02 * s; C[i0 + j + 3] = c03 * s;
        C[i1 + j] = c10 * s; C[i1 + j + 1] = c11 * s; C[i1 + j + 2] = c12 * s; C[i1 + j + 3] = c13 * s;
        C[i2 + j] = c20 * s; C[i2 + j + 1] = c21 * s; C[i2 + j + 2] = c22 * s; C[i2 + j + 3] = c23 * s;
        C[i3 + j] = c30 * s; C[i3 + j + 1] = c31 * s; C[i3 + j + 2] = c32 * s; C[i3 + j + 3] = c33 * s;
      }
    }
    fpPhase = (fpPhase + 0.013) % 628.3185;
    const perturb = Math.sin(fpPhase) * 0.05 + 0.5;
    for (let i = 0; i < N; i++) {
      A[i * N + ((i + rep) & 63)] = Math.sqrt(Math.abs(C[i * N + i]) + 0.25) * perturb;
    }
    totalOps += 2 * N * N * N;
  }

  sink = (sink + C[0] + C[MAT_N * MAT_N - 1]) % 10000;
  return totalOps;
}

function runSha256(hashes) {
  const W = shaW, K = SHA256_K;
  let h0 = shaH[0] | 0, h1 = shaH[1] | 0, h2 = shaH[2] | 0, h3 = shaH[3] | 0;
  let h4 = shaH[4] | 0, h5 = shaH[5] | 0, h6 = shaH[6] | 0, h7 = shaH[7] | 0;
  let nonce = shaNonce | 0;

  for (let n = 0; n < hashes; n++) {
    nonce = (nonce + 1) | 0;
    W[0] = h0 ^ nonce; W[1] = h1; W[2] = h2; W[3] = h3;
    W[4] = h4; W[5] = h5; W[6] = h6; W[7] = h7;
    W[8] = 0x80000000 | 0; W[9] = nonce ^ 0x5a827999;
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

    let a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
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
  return hashes;
}

// Target ~135ms tight slices at 100% duty -> <0.008% yield overhead = true silicon thermal saturation
let chunkUnits = { simd: 2500000, extreme: 250000, fp64: 64, crypto: 40000 };
let reportOps = 0;
let reportWorkMs = 0;
let lastReportTs = 0;

const channel = new MessageChannel();
channel.port1.onmessage = step;

function scheduleNext(workMs) {
  if (!running) return;
  if (duty >= 0.99) {
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
  const units = chunkUnits[mode] || 50000;

  let ops = 0;
  if (mode === 'simd') {
    ops = runSimd(units);
  } else if (mode === 'extreme') {
    ops = runExtreme(units);
  } else if (mode === 'fp64') {
    ops = runFp64(units);
  } else {
    ops = runSha256(units);
  }

  const t1 = performance.now();
  const workMs = Math.max(0.1, t1 - t0);

  const targetMs = duty >= 0.99 ? 135 : 50;
  if (workMs < targetMs * 0.55) {
    const maxCap = mode === 'fp64' ? 8192 : mode === 'simd' ? 80000000 : 40000000;
    chunkUnits[mode] = Math.min(units * 2, maxCap);
  } else if (workMs > targetMs * 1.75 && units > 2) {
    chunkUnits[mode] = Math.max(mode === 'fp64' ? 1 : 1024, Math.floor(units * 0.65));
  }

  reportOps += ops;
  reportWorkMs += workMs;

  if (t1 - lastReportTs >= 420) {
    const wallMs = Math.max(1, t1 - lastReportTs);
    self.postMessage({
      type: 'stats',
      workerId,
      mode,
      simdActive: simdReady,
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
    workload = msg.workload || 'simd';
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
    this.simdSupported = this._detectSimdSupport();

    this.config = {
      threads: Math.max(1, navigator.hardwareConcurrency || 8),
      workload: 'simd',
      duty: 1.0
    };

    this.peakRate = 0;
    this.samples = [];
  }

  _detectSimdSupport() {
    try {
      // Minimal valid 1-instruction WebAssembly SIMD128 feature test module
      const probe = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
        0x03, 0x02, 0x01, 0x00,
        0x0a, 0x16, 0x01, 0x14, 0x00, 0xfd, 0x0c,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0b
      ]);
      return typeof WebAssembly === 'object' && WebAssembly.validate(probe);
    } catch (_) {
      return false;
    }
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

    while (this.workers.length > clamped) {
      const w = this.workers.pop();
      this.workerStats.delete(w._kiiId);
      try {
        w.terminate();
      } catch (_) {}
    }

    while (this.workers.length < clamped) {
      const id = this.workers.length;
      const worker = new Worker(url);
      worker._kiiId = id;
      worker.onmessage = (e) => {
        const data = e.data;
        if (!data || data.type !== 'stats') return;
        const ratePerSec = data.ops / (data.wallMs / 1000);
        const saturation = Math.min(100, (data.workMs / data.wallMs) * 100);
        if (data.simdActive) this.simdSupported = true;
        this.workerStats.set(data.workerId, {
          ratePerSec,
          saturation,
          mode: data.mode,
          simdActive: data.simdActive,
          ts: performance.now()
        });
      };
      worker.onerror = () => {};
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
    let simdActive = this.simdSupported;

    for (const [, stat] of this.workerStats) {
      if (now - stat.ts < 2500 && stat.mode === this.config.workload) {
        totalRate += stat.ratePerSec;
        totalSat += stat.saturation;
        reportingWorkers++;
        if (stat.simdActive !== undefined) simdActive = stat.simdActive;
      }
    }

    // Return ops/sec in raw scale AND scaled metric:
    // simd/fp64 -> GFLOP/s (ops / 1e9) when large, MOPS (ops / 1e6), crypto -> MH/s
    const rateOps = totalRate;
    if (rateOps > this.peakRate) {
      this.peakRate = rateOps;
    }
    if (reportingWorkers > 0) {
      this.samples.push(rateOps);
      if (this.samples.length > 7200) this.samples.shift();
    }

    const avgSat = reportingWorkers > 0 ? totalSat / reportingWorkers : 0;

    if (this.onTelemetry) {
      this.onTelemetry({
        running: this.running,
        workload: this.config.workload,
        simdActive,
        rateOps,
        peakOps: this.peakRate,
        activeWorkers: this.workers.length,
        reportingWorkers,
        saturationPct: Math.min(100, avgSat)
      });
    }
  }

  getSummary() {
    const avgOps =
      this.samples.length > 0
        ? this.samples.reduce((a, b) => a + b, 0) / this.samples.length
        : 0;
    return {
      workload: this.config.workload,
      avgOps,
      peakOps: this.peakRate
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

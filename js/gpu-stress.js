// Nuclear Dual-Backend GPU Stress Engine v2
// WebGPU: 8x vec4 Dual-Issue Compute + Strided VRAM Bandwidth + Concurrent Offscreen Raymarch Raster Pipeline
// WebGL2: 4-MRT Offscreen FBO (up to 4096^2 x 4 targets) + TMU Texture-Fetch + 8x vec4 ALU Furnace + Deep FenceSync Queue

const WGSL_COMPUTE = /* wgsl */ `
struct Params {
  count: u32,
  iters: u32,
  mask: u32,
  seed: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bufIn: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bufOut: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) {
    return;
  }

  let mask = params.mask;
  let base = bufIn[idx & mask];
  let base2 = bufIn[(idx * 7u + 131u) & mask];
  let ph = params.seed + f32(idx & 1023u) * 0.001953125;

  // 8 independent vec4<f32> accumulators = 32 parallel FP32 lanes per thread (dual-issue saturation)
  var v0 = base + vec4<f32>(0.37, 0.71, 0.19, 0.53);
  var v1 = base.yzwx * 0.73 + vec4<f32>(ph, 0.41, 0.89, 0.29);
  var v2 = base.zwxy * 0.61 + vec4<f32>(0.67, ph * 0.5, 0.31, 0.83);
  var v3 = base.wxyz * 0.89 + vec4<f32>(0.23, 0.59, ph * 0.25, 0.47);
  var v4 = base2 + vec4<f32>(0.51, 0.29, 0.77, 0.43);
  var v5 = base2.yzwx * 0.67 + vec4<f32>(0.19, ph * 0.33, 0.61, 0.87);
  var v6 = base2.zwxy * 0.83 + vec4<f32>(0.73, 0.37, 0.53, ph * 0.41);
  var v7 = base2.wxyz * 0.71 + vec4<f32>(0.47, 0.83, 0.23, 0.69);

  let c0 = vec4<f32>(0.4873, 0.4817, 0.4841, 0.4863);
  let c1 = vec4<f32>(0.4631, 0.4779, 0.4613, 0.4787);
  let c2 = vec4<f32>(0.4723, 0.4677, 0.4811, 0.4689);
  let c3 = vec4<f32>(0.4591, 0.4613, 0.4587, 0.4709);

  let n = params.iters;
  var memTap = base;
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    // Periodic strided VRAM / L2 cache fetch to hammer memory bus alongside shader ALUs
    if ((i & 7u) == 0u) {
      memTap = bufIn[(idx + i * 257u) & mask];
      v0 = v0 + memTap * 0.25;
      v4 = v4 + memTap.wzyx * 0.25;
    }

    // 16 x vec4 FMA + 4 x vec4 SFU transcendental ops (~192 FP32 FLOPs/thread/iter)
    v0 = fma(v0, c0, fma(v1, c1, vec4<f32>(0.21)));
    v1 = fma(v1, c2, fma(v2, c3, vec4<f32>(0.23)));
    v2 = fma(v2, c1, fma(v3, c0, vec4<f32>(0.27)));
    v3 = fma(v3, c3, fma(v4, c2, vec4<f32>(0.29)));
    v4 = fma(v4, c0, fma(v5, c2, vec4<f32>(0.31)));
    v5 = fma(v5, c1, fma(v6, c3, vec4<f32>(0.33)));
    v6 = fma(v6, c2, fma(v7, c0, vec4<f32>(0.37)));
    v7 = fma(v7, c3, fma(v0, c1, vec4<f32>(0.39)));

    // SFU transcendental pipe saturation
    v0 = sin(v0) + vec4<f32>(1.15);
    v2 = cos(v2) + vec4<f32>(1.15);
    v4 = sqrt(abs(v4) + vec4<f32>(0.25));
    v6 = inverseSqrt(abs(v6) + vec4<f32>(0.5)) * 1.45;
  }

  let outA = fract(abs((v0 + v1 + v2 + v3) * 0.25)) + vec4<f32>(0.1);
  let outB = fract(abs((v4 + v5 + v6 + v7) * 0.25)) + vec4<f32>(0.1);
  bufOut[idx & mask] = outA;
  bufOut[(idx + 1024u) & mask] = outB;
}
`;

const WGSL_RASTER = /* wgsl */ `
struct PreviewUniforms {
  time: f32,
  aspect: f32,
  extraIters: f32,
  passOffset: f32,
};

@group(0) @binding(0) var<uniform> u: PreviewUniforms;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  let p = positions[vid];
  var out: VSOut;
  out.pos = vec4<f32>(p, 0.0, 1.0);
  out.uv = p * 0.5 + 0.5;
  return out;
}

fn mandelbulbDE(pos: vec3<f32>) -> vec2<f32> {
  var z = pos;
  var dr: f32 = 1.0;
  var r: f32 = 0.0;
  var trap: f32 = 1e6;
  for (var i: i32 = 0; i < 8; i = i + 1) {
    r = length(z);
    if (r > 2.2) { break; }
    let rs = max(r, 1e-5);
    trap = min(trap, rs);
    let theta = acos(clamp(z.z / rs, -1.0, 1.0));
    let phi = atan2(z.y, z.x);
    dr = pow(rs, 7.0) * 8.0 * dr + 1.0;
    let zr = pow(rs, 8.0);
    let nt = theta * 8.0;
    let np = phi * 8.0;
    z = zr * vec3<f32>(sin(nt) * cos(np), sin(np) * sin(nt), cos(nt)) + pos;
  }
  let rs = max(r, 1e-5);
  return vec2<f32>(0.5 * log(rs) * rs / max(dr, 1e-5), trap);
}

fn calcNormal(p: vec3<f32>) -> vec3<f32> {
  let e = vec2<f32>(0.0022, -0.0022);
  return normalize(
    e.xyy * mandelbulbDE(p + e.xyy).x +
    e.yyx * mandelbulbDE(p + e.yyx).x +
    e.yxy * mandelbulbDE(p + e.yxy).x +
    e.xxx * mandelbulbDE(p + e.xxx).x
  );
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let p = (in.uv * 2.0 - 1.0) * vec2<f32>(u.aspect, 1.0);
  let tPhase = u.time + u.passOffset * 0.019;
  let ang = tPhase * 0.35;
  let ro = vec3<f32>(cos(ang) * 2.45, sin(tPhase * 0.22) * 0.55, sin(ang) * 2.45);
  let ww = normalize(-ro);
  let uu = normalize(cross(ww, vec3<f32>(0.0, 1.0, 0.0)));
  let vv = cross(uu, ww);
  let rd = normalize(p.x * uu + p.y * vv + 1.85 * ww);

  var bg = mix(vec3<f32>(0.895, 0.922, 0.958), vec3<f32>(0.952, 0.966, 0.986), in.uv.y);

  var t: f32 = 0.2;
  var hit: bool = false;
  var trap: f32 = 1.0;
  var steps: f32 = 0.0;

  for (var i: i32 = 0; i < 46; i = i + 1) {
    let pos = ro + rd * t;
    let res = mandelbulbDE(pos);
    trap = res.y;
    if (res.x < 0.002) {
      hit = true;
      break;
    }
    t = t + res.x * 0.85;
    steps = steps + 1.0;
    if (t > 5.0) { break; }
  }

  var col = bg;
  if (hit) {
    let pos = ro + rd * t;
    let nor = calcNormal(pos);
    let lig1 = normalize(vec3<f32>(0.7, 0.85, 0.55));
    let lig2 = normalize(vec3<f32>(-0.6, -0.3, -0.7));
    let dif1 = max(dot(nor, lig1), 0.0);
    let dif2 = max(dot(nor, lig2), 0.0);
    let fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 3.0);
    let ao = clamp(1.0 - steps / 48.0, 0.22, 1.0);

    let cobalt = vec3<f32>(0.184, 0.435, 0.929);
    let teal = vec3<f32>(0.055, 0.647, 0.549);
    let ceramic = vec3<f32>(0.93, 0.95, 0.98);
    let tint = mix(cobalt, teal, clamp(trap * 1.15, 0.0, 1.0));
    let baseCol = mix(tint, ceramic, 0.35 + 0.35 * dif1);
    let h = normalize(lig1 - rd);
    let spec = pow(max(dot(nor, h), 0.0), 32.0) * 0.65;
    col = baseCol * (0.48 + 0.58 * dif1 + 0.22 * dif2) * ao + vec3<f32>(spec) + mix(cobalt, teal, 0.5) * fre * 0.35;
  } else {
    let glow = exp(-steps * 0.088);
    col = mix(vec3<f32>(0.84, 0.89, 0.96), bg, glow);
  }

  // Optional offscreen raster ALU boost when dual-pipeline furnace is active
  let extra = u32(max(u.extraIters, 0.0));
  if (extra > 0u) {
    var a0 = vec4<f32>(col, 0.618) + vec4<f32>(p, u.passOffset * 0.13, t * 0.2);
    var a1 = a0.yzwx * 0.77 + vec4<f32>(0.31, 0.57, 0.83, 0.19);
    let k0 = vec4<f32>(0.487, 0.481, 0.484, 0.486);
    let k1 = vec4<f32>(0.463, 0.478, 0.461, 0.479);
    for (var j: u32 = 0u; j < extra; j = j + 1u) {
      a0 = fma(a0, k0, fma(a1, k1, vec4<f32>(0.23)));
      a1 = sin(a0) + cos(a1) + vec4<f32>(1.2);
    }
    let micro = fract(abs((a0.rgb + a1.gbr) * 0.5)) * 0.01 - 0.005;
    col = col + micro;
  }

  return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
`;

// ==================== WebGL2 GLSL ES 3.00 4-MRT + TMU Furnace ====================
const GLSL_VERT = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 pos = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
  vUv = pos * 0.5 + 0.5;
  gl_Position = vec4(pos, 0.0, 1.0);
}
`;

const GLSL_STRESS_FRAG_MRT = `#version 300 es
precision highp float;

in vec2 vUv;
layout(location = 0) out vec4 fragOut0;
layout(location = 1) out vec4 fragOut1;
layout(location = 2) out vec4 fragOut2;
layout(location = 3) out vec4 fragOut3;

uniform float uTime;
uniform float uAspect;
uniform float uPassOffset;
uniform int uExtraIters;
uniform int uRaySteps;
uniform sampler2D uNoiseTex;

vec2 mandelbulbDE(vec3 pos) {
  vec3 z = pos;
  float dr = 1.0;
  float r = 0.0;
  float trap = 1e5;
  for (int i = 0; i < 8; i++) {
    r = length(z);
    if (r > 2.2) break;
    float rs = max(r, 1e-5);
    trap = min(trap, rs);
    float theta = acos(clamp(z.z / rs, -1.0, 1.0));
    float phi = atan(z.y, z.x);
    dr = pow(rs, 7.0) * 8.0 * dr + 1.0;
    float zr = pow(rs, 8.0);
    theta *= 8.0;
    phi *= 8.0;
    z = zr * vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta)) + pos;
  }
  float rs = max(r, 1e-5);
  return vec2(0.5 * log(rs) * rs / max(dr, 1e-5), trap);
}

vec3 calcNormal(vec3 p) {
  vec2 e = vec2(0.0024, -0.0024);
  return normalize(
    e.xyy * mandelbulbDE(p + e.xyy).x +
    e.yyx * mandelbulbDE(p + e.yyx).x +
    e.yxy * mandelbulbDE(p + e.yxy).x +
    e.xxx * mandelbulbDE(p + e.xxx).x
  );
}

void main() {
  vec2 uv = vUv;
  vec2 p = (uv * 2.0 - 1.0) * vec2(uAspect, 1.0);

  float tPhase = uTime + uPassOffset * 0.017;
  float ang = tPhase * 0.36;
  vec3 ro = vec3(cos(ang) * 2.45, sin(tPhase * 0.24) * 0.55, sin(ang) * 2.45);
  vec3 ww = normalize(-ro);
  vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
  vec3 vv = cross(uu, ww);
  vec3 rd = normalize(p.x * uu + p.y * vv + 1.85 * ww);

  vec3 bg = mix(vec3(0.895, 0.922, 0.958), vec3(0.952, 0.966, 0.986), uv.y);

  float t = 0.2;
  bool hit = false;
  float trap = 1.0;
  float steps = 0.0;

  for (int i = 0; i < 64; i++) {
    if (i >= uRaySteps) break;
    vec3 pos = ro + rd * t;
    vec2 res = mandelbulbDE(pos);
    trap = res.y;
    if (res.x < 0.0022) {
      hit = true;
      break;
    }
    t += res.x * 0.85;
    steps += 1.0;
    if (t > 5.0) break;
  }

  vec3 col = bg;
  if (hit) {
    vec3 pos = ro + rd * t;
    vec3 nor = calcNormal(pos);
    vec3 lig1 = normalize(vec3(0.7, 0.85, 0.55));
    vec3 lig2 = normalize(vec3(-0.6, -0.3, -0.7));
    float dif1 = max(dot(nor, lig1), 0.0);
    float dif2 = max(dot(nor, lig2), 0.0);
    float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 3.0);
    float ao = clamp(1.0 - steps / float(max(uRaySteps, 1)), 0.24, 1.0);

    vec3 cobalt = vec3(0.184, 0.435, 0.929);
    vec3 teal = vec3(0.055, 0.647, 0.549);
    vec3 ceramic = vec3(0.935, 0.952, 0.982);
    vec3 tint = mix(cobalt, teal, clamp(trap * 1.15, 0.0, 1.0));
    vec3 baseCol = mix(tint, ceramic, 0.34 + 0.36 * dif1);

    vec3 h = normalize(lig1 - rd);
    float spec = pow(max(dot(nor, h), 0.0), 32.0) * 0.65;
    col = baseCol * (0.48 + 0.58 * dif1 + 0.22 * dif2) * ao + vec3(spec) + mix(cobalt, teal, 0.5) * fre * 0.34;
  } else {
    float glow = exp(-steps * 0.085);
    col = mix(vec3(0.835, 0.885, 0.958), bg, glow);
  }

  // 8 x vec4 Dual-Issue ALU + SFU + TMU Texture Fetch Furnace (up to 1024 iterations!)
  vec4 tex0 = texture(uNoiseTex, uv + vec2(uPassOffset * 0.071, uTime * 0.037));
  vec4 v0 = vec4(col, 0.618) + tex0 * 0.5;
  vec4 v1 = v0.yzwx * 0.77 + vec4(0.31, 0.57, 0.83, 0.19);
  vec4 v2 = v0.zwxy * 0.63 + vec4(0.41, 0.29, 0.73, 0.53);
  vec4 v3 = v0.wxyz * 0.89 + vec4(0.17, 0.67, 0.37, 0.71);
  vec4 v4 = tex0.wzyx + vec4(p, uPassOffset * 0.13, t * 0.2);
  vec4 v5 = v4.yzwx * 0.71 + vec4(0.53, 0.23, 0.79, 0.37);
  vec4 v6 = v4.zwxy * 0.67 + vec4(0.29, 0.73, 0.43, 0.61);
  vec4 v7 = v4.wxyz * 0.83 + vec4(0.61, 0.41, 0.89, 0.17);

  const vec4 c0 = vec4(0.487, 0.481, 0.484, 0.486);
  const vec4 c1 = vec4(0.463, 0.478, 0.461, 0.479);
  const vec4 c2 = vec4(0.472, 0.468, 0.481, 0.469);
  const vec4 c3 = vec4(0.459, 0.461, 0.458, 0.471);

  for (int i = 0; i < 1024; i++) {
    if (i >= uExtraIters) break;
    // Periodic chaotic TMU texture fetch hammers L1/L2 texture caches & TMU pipelines
    if ((i & 7) == 0) {
      vec4 tap = texture(uNoiseTex, fract(v0.xy + v4.zw * 0.37 + float(i) * 0.013));
      v0 += tap * 0.25;
      v4 += tap.wzyx * 0.25;
    }

    v0 = v0 * c0 + v1 * c1 + 0.21;
    v1 = v1 * c2 + v2 * c3 + 0.23;
    v2 = v2 * c1 + v3 * c0 + 0.27;
    v3 = v3 * c3 + v4 * c2 + 0.29;
    v4 = v4 * c0 + v5 * c2 + 0.31;
    v5 = v5 * c1 + v6 * c3 + 0.33;
    v6 = v6 * c2 + v7 * c0 + 0.37;
    v7 = v7 * c3 + v0 * c1 + 0.39;

    v0 = sin(v0) + 1.15;
    v2 = cos(v2) + 1.15;
    v4 = sqrt(abs(v4) + 0.25);
    v6 = inversesqrt(abs(v6) + 0.5) * 1.45;
  }

  vec3 micro = fract(abs((v0.rgb + v2.gbr + v4.brg + v6.rgb) * 0.25)) * 0.012 - 0.006;
  fragOut0 = vec4(clamp(col + micro, 0.0, 1.0), 1.0);
  // Write all furnace lanes into MRT attachments 1..3 -> 4x ROP & VRAM write bandwidth!
  fragOut1 = fract(abs((v0 + v1) * 0.5));
  fragOut2 = fract(abs((v2 + v3 + v4) * 0.333));
  fragOut3 = fract(abs((v5 + v6 + v7) * 0.333));
}
`;

const GLSL_BLIT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
  fragColor = texture(uTex, vUv);
}
`;

function largestPowerOfTwo(n) {
  let p = 1;
  while ((p << 1) <= n) p <<= 1;
  return p;
}

export class GpuStressEngine {
  constructor(webgpuCanvas, webglCanvas, onTelemetry, onError) {
    this.webgpuCanvas = webgpuCanvas;
    this.webglCanvas = webglCanvas;
    this.onTelemetry = onTelemetry;
    this.onError = onError;

    this.running = false;
    this.capabilities = {
      webgpu: false,
      webgl2: false,
      rendererName: '',
      preferredEngine: 'webgl2',
      mrtCount: 1
    };

    this.config = {
      engine: 'webgl2',
      preset: 'auto',      // 'auto' | 'high' | 'extreme' | 'overdrive'
      scale: 8,            // 1..32
      uncapVsync: true,
      dualPipeline: true,  // WebGPU concurrent Compute + Offscreen Raster
      showPreview: true
    };

    // WebGPU resources
    this.gpuAdapter = null;
    this.gpuDevice = null;
    this.gpuContext = null;
    this.gpuFormat = null;
    this.wgComputePipeline = null;
    this.wgPreviewPipeline = null;
    this.wgOffscreenRasterPipeline = null;
    this.wgUniformBuf = null;
    this.wgPreviewUniformBuf = null;
    this.wgOffscreenUniformBuf = null;
    this.wgOffscreenTex = null;
    this.wgOffscreenView = null;
    this.wgOffscreenSize = 0;
    this.wgBufA = null;
    this.wgBufB = null;
    this.wgBindAB = null;
    this.wgBindBA = null;
    this.wgPreviewBind = null;
    this.wgOffscreenBind = null;
    this.wgMaxWorkgroups = 4096;
    this.wgElementCount = 0;

    // WebGL2 resources
    this.gl = null;
    this.glStressProg = null;
    this.glBlitProg = null;
    this.glUniforms = {};
    this.glFbo = null;
    this.glFboTextures = [];
    this.glNoiseTex = null;
    this.glFboSize = 0;
    this.glMaxTexSize = 2048;
    this.glMrtCount = 1;

    // Runtime loop / telemetry state
    this._runToken = 0;
    this._rafId = null;
    this._pollTimer = null;
    this._telemetryTimer = null;
    this._startTime = 0;
    this._glPending = [];

    this._dyn = {
      // WebGPU
      wgWorkgroups: 1024,
      wgIters: 24,
      wgDispatches: 1,
      wgRasterPasses: 1,
      wgRasterExtraIters: 16,
      // WebGL2
      glSize: 512,
      glExtraIters: 8,
      glRaySteps: 36,
      glPasses: 1,
      // Shared stats
      emaBatchMs: 18,
      inFlight: 0,
      windowBatches: 0,
      windowFlops: 0,
      windowPixels: 0,
      windowStart: 0,
      lastBatchMs: 0,
      samplesGflops: []
    };

    this._onContextLost = (e) => {
      e.preventDefault();
      if (this.running) {
        this.stop();
        if (this.onError) this.onError('context_lost');
      }
    };
    if (this.webglCanvas) {
      this.webglCanvas.addEventListener('webglcontextlost', this._onContextLost, false);
    }
  }

  async init() {
    this._probeWebGL2();
    await this._probeWebGPU();

    if (this.capabilities.webgpu) {
      this.capabilities.preferredEngine = 'webgpu';
      this.config.engine = 'webgpu';
    } else if (this.capabilities.webgl2) {
      this.capabilities.preferredEngine = 'webgl2';
      this.config.engine = 'webgl2';
    } else {
      this.capabilities.preferredEngine = 'none';
      this.config.engine = 'none';
    }

    return this.capabilities;
  }

  _probeWebGL2() {
    if (!this.webglCanvas) return;
    try {
      const gl = this.webglCanvas.getContext('webgl2', {
        powerPreference: 'high-performance',
        antialias: false,
        depth: false,
        stencil: false,
        alpha: false,
        preserveDrawingBuffer: false
      });
      if (!gl) return;
      this.gl = gl;
      this.capabilities.webgl2 = true;
      const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 2048;
      const maxRb = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) || 2048;
      this.glMaxTexSize = Math.min(4096, maxTex, maxRb);

      const maxAttach = gl.getParameter(gl.MAX_COLOR_ATTACHMENTS) || 1;
      const maxDrawBuf = gl.getParameter(gl.MAX_DRAW_BUFFERS) || 1;
      this.glMrtCount = Math.min(4, maxAttach, maxDrawBuf);
      this.capabilities.mrtCount = this.glMrtCount;

      const dbg = gl.getExtension('WEBGL_debug_renderer_info');
      if (dbg) {
        const renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
        if (renderer && typeof renderer === 'string') {
          this.capabilities.rendererName = this._cleanGpuName(renderer);
        }
      }
      this._initWebGL2Pipelines();
    } catch (_) {
      this.capabilities.webgl2 = false;
    }
  }

  async _probeWebGPU() {
    if (!navigator.gpu || !this.webgpuCanvas) return;
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });
      if (!adapter) return;

      let adapterName = '';
      if (adapter.info) {
        adapterName = adapter.info.description || adapter.info.device || adapter.info.architecture || '';
      } else if (typeof adapter.requestAdapterInfo === 'function') {
        try {
          const info = await adapter.requestAdapterInfo();
          adapterName = info.description || info.device || info.architecture || '';
        } catch (_) {}
      }
      if (adapterName && !this.capabilities.rendererName) {
        this.capabilities.rendererName = this._cleanGpuName(adapterName);
      }

      // Request default device cleanly (spec minimum is already >=128MB storage buffer)
      const device = await adapter.requestDevice();
      const maxStorageBytes = Math.min(
        128 * 1024 * 1024,
        device.limits.maxStorageBufferBindingSize || 64 * 1024 * 1024,
        device.limits.maxBufferSize || 64 * 1024 * 1024
      );

      this.gpuAdapter = adapter;
      this.gpuDevice = device;

      device.lost.then((info) => {
        if (info && info.reason !== 'destroyed' && this.running && this.config.engine === 'webgpu') {
          this.stop();
          if (this.onError) this.onError('context_lost');
        }
      });

      const ctx = this.webgpuCanvas.getContext('webgpu');
      if (!ctx) return;
      const format = navigator.gpu.getPreferredCanvasFormat();
      ctx.configure({
        device,
        format,
        alphaMode: 'opaque'
      });
      this.gpuContext = ctx;
      this.gpuFormat = format;

      this._initWebGPUPipelines(maxStorageBytes);
      this.capabilities.webgpu = true;
    } catch (_) {
      this.capabilities.webgpu = false;
    }
  }

  _cleanGpuName(raw) {
    let name = raw.trim();
    const angleMatch = name.match(/ANGLE\s*\(([^,]+),\s*([^,()]+)/i);
    if (angleMatch) {
      name = angleMatch[2].trim();
    }
    name = name.replace(/\s*\(0x[0-9a-f]+\)/gi, '').replace(/Direct3D11.*$/i, '').trim();
    return name.slice(0, 44);
  }

  _initWebGPUPipelines(maxStorageBytes) {
    const device = this.gpuDevice;
    // Target up to 64 MB per ping-pong storage buffer (128 MB total active VRAM working set!)
    // Use exact power-of-two workgroups so bitmask wrapping is 100% bounds-safe
    const targetBytes = Math.min(64 * 1024 * 1024, maxStorageBytes);
    const rawMaxWg = Math.max(256, Math.floor(targetBytes / (16 * 256)));
    const limitWg = device.limits.maxComputeWorkgroupsPerDimension || 65535;
    this.wgMaxWorkgroups = largestPowerOfTwo(Math.min(16384, rawMaxWg, limitWg));
    this.wgElementCount = this.wgMaxWorkgroups * 256;
    const byteLength = this.wgElementCount * 16;

    this.wgBufA = device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.wgBufB = device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    const seedArr = new Float32Array(Math.min(this.wgElementCount * 4, 131072));
    for (let i = 0; i < seedArr.length; i++) {
      seedArr[i] = 0.2 + ((i * 37) % 100) * 0.007;
    }
    device.queue.writeBuffer(this.wgBufA, 0, seedArr);

    this.wgUniformBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const computeMod = device.createShaderModule({ code: WGSL_COMPUTE });
    this.wgComputePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: computeMod, entryPoint: 'main' }
    });

    const bindLayout = this.wgComputePipeline.getBindGroupLayout(0);
    this.wgBindAB = device.createBindGroup({
      layout: bindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.wgUniformBuf } },
        { binding: 1, resource: { buffer: this.wgBufA } },
        { binding: 2, resource: { buffer: this.wgBufB } }
      ]
    });
    this.wgBindBA = device.createBindGroup({
      layout: bindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.wgUniformBuf } },
        { binding: 1, resource: { buffer: this.wgBufB } },
        { binding: 2, resource: { buffer: this.wgBufA } }
      ]
    });

    // Raster module shared by Preview & Concurrent Offscreen Dual-Pipeline Furnace
    const rasterMod = device.createShaderModule({ code: WGSL_RASTER });

    this.wgPreviewUniformBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.wgPreviewPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: rasterMod, entryPoint: 'vs_main' },
      fragment: {
        module: rasterMod,
        entryPoint: 'fs_main',
        targets: [{ format: this.gpuFormat }]
      },
      primitive: { topology: 'triangle-list' }
    });
    this.wgPreviewBind = device.createBindGroup({
      layout: this.wgPreviewPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.wgPreviewUniformBuf } }]
    });

    // Offscreen dual-pipeline raster target (fires Raster + Fragment + ROP concurrently with Compute!)
    this.wgOffscreenUniformBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.wgOffscreenRasterPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: rasterMod, entryPoint: 'vs_main' },
      fragment: {
        module: rasterMod,
        entryPoint: 'fs_main',
        targets: [{ format: 'rgba8unorm' }]
      },
      primitive: { topology: 'triangle-list' }
    });
    this.wgOffscreenBind = device.createBindGroup({
      layout: this.wgOffscreenRasterPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.wgOffscreenUniformBuf } }]
    });

    this._ensureWgOffscreenTex(1024);
  }

  _ensureWgOffscreenTex(size) {
    if (!this.gpuDevice) return;
    const max2D = Math.min(2048, this.gpuDevice.limits.maxTextureDimension2D || 2048);
    const clamped = Math.max(512, Math.min(max2D, size));
    if (this.wgOffscreenTex && this.wgOffscreenSize === clamped) return;

    if (this.wgOffscreenTex) {
      try {
        this.wgOffscreenTex.destroy();
      } catch (_) {}
    }
    this.wgOffscreenTex = this.gpuDevice.createTexture({
      size: [clamped, clamped, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    this.wgOffscreenView = this.wgOffscreenTex.createView();
    this.wgOffscreenSize = clamped;
  }

  _compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error('Shader compile failed: ' + log);
    }
    return shader;
  }

  _linkProgram(gl, vsSrc, fsSrc) {
    const vs = this._compileShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = this._compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error('Program link failed: ' + log);
    }
    return prog;
  }

  _initWebGL2Pipelines() {
    const gl = this.gl;
    if (!gl) return;
    this.glStressProg = this._linkProgram(gl, GLSL_VERT, GLSL_STRESS_FRAG_MRT);
    this.glBlitProg = this._linkProgram(gl, GLSL_VERT, GLSL_BLIT_FRAG);
    this.glUniforms = {
      uTime: gl.getUniformLocation(this.glStressProg, 'uTime'),
      uAspect: gl.getUniformLocation(this.glStressProg, 'uAspect'),
      uPassOffset: gl.getUniformLocation(this.glStressProg, 'uPassOffset'),
      uExtraIters: gl.getUniformLocation(this.glStressProg, 'uExtraIters'),
      uRaySteps: gl.getUniformLocation(this.glStressProg, 'uRaySteps'),
      uNoiseTex: gl.getUniformLocation(this.glStressProg, 'uNoiseTex'),
      uTex: gl.getUniformLocation(this.glBlitProg, 'uTex')
    };

    // Create 256x256 procedural RGBA noise texture for TMU & L1/L2 texture cache thrashing
    const noiseSize = 256;
    const noiseData = new Uint8Array(noiseSize * noiseSize * 4);
    let s = 0x12345678;
    for (let i = 0; i < noiseData.length; i++) {
      s = (Math.imul(s ^ (s >>> 15), 0x85ebca6b) + 0x9e3779b9) >>> 0;
      noiseData[i] = (s >>> 24) & 0xff;
    }
    const nTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, nTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, noiseSize, noiseSize, 0, gl.RGBA, gl.UNSIGNED_BYTE, noiseData);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    this.glNoiseTex = nTex;

    this._ensureGlFbo(512);
  }

  _ensureGlFbo(size) {
    const gl = this.gl;
    if (!gl) return;
    const clamped = Math.max(256, Math.min(this.glMaxTexSize, size));
    if (this.glFbo && this.glFboSize === clamped) return;

    for (const tex of this.glFboTextures) {
      gl.deleteTexture(tex);
    }
    this.glFboTextures = [];
    if (this.glFbo) gl.deleteFramebuffer(this.glFbo);

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);

    const count = Math.max(1, this.glMrtCount);
    const drawBufs = [];
    for (let i = 0; i < count; i++) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, clamped, clamped, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, tex, 0);
      this.glFboTextures.push(tex);
      drawBufs.push(gl.COLOR_ATTACHMENT0 + i);
    }
    gl.drawBuffers(drawBufs);

    // Fallback safely to single attachment if MRT framebuffer is ever rejected
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE && count > 1) {
      for (let i = 1; i < this.glFboTextures.length; i++) {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, null, 0);
        gl.deleteTexture(this.glFboTextures[i]);
      }
      this.glFboTextures.length = 1;
      this.glMrtCount = 1;
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.glFbo = fbo;
    this.glFboSize = clamped;
  }

  _syncPreviewCanvasSize(canvas) {
    if (!canvas) return { w: 400, h: 180 };
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(160, Math.round((rect.width || 400) * dpr));
    const h = Math.max(96, Math.round((rect.height || 180) * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    return { w, h };
  }

  updateConfig(partial = {}) {
    const prevEngine = this.config.engine;
    const prevPreset = this.config.preset;
    const prevScale = this.config.scale;
    this.config = { ...this.config, ...partial };

    if (this.config.engine === 'webgpu' && !this.capabilities.webgpu) {
      this.config.engine = 'webgl2';
    }

    if (!this.running) return;

    if (this.config.engine !== prevEngine) {
      this.stop();
      this.start(this.config);
      return;
    }

    if (this.config.preset !== prevPreset || (this.config.preset !== 'auto' && this.config.scale !== prevScale)) {
      this._applyPresetInitialLoad();
    }
  }

  _applyPresetInitialLoad() {
    const p = this.config.preset;
    const s = Math.max(1, Math.min(32, this.config.scale));
    if (p === 'auto') {
      this._dyn.wgWorkgroups = Math.min(this.wgMaxWorkgroups, 1024);
      this._dyn.wgIters = 24;
      this._dyn.wgDispatches = 1;
      this._dyn.wgRasterPasses = 1;
      this._dyn.wgRasterExtraIters = 12;
      this._dyn.glSize = 512;
      this._dyn.glExtraIters = 8;
      this._dyn.glRaySteps = 36;
      this._dyn.glPasses = 1;
    } else if (p === 'high') {
      this._dyn.wgWorkgroups = Math.min(this.wgMaxWorkgroups, Math.round(2048 * Math.sqrt(s * 0.5)));
      this._dyn.wgIters = Math.min(512, Math.round(48 + s * 10));
      this._dyn.wgDispatches = Math.max(2, Math.min(12, Math.ceil(s / 3)));
      this._dyn.wgRasterPasses = 1;
      this._dyn.wgRasterExtraIters = Math.min(96, Math.round(16 + s * 2));
      this._dyn.glSize = Math.min(this.glMaxTexSize, s >= 8 ? 2048 : 1536);
      this._dyn.glExtraIters = Math.min(384, Math.round(24 + s * 10));
      this._dyn.glRaySteps = 44;
      this._dyn.glPasses = Math.max(2, Math.min(12, Math.ceil(s / 3)));
    } else if (p === 'extreme') {
      this._dyn.wgWorkgroups = Math.min(this.wgMaxWorkgroups, Math.round(4096 * Math.sqrt(s * 0.5)));
      this._dyn.wgIters = Math.min(768, Math.round(96 + s * 20));
      this._dyn.wgDispatches = Math.max(4, Math.min(24, Math.ceil(s * 0.75)));
      this._dyn.wgRasterPasses = 2;
      this._dyn.wgRasterExtraIters = Math.min(192, Math.round(32 + s * 4));
      this._dyn.glSize = Math.min(this.glMaxTexSize, s >= 6 ? 3072 : 2048);
      this._dyn.glExtraIters = Math.min(640, Math.round(64 + s * 18));
      this._dyn.glRaySteps = 52;
      this._dyn.glPasses = Math.max(3, Math.min(24, Math.ceil(s * 0.75)));
    } else {
      // Insane / Overdrive — maximum flagship silicon saturation (RTX 4090/5090/M4 Max)
      this._dyn.wgWorkgroups = this.wgMaxWorkgroups;
      this._dyn.wgIters = Math.min(1024, Math.round(192 + s * 26));
      this._dyn.wgDispatches = Math.max(6, Math.min(32, Math.ceil(s * 1.1)));
      this._dyn.wgRasterPasses = Math.max(2, Math.min(6, Math.ceil(s / 6)));
      this._dyn.wgRasterExtraIters = Math.min(256, Math.round(64 + s * 6));
      this._dyn.glSize = Math.min(this.glMaxTexSize, s >= 4 ? 4096 : 2048);
      this._dyn.glExtraIters = Math.min(1024, Math.round(128 + s * 28));
      this._dyn.glRaySteps = 60;
      this._dyn.glPasses = Math.max(4, Math.min(48, Math.ceil(s * 1.5)));
    }
  }

  start(config = {}) {
    this.config = { ...this.config, ...config };
    if (this.config.engine === 'webgpu' && !this.capabilities.webgpu) {
      this.config.engine = 'webgl2';
    }
    if (this.config.engine === 'none' || (!this.capabilities.webgpu && !this.capabilities.webgl2)) {
      if (this.onError) this.onError('unsupported');
      return;
    }

    if (this.running) {
      this.updateConfig(config);
      return;
    }

    this.running = true;
    this._runToken++;
    const token = this._runToken;
    this._startTime = performance.now();

    this._dyn.emaBatchMs = 18;
    this._dyn.inFlight = 0;
    this._dyn.windowBatches = 0;
    this._dyn.windowFlops = 0;
    this._dyn.windowPixels = 0;
    this._dyn.windowStart = performance.now();
    this._dyn.lastBatchMs = 0;
    this._dyn.samplesGflops = [];
    this._glPending = [];

    this._applyPresetInitialLoad();

    if (this.config.engine === 'webgpu') {
      this._startWebGPULoop(token);
    } else {
      this._startWebGL2Loop(token);
    }

    this._telemetryTimer = setInterval(() => {
      if (this.running && token === this._runToken) {
        this._emitTelemetry();
      }
    }, 500);
  }

  // ==================== WEBGPU DEEP-QUEUE COMPUTE + CONCURRENT RASTER ====================
  _startWebGPULoop(token) {
    const maxInFlight = () => {
      if (!this.config.uncapVsync) return 1;
      if (this.config.preset === 'overdrive') return 6;
      if (this.config.preset === 'extreme') return 5;
      return 4;
    };

    const submitComputeBatch = () => {
      if (!this.running || token !== this._runToken || !this.gpuDevice) return;
      const device = this.gpuDevice;

      const wg = Math.max(128, Math.min(this.wgMaxWorkgroups, Math.round(this._dyn.wgWorkgroups)));
      const iters = Math.max(8, Math.min(1024, Math.round(this._dyn.wgIters)));
      const dispatches = Math.max(1, Math.min(32, Math.round(this._dyn.wgDispatches)));
      const nowSec = (performance.now() - this._startTime) * 0.001;

      const paramData = new ArrayBuffer(16);
      const u32View = new Uint32Array(paramData);
      const f32View = new Float32Array(paramData);
      u32View[0] = wg * 256;
      u32View[1] = iters;
      u32View[2] = this.wgElementCount - 1; // guaranteed power-of-2 mask
      f32View[3] = nowSec % 62.83;
      device.queue.writeBuffer(this.wgUniformBuf, 0, paramData);

      const encoder = device.createCommandEncoder();

      // Stage 1: Heavy Compute Shader passes (Shader Cores + SFU + VRAM/L2 Bandwidth)
      for (let d = 0; d < dispatches; d++) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.wgComputePipeline);
        pass.setBindGroup(0, (d & 1) === 0 ? this.wgBindAB : this.wgBindBA);
        pass.dispatchWorkgroups(wg, 1, 1);
        pass.end();
      }

      // Stage 2: Optional Concurrent Offscreen Raster Pass (fires Rasterizer + Fragment + ROP simultaneously!)
      let rasterFlops = 0;
      if (this.config.dualPipeline && this.wgOffscreenRasterPipeline) {
        const targetTexSize = (this.config.preset === 'extreme' || this.config.preset === 'overdrive') ? 2048 : 1024;
        this._ensureWgOffscreenTex(targetTexSize);
        const rPasses = Math.max(1, Math.min(6, Math.round(this._dyn.wgRasterPasses)));
        const rExtra = Math.max(0, Math.min(256, Math.round(this._dyn.wgRasterExtraIters)));

        const rUniform = new Float32Array([nowSec, 1.0, rExtra, dispatches * 0.17]);
        device.queue.writeBuffer(this.wgOffscreenUniformBuf, 0, rUniform);

        for (let rp = 0; rp < rPasses; rp++) {
          const rpass = encoder.beginRenderPass({
            colorAttachments: [{
              view: this.wgOffscreenView,
              loadOp: 'clear',
              storeOp: 'store',
              clearValue: { r: 0.9, g: 0.92, b: 0.96, a: 1.0 }
            }]
          });
          rpass.setPipeline(this.wgOffscreenRasterPipeline);
          rpass.setBindGroup(0, this.wgOffscreenBind);
          rpass.draw(3, 1, 0, 0);
          rpass.end();
        }
        rasterFlops = this.wgOffscreenSize * this.wgOffscreenSize * rPasses * (680 + rExtra * 48);
      }

      const tSubmit = performance.now();
      this._dyn.inFlight++;
      const depthAtSubmit = this._dyn.inFlight;
      device.queue.submit([encoder.finish()]);

      // 192 FP32 FLOPs per iteration per thread x 256 threads per workgroup
      const computeFlops = wg * 256 * iters * 192 * dispatches;
      const batchFlops = computeFlops + rasterFlops;

      device.queue.onSubmittedWorkDone().then(() => {
        if (!this.running || token !== this._runToken) return;
        const tDone = performance.now();
        const elapsed = Math.max(0.2, tDone - tSubmit);
        const effectiveMs = elapsed / Math.max(1, depthAtSubmit);
        this._dyn.inFlight = Math.max(0, this._dyn.inFlight - 1);
        this._dyn.lastBatchMs = effectiveMs;
        this._dyn.emaBatchMs = this._dyn.emaBatchMs * 0.65 + effectiveMs * 0.35;
        this._dyn.windowBatches++;
        this._dyn.windowFlops += batchFlops;

        this._adaptWebGPULoad(effectiveMs);

        if (this.config.uncapVsync) {
          while (this.running && token === this._runToken && this._dyn.inFlight < maxInFlight()) {
            submitComputeBatch();
          }
        }
      }).catch(() => {
        if (this.running && token === this._runToken) {
          this.stop();
          if (this.onError) this.onError('context_lost');
        }
      });
    };

    const initialBurst = maxInFlight();
    for (let i = 0; i < initialBurst; i++) {
      submitComputeBatch();
    }

    const renderFrame = () => {
      if (!this.running || token !== this._runToken) return;

      if (!this.config.uncapVsync && this._dyn.inFlight < 1) {
        submitComputeBatch();
      }

      if (this.config.showPreview && this.gpuContext && this.gpuDevice && this.wgPreviewPipeline) {
        try {
          const { w, h } = this._syncPreviewCanvasSize(this.webgpuCanvas);
          const uData = new Float32Array([
            (performance.now() - this._startTime) * 0.001,
            w / Math.max(1, h),
            0.0, // Keep live preview silky smooth; offscreen furnace does the heavy lifting
            0.0
          ]);
          this.gpuDevice.queue.writeBuffer(this.wgPreviewUniformBuf, 0, uData);
          const encoder = this.gpuDevice.createCommandEncoder();
          const texView = this.gpuContext.getCurrentTexture().createView();
          const rpass = encoder.beginRenderPass({
            colorAttachments: [{
              view: texView,
              loadOp: 'clear',
              storeOp: 'store',
              clearValue: { r: 0.90, g: 0.92, b: 0.96, a: 1.0 }
            }]
          });
          rpass.setPipeline(this.wgPreviewPipeline);
          rpass.setBindGroup(0, this.wgPreviewBind);
          rpass.draw(3, 1, 0, 0);
          rpass.end();
          this.gpuDevice.queue.submit([encoder.finish()]);
        } catch (_) {}
      }

      this._rafId = requestAnimationFrame(renderFrame);
    };
    this._rafId = requestAnimationFrame(renderFrame);
  }

  _adaptWebGPULoad(batchMs) {
    // TDR watchdog guard
    if (batchMs > 380) {
      if (this._dyn.wgDispatches > 1) {
        this._dyn.wgDispatches = Math.max(1, Math.floor(this._dyn.wgDispatches * 0.6));
      } else if (this._dyn.wgIters > 16) {
        this._dyn.wgIters = Math.max(16, Math.floor(this._dyn.wgIters * 0.6));
      } else {
        this._dyn.wgWorkgroups = Math.max(256, Math.floor(this._dyn.wgWorkgroups * 0.6));
      }
      return;
    }

    if (this.config.preset !== 'auto') return;

    // Aggressive ramp up to lock even an RTX 5090 at 100% in <1s
    const targetMs = 28 + this.config.scale * 2.2;
    const ema = this._dyn.emaBatchMs;

    if (ema < targetMs * 0.72) {
      if (this._dyn.wgWorkgroups < this.wgMaxWorkgroups) {
        this._dyn.wgWorkgroups = Math.min(this.wgMaxWorkgroups, Math.ceil(this._dyn.wgWorkgroups * 1.85));
      } else if (this._dyn.wgIters < 768) {
        this._dyn.wgIters = Math.min(768, Math.ceil(this._dyn.wgIters * 1.65));
      } else if (this._dyn.wgDispatches < 32) {
        this._dyn.wgDispatches = Math.min(32, Math.max(this._dyn.wgDispatches + 1, Math.ceil(this._dyn.wgDispatches * 1.4)));
      } else if (this.config.dualPipeline && this._dyn.wgRasterPasses < 6) {
        this._dyn.wgRasterPasses++;
        this._dyn.wgRasterExtraIters = Math.min(192, this._dyn.wgRasterExtraIters + 24);
      }
    } else if (ema > targetMs * 1.38) {
      if (this._dyn.wgDispatches > 1) {
        this._dyn.wgDispatches = Math.max(1, Math.floor(this._dyn.wgDispatches * 0.78));
      } else if (this._dyn.wgIters > 16) {
        this._dyn.wgIters = Math.max(16, Math.floor(this._dyn.wgIters * 0.78));
      } else if (this._dyn.wgWorkgroups > 256) {
        this._dyn.wgWorkgroups = Math.max(256, Math.floor(this._dyn.wgWorkgroups * 0.78));
      }
    }
  }

  // ==================== WEBGL2 4-MRT MULTI-PASS FBO + DEEP FENCE QUEUE ====================
  _startWebGL2Loop(token) {
    const gl = this.gl;
    if (!gl) return;

    const maxInFlight = () => {
      if (!this.config.uncapVsync) return 1;
      if (this.config.preset === 'overdrive') return 5;
      if (this.config.preset === 'extreme') return 4;
      return 3;
    };

    const submitGlBatch = (doPreviewBlit) => {
      if (!this.running || token !== this._runToken || !this.gl) return;
      if (gl.isContextLost()) {
        this.stop();
        if (this.onError) this.onError('context_lost');
        return;
      }

      const size = Math.max(256, Math.min(this.glMaxTexSize, Math.round(this._dyn.glSize)));
      this._ensureGlFbo(size);

      const extraIters = Math.max(1, Math.min(1024, Math.round(this._dyn.glExtraIters)));
      const raySteps = Math.max(16, Math.min(64, Math.round(this._dyn.glRaySteps)));
      const passes = Math.max(1, Math.min(48, Math.round(this._dyn.glPasses)));

      const tNowSec = (performance.now() - this._startTime) * 0.001;
      const previewDim = this.config.showPreview
        ? this._syncPreviewCanvasSize(this.webglCanvas)
        : { w: 400, h: 180 };
      const aspect = previewDim.w / Math.max(1, previewDim.h);

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.glFbo);
      gl.viewport(0, 0, this.glFboSize, this.glFboSize);
      gl.useProgram(this.glStressProg);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.glNoiseTex);
      gl.uniform1i(this.glUniforms.uNoiseTex, 1);
      gl.uniform1f(this.glUniforms.uTime, tNowSec);
      gl.uniform1f(this.glUniforms.uAspect, aspect);
      gl.uniform1i(this.glUniforms.uExtraIters, extraIters);
      gl.uniform1i(this.glUniforms.uRaySteps, raySteps);

      for (let p = 0; p < passes; p++) {
        gl.uniform1f(this.glUniforms.uPassOffset, p + (tNowSec % 10.0));
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }

      if (doPreviewBlit && this.config.showPreview && this.glFboTextures[0]) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, previewDim.w, previewDim.h);
        gl.useProgram(this.glBlitProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.glFboTextures[0]);
        gl.uniform1i(this.glUniforms.uTex, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      } else {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      }

      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();

      const totalPixels = this.glFboSize * this.glFboSize * passes * this.glMrtCount;
      // Conservative FLOPs/fragment: raymarch (~700) + extraIters * 192 (8x vec4 dual-issue + SFU)
      const estFlops = (this.glFboSize * this.glFboSize * passes) * (700 + extraIters * 192);

      this._glPending.push({
        sync,
        tSubmit: performance.now(),
        pixels: totalPixels,
        flops: estFlops,
        inFlightAtSubmit: this._glPending.length + 1
      });
      this._dyn.inFlight = this._glPending.length;
    };

    const pollFences = () => {
      if (!this.running || token !== this._runToken || !this.gl) return;
      while (this._glPending.length > 0) {
        const head = this._glPending[0];
        let signaled = true;
        if (head.sync) {
          const status = gl.getSyncParameter(head.sync, gl.SYNC_STATUS);
          signaled = status === gl.SIGNALED;
        }
        if (!signaled) break;

        this._glPending.shift();
        if (head.sync) gl.deleteSync(head.sync);

        const tDone = performance.now();
        const elapsed = Math.max(0.2, tDone - head.tSubmit);
        const effectiveMs = elapsed / Math.max(1, head.inFlightAtSubmit);
        this._dyn.lastBatchMs = effectiveMs;
        this._dyn.emaBatchMs = this._dyn.emaBatchMs * 0.65 + effectiveMs * 0.35;
        this._dyn.windowBatches++;
        this._dyn.windowPixels += head.pixels;
        this._dyn.windowFlops += head.flops;
        this._dyn.inFlight = this._glPending.length;

        this._adaptWebGL2Load(effectiveMs);
      }
    };

    let lastPreviewMs = 0;
    const pumpUncapped = () => {
      if (!this.running || token !== this._runToken) return;
      pollFences();

      if (this.config.uncapVsync) {
        const limit = maxInFlight();
        const now = performance.now();
        while (this._glPending.length < limit) {
          const shouldBlit = this.config.showPreview && (now - lastPreviewMs >= 16);
          if (shouldBlit) lastPreviewMs = now;
          submitGlBatch(shouldBlit);
        }
      }
      // 1ms lightweight check with 3-5 deep queue -> zero GPU bubbles AND <0.1% CPU polling overhead
      this._pollTimer = setTimeout(pumpUncapped, 1);
    };

    const onRaf = () => {
      if (!this.running || token !== this._runToken) return;
      pollFences();
      if (!this.config.uncapVsync) {
        if (this._glPending.length < 1) {
          submitGlBatch(true);
        }
      } else if (this.config.showPreview && this._glPending.length < maxInFlight()) {
        lastPreviewMs = performance.now();
        submitGlBatch(true);
      }
      this._rafId = requestAnimationFrame(onRaf);
    };

    submitGlBatch(true);
    this._pollTimer = setTimeout(pumpUncapped, 1);
    this._rafId = requestAnimationFrame(onRaf);
  }

  _adaptWebGL2Load(batchMs) {
    // TDR safety guard
    if (batchMs > 380) {
      if (this._dyn.glPasses > 1) {
        this._dyn.glPasses = Math.max(1, Math.floor(this._dyn.glPasses * 0.6));
      } else if (this._dyn.glExtraIters > 8) {
        this._dyn.glExtraIters = Math.max(8, Math.floor(this._dyn.glExtraIters * 0.6));
      } else if (this._dyn.glSize > 512) {
        this._dyn.glSize = Math.max(512, Math.floor(this._dyn.glSize * 0.75));
      }
      return;
    }

    if (this.config.preset !== 'auto') return;

    const targetMs = 28 + this.config.scale * 2.2;
    const ema = this._dyn.emaBatchMs;
    const sizeLadder = [512, 768, 1024, 1536, 2048, 3072, 4096].filter((s) => s <= this.glMaxTexSize);

    if (ema < targetMs * 0.72) {
      const curIdx = sizeLadder.findIndex((s) => s >= this._dyn.glSize);
      if (curIdx >= 0 && curIdx < sizeLadder.length - 1 && this._dyn.glExtraIters >= 16) {
        this._dyn.glSize = sizeLadder[curIdx + 1];
      } else if (this._dyn.glExtraIters < 640) {
        this._dyn.glExtraIters = Math.min(640, Math.max(this._dyn.glExtraIters + 8, Math.ceil(this._dyn.glExtraIters * 1.65)));
      } else if (this._dyn.glPasses < 48) {
        this._dyn.glPasses = Math.min(48, Math.max(this._dyn.glPasses + 1, Math.ceil(this._dyn.glPasses * 1.4)));
      }
    } else if (ema > targetMs * 1.38) {
      if (this._dyn.glPasses > 1) {
        this._dyn.glPasses = Math.max(1, Math.floor(this._dyn.glPasses * 0.78));
      } else if (this._dyn.glExtraIters > 8) {
        this._dyn.glExtraIters = Math.max(8, Math.floor(this._dyn.glExtraIters * 0.78));
      } else {
        const curIdx = sizeLadder.findIndex((s) => s >= this._dyn.glSize);
        if (curIdx > 0) this._dyn.glSize = sizeLadder[curIdx - 1];
      }
    }
  }

  _emitTelemetry() {
    const now = performance.now();
    const dtSec = Math.max(0.1, (now - this._dyn.windowStart) / 1000);
    const batchesPerSec = this._dyn.windowBatches / dtSec;
    const gflops = (this._dyn.windowFlops / dtSec) / 1e9;

    if (this._dyn.windowBatches > 0) {
      this._dyn.samplesGflops.push(gflops);
      if (this._dyn.samplesGflops.length > 7200) this._dyn.samplesGflops.shift();
    }

    this._dyn.windowBatches = 0;
    this._dyn.windowFlops = 0;
    this._dyn.windowPixels = 0;
    this._dyn.windowStart = now;

    const spec =
      this.config.engine === 'webgpu'
        ? `${this._dyn.wgWorkgroups}wg×${this._dyn.wgIters}it×${this._dyn.wgDispatches}${this.config.dualPipeline ? ' +Raster' : ''}`
        : `${this.glFboSize}²×${this.glMrtCount}MRT×${this._dyn.glPasses}p (${this._dyn.glExtraIters}it)`;

    if (this.onTelemetry) {
      this.onTelemetry({
        running: this.running,
        engine: this.config.engine,
        gflops,
        batchesPerSec,
        batchMs: this._dyn.emaBatchMs,
        spec
      });
    }
  }

  getSummary() {
    const avgGflops =
      this._dyn.samplesGflops.length > 0
        ? this._dyn.samplesGflops.reduce((a, b) => a + b, 0) / this._dyn.samplesGflops.length
        : 0;
    return {
      engine: this.config.engine,
      avgGflops
    };
  }

  stop() {
    this.running = false;
    this._runToken++;

    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
    if (this._pollTimer) {
      clearTimeout(this._pollTimer);
      this._pollTimer = null;
    }
    if (this._telemetryTimer) {
      clearInterval(this._telemetryTimer);
      this._telemetryTimer = null;
    }

    if (this.gl && this._glPending.length > 0) {
      for (const item of this._glPending) {
        if (item.sync) {
          try {
            this.gl.deleteSync(item.sync);
          } catch (_) {}
        }
      }
      this._glPending = [];
    }
    this._dyn.inFlight = 0;
  }

  destroy() {
    this.stop();
    if (this.webglCanvas) {
      this.webglCanvas.removeEventListener('webglcontextlost', this._onContextLost);
    }
    if (this.wgOffscreenTex) {
      try {
        this.wgOffscreenTex.destroy();
      } catch (_) {}
    }
    if (this.gpuDevice) {
      try {
        this.gpuDevice.destroy();
      } catch (_) {}
    }
  }
}

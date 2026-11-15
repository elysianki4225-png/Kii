import { I18N } from './i18n.js';
import { CpuStressEngine } from './cpu-stress.js';
import { GpuStressEngine } from './gpu-stress.js';
import { RamStressEngine } from './ram-stress.js';
import { Sparkline } from './sparkline.js';

const detectedCores = Math.max(1, navigator.hardwareConcurrency || 8);
const detectedDeviceMemGB = typeof navigator.deviceMemory === 'number' ? navigator.deviceMemory : null;

const state = {
  lang: 'zh',
  durationPreset: '0', // '60' | '300' | '900' | '1800' | '3600' | '0' | 'custom'
  customMinutes: 10,
  customSeconds: 0,

  // Session runtime
  sessionActive: false,
  sessionStartMs: 0,
  sessionTargetSec: 0,
  ranCpuInSession: false,
  ranGpuInSession: false,
  ranRamInSession: false,
  timerInterval: null,
  wakeLock: null,

  // CPU config
  cpuThreads: detectedCores,
  cpuWorkload: 'simd',
  cpuDuty: 1.0,

  // GPU config
  gpuEngine: 'webgl2',
  gpuPreset: 'auto',
  gpuScale: 8,
  gpuUncap: true,
  gpuDualPipe: true,
  gpuShowPreview: true,
  gpuHasError: false,

  // RAM config
  ramTargetMB: 2048,
  ramMode: 'bandwidth',
  ramThreads: Math.max(2, Math.min(16, detectedCores)),
  ramAllocating: false,
  ramOomHit: false,
  ramAllocatedMB: 0,
  ramErrors: 0
};

const $ = (id) => document.getElementById(id);

const els = {
  hwCpuText: $('hwCpuText'),
  hwGpuText: $('hwGpuText'),
  hwRamText: $('hwRamText'),
  wakeLockPill: $('wakeLockPill'),
  langToggleBtn: $('langToggleBtn'),
  escStopBtn: $('escStopBtn'),

  durationBtns: Array.from(document.querySelectorAll('.duration-presets .seg-btn')),
  customDurationRow: $('customDurationRow'),
  customMinutesInput: $('customMinutesInput'),
  customSecondsInput: $('customSecondsInput'),
  timerStateTag: $('timerStateTag'),
  timerDisplay: $('timerDisplay'),
  timerProgressFill: $('timerProgressFill'),
  masterToggleBtn: $('masterToggleBtn'),
  masterToggleLabel: $('masterToggleLabel'),

  // CPU
  cpuCard: $('cpuCard'),
  cpuLed: $('cpuLed'),
  cpuStatusText: $('cpuStatusText'),
  cpuToggleBtn: $('cpuToggleBtn'),
  cpuToggleLabel: $('cpuToggleLabel'),
  cpuRateVal: $('cpuRateVal'),
  cpuRateUnit: $('cpuRateUnit'),
  cpuPeakVal: $('cpuPeakVal'),
  cpuActiveVal: $('cpuActiveVal'),
  cpuTargetThreadsUnit: $('cpuTargetThreadsUnit'),
  cpuKernelBadge: $('cpuKernelBadge'),
  cpuSatVal: $('cpuSatVal'),
  cpuDutySub: $('cpuDutySub'),
  cpuSparkline: $('cpuSparkline'),
  cpuThreadsSlider: $('cpuThreadsSlider'),
  cpuThreadsInput: $('cpuThreadsInput'),
  cpuThreadChips: Array.from(document.querySelectorAll('#cpuThreadChips .chip-btn')),
  cpuThreadsHint: $('cpuThreadsHint'),
  cpuWorkloadBtns: Array.from(document.querySelectorAll('#cpuWorkloadGroup .seg-btn')),
  cpuDutyBtns: Array.from(document.querySelectorAll('#cpuDutyGroup .seg-btn')),

  // GPU
  gpuCard: $('gpuCard'),
  gpuLed: $('gpuLed'),
  gpuStatusText: $('gpuStatusText'),
  gpuToggleBtn: $('gpuToggleBtn'),
  gpuToggleLabel: $('gpuToggleLabel'),
  webgpuCanvas: $('webgpuCanvas'),
  webglCanvas: $('webglCanvas'),
  gpuOverlay: $('gpuOverlay'),
  gpuOverlayText: $('gpuOverlayText'),
  gpuPreviewToggleBtn: $('gpuPreviewToggleBtn'),
  gpuThroughputVal: $('gpuThroughputVal'),
  gpuThroughputUnit: $('gpuThroughputUnit'),
  gpuSpecVal: $('gpuSpecVal'),
  gpuRateVal: $('gpuRateVal'),
  gpuRateUnit: $('gpuRateUnit'),
  gpuEngineBadge: $('gpuEngineBadge'),
  gpuBatchMsVal: $('gpuBatchMsVal'),
  gpuModeBadge: $('gpuModeBadge'),
  gpuEngineBtns: Array.from(document.querySelectorAll('#gpuEngineGroup .seg-btn')),
  btnEngineWebgpu: $('btnEngineWebgpu'),
  btnEngineWebgl2: $('btnEngineWebgl2'),
  gpuEngineHint: $('gpuEngineHint'),
  gpuPresetBtns: Array.from(document.querySelectorAll('#gpuPresetGroup .seg-btn')),
  gpuOverdriveHint: $('gpuOverdriveHint'),
  gpuScaleSlider: $('gpuScaleSlider'),
  gpuScaleReadout: $('gpuScaleReadout'),
  gpuUncapCheckbox: $('gpuUncapCheckbox'),
  gpuDualPipeRow: $('gpuDualPipeRow'),
  gpuDualPipeCheckbox: $('gpuDualPipeCheckbox'),

  // RAM
  ramCard: $('ramCard'),
  ramLed: $('ramLed'),
  ramStatusText: $('ramStatusText'),
  ramToggleBtn: $('ramToggleBtn'),
  ramToggleLabel: $('ramToggleLabel'),
  ramBwVal: $('ramBwVal'),
  ramPeakVal: $('ramPeakVal'),
  ramAllocVal: $('ramAllocVal'),
  ramAllocUnit: $('ramAllocUnit'),
  ramTargetSub: $('ramTargetSub'),
  ramPassesVal: $('ramPassesVal'),
  ramErrorsSub: $('ramErrorsSub'),
  ramSparkline: $('ramSparkline'),
  ramSizeReadout: $('ramSizeReadout'),
  ramSizePresetBtns: Array.from(document.querySelectorAll('#ramSizePresetGroup .seg-btn')),
  ramSizeSlider: $('ramSizeSlider'),
  ramModeBtns: Array.from(document.querySelectorAll('#ramModeGroup .seg-btn')),
  ramThreadsSlider: $('ramThreadsSlider'),
  ramThreadsReadout: $('ramThreadsReadout'),
  ramOomHint: $('ramOomHint'),

  // Summary & Error
  summaryStrip: $('summaryStrip'),
  summaryDurationVal: $('summaryDurationVal'),
  summaryCpuItem: $('summaryCpuItem'),
  summaryCpuVal: $('summaryCpuVal'),
  summaryGpuItem: $('summaryGpuItem'),
  summaryGpuVal: $('summaryGpuVal'),
  summaryRamItem: $('summaryRamItem'),
  summaryRamVal: $('summaryRamVal'),
  summaryDismissBtn: $('summaryDismissBtn'),

  errorBanner: $('errorBanner'),
  errorBannerText: $('errorBannerText'),
  errorDismissBtn: $('errorDismissBtn')
};

function t(key) {
  return (I18N[state.lang] && I18N[state.lang][key]) || I18N.zh[key] || key;
}

function formatDuration(totalSec) {
  const s = Math.max(0, Math.floor(totalSec));
  const hrs = Math.floor(s / 3600);
  const mins = Math.floor((s % 3600) / 60);
  const secs = s % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return hrs > 0 ? `${pad(hrs)}:${pad(mins)}:${pad(secs)}` : `${pad(mins)}:${pad(secs)}`;
}

function formatCpuRate(opsPerSec, workload) {
  if (workload === 'crypto') {
    return { val: (opsPerSec / 1e6).toFixed(2), unit: t('unitMh') };
  }
  if (opsPerSec >= 1e9 || workload === 'simd') {
    return { val: (opsPerSec / 1e9).toFixed(1), unit: t('unitGflops') };
  }
  return { val: (opsPerSec / 1e6).toFixed(1), unit: t('unitMops') };
}

function formatGflops(gflops) {
  if (gflops >= 1000) {
    return { val: (gflops / 1000).toFixed(2), unit: t('unitTflops') };
  }
  return { val: gflops.toFixed(1), unit: t('unitGflops') };
}

function formatMemoryMB(mb) {
  if (mb >= 1024) {
    const gb = mb / 1024;
    const val = Number.isInteger(gb) ? gb.toFixed(1) : gb.toFixed(2);
    return { val, unit: 'GB', text: `${val} GB` };
  }
  const val = String(Math.round(mb));
  return { val, unit: 'MB', text: `${val} MB` };
}

function updateSliderFill(slider) {
  const min = Number(slider.min) || 0;
  const max = Number(slider.max) || 100;
  const val = Number(slider.value) || min;
  const pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
  slider.style.setProperty('--pct', `${pct.toFixed(1)}%`);
}

function getSelectedDurationSec() {
  if (state.durationPreset === '0') return 0;
  if (state.durationPreset === 'custom') {
    const mins = Math.max(0, Math.min(1440, parseInt(els.customMinutesInput.value, 10) || 0));
    const secs = Math.max(0, Math.min(59, parseInt(els.customSecondsInput.value, 10) || 0));
    return Math.max(1, mins * 60 + secs);
  }
  return Math.max(0, parseInt(state.durationPreset, 10) || 0);
}

// ==================== Engines & Sparklines ====================
const cpuSparkline = new Sparkline(els.cpuSparkline, {
  maxPoints: 48,
  strokeColor: '#2F6FED',
  fillTop: 'rgba(47, 111, 237, 0.2)',
  fillBottom: 'rgba(47, 111, 237, 0.0)'
});

const ramSparkline = new Sparkline(els.ramSparkline, {
  maxPoints: 48,
  strokeColor: '#5B64EC',
  fillTop: 'rgba(91, 100, 236, 0.2)',
  fillBottom: 'rgba(91, 100, 236, 0.0)'
});

const cpuEngine = new CpuStressEngine((stats) => {
  const cur = formatCpuRate(stats.rateOps, stats.workload);
  const peak = formatCpuRate(stats.peakOps, stats.workload);
  els.cpuRateVal.textContent = cur.val;
  els.cpuRateUnit.textContent = cur.unit;
  els.cpuPeakVal.textContent = `${peak.val} ${peak.unit}`;
  els.cpuActiveVal.textContent = String(stats.activeWorkers);
  els.cpuTargetThreadsUnit.textContent = `/ ${state.cpuThreads}`;
  els.cpuSatVal.textContent = Math.round(stats.saturationPct).toString();
  cpuSparkline.push(stats.rateOps / 1e6);
});

const gpuEngine = new GpuStressEngine(
  els.webgpuCanvas,
  els.webglCanvas,
  (stats) => {
    const formatted = formatGflops(stats.gflops);
    els.gpuThroughputVal.textContent = formatted.val;
    els.gpuThroughputUnit.textContent = formatted.unit;
    els.gpuSpecVal.textContent = stats.spec;
    els.gpuRateVal.textContent = stats.batchesPerSec.toFixed(1);
    els.gpuRateUnit.textContent = t('batchesPerSec');
    els.gpuBatchMsVal.textContent = stats.batchMs.toFixed(1);
    els.gpuEngineBadge.textContent =
      stats.engine === 'webgpu' ? t('gpuEngineWebgpu') : t('gpuEngineWebgl2');
  },
  (errCode) => {
    state.gpuHasError = true;
    showError(errCode === 'unsupported' ? t('gpuUnsupported') : t('gpuContextLost'));
    syncUIState();
    checkSessionAutoEnd(false);
  }
);

const ramEngine = new RamStressEngine((stats) => {
  state.ramAllocating = stats.allocating;
  state.ramOomHit = stats.oomHit;
  state.ramAllocatedMB = stats.allocatedMB;
  state.ramErrors = stats.errors;

  els.ramBwVal.textContent = stats.gbs.toFixed(1);
  els.ramPeakVal.textContent = `${stats.peakGBs.toFixed(1)} ${t('unitGbs')}`;

  const allocFmt = formatMemoryMB(stats.allocatedMB);
  els.ramAllocVal.textContent = allocFmt.val;
  els.ramAllocUnit.textContent = allocFmt.unit;
  els.ramPassesVal.textContent = String(stats.passes);

  if (stats.errors > 0) {
    els.ramErrorsSub.textContent = t('ramErrorsFound').replace('{n}', String(stats.errors));
    els.ramErrorsSub.classList.add('ram-error-alert');
  } else if (stats.verifiedGB > 0.05) {
    els.ramErrorsSub.textContent = `${t('ramErrorsZero')} · ${t('ramVerifiedTotal')} ${stats.verifiedGB.toFixed(1)} GB`;
    els.ramErrorsSub.classList.remove('ram-error-alert');
  } else {
    els.ramErrorsSub.textContent = t('ramErrorsZero');
    els.ramErrorsSub.classList.remove('ram-error-alert');
  }

  if (stats.oomHit) {
    els.ramOomHint.textContent = t('ramOomNotice').replace('{mb}', String(Math.round(stats.allocatedMB)));
    els.ramOomHint.classList.remove('hidden');
  } else {
    els.ramOomHint.classList.add('hidden');
  }

  if (!stats.allocating) {
    ramSparkline.push(stats.gbs);
  }
  syncUIState();
});

// ==================== Wake Lock ====================
async function requestWakeLock() {
  if (!('wakeLock' in navigator) || state.wakeLock) return;
  try {
    state.wakeLock = await navigator.wakeLock.request('screen');
    els.wakeLockPill.classList.remove('hidden');
    state.wakeLock.addEventListener('release', () => {
      state.wakeLock = null;
      els.wakeLockPill.classList.add('hidden');
    });
  } catch (_) {}
}

async function releaseWakeLock() {
  if (!state.wakeLock) return;
  try {
    await state.wakeLock.release();
  } catch (_) {}
  state.wakeLock = null;
  els.wakeLockPill.classList.add('hidden');
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && state.sessionActive) {
    requestWakeLock();
  }
});

// ==================== Session & Timer Management ====================
function beginSessionIfNeeded() {
  if (state.sessionActive) return;
  state.sessionActive = true;
  state.sessionStartMs = performance.now();
  state.sessionTargetSec = getSelectedDurationSec();
  state.ranCpuInSession = cpuEngine.running;
  state.ranGpuInSession = gpuEngine.running;
  state.ranRamInSession = ramEngine.running;
  els.summaryStrip.classList.add('hidden');

  requestWakeLock();

  if (state.timerInterval) clearInterval(state.timerInterval);
  state.timerInterval = setInterval(tickSessionTimer, 200);
  tickSessionTimer();
}

function tickSessionTimer() {
  if (!state.sessionActive) return;
  const elapsedSec = (performance.now() - state.sessionStartMs) / 1000;

  if (state.sessionTargetSec > 0) {
    const remaining = Math.max(0, state.sessionTargetSec - elapsedSec);
    els.timerDisplay.textContent = formatDuration(Math.ceil(remaining));
    const pct = Math.min(100, (elapsedSec / state.sessionTargetSec) * 100);
    els.timerProgressFill.classList.remove('is-continuous');
    els.timerProgressFill.style.width = `${pct.toFixed(1)}%`;
    els.timerStateTag.textContent = t('timerRemaining');
    els.timerStateTag.className = 'timer-state-tag is-active';

    if (remaining <= 0) {
      stopAll(true);
    }
  } else {
    els.timerDisplay.textContent = formatDuration(elapsedSec);
    els.timerProgressFill.classList.add('is-continuous');
    els.timerProgressFill.style.width = '100%';
    els.timerStateTag.textContent = t('timerElapsed');
    els.timerStateTag.className = 'timer-state-tag is-active';
  }
}

function updateIdleTimerPreview() {
  if (state.sessionActive) return;
  const target = getSelectedDurationSec();
  els.timerDisplay.textContent = target > 0 ? formatDuration(target) : '∞';
  els.timerProgressFill.classList.remove('is-continuous');
  els.timerProgressFill.style.width = '0%';
  els.timerStateTag.textContent = t('timerReady');
  els.timerStateTag.className = 'timer-state-tag';
}

function checkSessionAutoEnd(completedByTimer = false) {
  if (!state.sessionActive) return;
  if (cpuEngine.running || gpuEngine.running || ramEngine.running) return;

  const elapsedSec = (performance.now() - state.sessionStartMs) / 1000;
  state.sessionActive = false;
  if (state.timerInterval) {
    clearInterval(state.timerInterval);
    state.timerInterval = null;
  }
  releaseWakeLock();

  if (completedByTimer) {
    els.timerDisplay.textContent = '00:00';
    els.timerProgressFill.classList.remove('is-continuous');
    els.timerProgressFill.style.width = '100%';
    els.timerStateTag.textContent = t('timerDone');
    els.timerStateTag.className = 'timer-state-tag is-done';
  } else {
    updateIdleTimerPreview();
  }

  if (elapsedSec >= 1.0 && (state.ranCpuInSession || state.ranGpuInSession || state.ranRamInSession)) {
    renderSummary(elapsedSec);
  }
}

function renderSummary(elapsedSec) {
  els.summaryDurationVal.textContent = formatDuration(elapsedSec);

  if (state.ranCpuInSession) {
    const cpuSum = cpuEngine.getSummary();
    const avg = formatCpuRate(cpuSum.avgOps, cpuSum.workload);
    const peak = formatCpuRate(cpuSum.peakOps, cpuSum.workload);
    els.summaryCpuVal.textContent = `${avg.val} / ${peak.val} ${peak.unit}`;
    els.summaryCpuItem.classList.remove('hidden');
  } else {
    els.summaryCpuItem.classList.add('hidden');
  }

  if (state.ranGpuInSession) {
    const gpuSum = gpuEngine.getSummary();
    const fmt = formatGflops(gpuSum.avgGflops);
    els.summaryGpuVal.textContent = `${fmt.val} ${fmt.unit}`;
    els.summaryGpuItem.classList.remove('hidden');
  } else {
    els.summaryGpuItem.classList.add('hidden');
  }

  if (state.ranRamInSession) {
    const ramSum = ramEngine.getSummary();
    const errText =
      ramSum.errors > 0
        ? t('ramErrorsFound').replace('{n}', String(ramSum.errors))
        : t('ramErrorsZero');
    els.summaryRamVal.textContent = `${ramSum.avgGBs.toFixed(1)} ${t('unitGbs')} · ${errText}`;
    els.summaryRamItem.classList.remove('hidden');
  } else {
    els.summaryRamItem.classList.add('hidden');
  }

  els.summaryStrip.classList.remove('hidden');
}

// ==================== Start / Stop Actions ====================
function startCpu() {
  cpuSparkline.reset();
  cpuEngine.start({
    threads: state.cpuThreads,
    workload: state.cpuWorkload,
    duty: state.cpuDuty
  });
  if (state.sessionActive) state.ranCpuInSession = true;
  else beginSessionIfNeeded();
  syncUIState();
}

function stopCpu() {
  cpuEngine.stop();
  els.cpuActiveVal.textContent = '0';
  els.cpuSatVal.textContent = '0';
  els.cpuRateVal.textContent = '0.0';
  syncUIState();
  checkSessionAutoEnd(false);
}

function startGpu() {
  if (!gpuEngine.capabilities.webgpu && !gpuEngine.capabilities.webgl2) {
    showError(t('gpuUnsupported'));
    return;
  }
  state.gpuHasError = false;
  hideError();
  gpuEngine.start({
    engine: state.gpuEngine,
    preset: state.gpuPreset,
    scale: state.gpuScale,
    uncapVsync: state.gpuUncap,
    dualPipeline: state.gpuDualPipe,
    showPreview: state.gpuShowPreview
  });
  if (state.sessionActive) state.ranGpuInSession = true;
  else beginSessionIfNeeded();
  syncUIState();
}

function stopGpu() {
  gpuEngine.stop();
  els.gpuThroughputVal.textContent = '0.0';
  els.gpuRateVal.textContent = '0.0';
  els.gpuBatchMsVal.textContent = '0.0';
  els.gpuSpecVal.textContent = '--';
  syncUIState();
  checkSessionAutoEnd(false);
}

function startRam() {
  ramSparkline.reset();
  state.ramAllocating = true;
  state.ramOomHit = false;
  state.ramAllocatedMB = 0;
  state.ramErrors = 0;
  els.ramOomHint.classList.add('hidden');
  els.ramErrorsSub.classList.remove('ram-error-alert');
  els.ramErrorsSub.textContent = t('ramErrorsZero');

  ramEngine.start({
    targetMB: state.ramTargetMB,
    mode: state.ramMode,
    threads: state.ramThreads
  });
  if (state.sessionActive) state.ranRamInSession = true;
  else beginSessionIfNeeded();
  syncUIState();
}

function stopRam() {
  ramEngine.stop();
  state.ramAllocating = false;
  state.ramAllocatedMB = 0;
  els.ramBwVal.textContent = '0.0';
  els.ramAllocVal.textContent = '0';
  els.ramAllocUnit.textContent = 'MB';
  els.ramPassesVal.textContent = '0';
  syncUIState();
  checkSessionAutoEnd(false);
}

function startAll() {
  if (!cpuEngine.running) startCpu();
  if (!gpuEngine.running && (gpuEngine.capabilities.webgpu || gpuEngine.capabilities.webgl2)) {
    startGpu();
  }
  if (!ramEngine.running) startRam();
  syncUIState();
}

function stopAll(completedByTimer = false) {
  if (cpuEngine.running) {
    cpuEngine.stop();
    els.cpuActiveVal.textContent = '0';
    els.cpuSatVal.textContent = '0';
    els.cpuRateVal.textContent = '0.0';
  }
  if (gpuEngine.running) {
    gpuEngine.stop();
    els.gpuThroughputVal.textContent = '0.0';
    els.gpuRateVal.textContent = '0.0';
    els.gpuBatchMsVal.textContent = '0.0';
    els.gpuSpecVal.textContent = '--';
  }
  if (ramEngine.running) {
    ramEngine.stop();
    state.ramAllocating = false;
    state.ramAllocatedMB = 0;
    els.ramBwVal.textContent = '0.0';
    els.ramAllocVal.textContent = '0';
    els.ramAllocUnit.textContent = 'MB';
    els.ramPassesVal.textContent = '0';
  }
  syncUIState();
  checkSessionAutoEnd(completedByTimer);
}

function showError(msg) {
  els.errorBannerText.textContent = msg;
  els.errorBanner.classList.remove('hidden');
}

function hideError() {
  els.errorBanner.classList.add('hidden');
}

// ==================== UI Synchronization ====================
function syncGpuCanvasVisibility() {
  const running = gpuEngine.running;
  const showWebgpu = running && state.gpuEngine === 'webgpu' && state.gpuShowPreview;
  const showWebgl = running && state.gpuEngine === 'webgl2' && state.gpuShowPreview;

  els.webgpuCanvas.classList.toggle('hidden', !showWebgpu);
  els.webglCanvas.classList.toggle('hidden', !showWebgl);

  if (!running) {
    els.gpuOverlay.classList.remove('is-hidden');
    els.gpuOverlayText.textContent = t('gpuIdleText');
  } else if (!state.gpuShowPreview) {
    els.gpuOverlay.classList.remove('is-hidden');
    els.gpuOverlayText.textContent = t('gpuHiddenText');
  } else {
    els.gpuOverlay.classList.add('is-hidden');
  }

  els.gpuPreviewToggleBtn.textContent = state.gpuShowPreview
    ? t('gpuHidePreview')
    : t('gpuShowPreview');
}

function syncUIState() {
  const cpuOn = cpuEngine.running;
  const gpuOn = gpuEngine.running;
  const ramOn = ramEngine.running;
  const anyOn = cpuOn || gpuOn || ramOn;

  // Master button
  els.masterToggleBtn.classList.toggle('is-running', anyOn);
  els.masterToggleLabel.textContent = anyOn ? t('stopAll') : t('startAll');

  // CPU module
  els.cpuCard.classList.toggle('is-running-cpu', cpuOn);
  els.cpuLed.className = cpuOn ? 'status-led is-cpu-on' : 'status-led';
  els.cpuStatusText.textContent = cpuOn ? t('statusRunning') : t('statusIdle');
  els.cpuToggleBtn.classList.toggle('is-running', cpuOn);
  els.cpuToggleLabel.textContent = cpuOn ? t('stop') : t('start');
  els.cpuTargetThreadsUnit.textContent = `/ ${state.cpuThreads}`;
  els.cpuDutySub.textContent = `${Math.round(state.cpuDuty * 100)}%`;

  if (state.cpuWorkload === 'simd') {
    els.cpuKernelBadge.textContent = cpuEngine.simdSupported ? t('cpuSimdActive') : t('cpuJsFallback');
    if (!cpuOn) els.cpuRateUnit.textContent = t('unitGflops');
  } else if (state.cpuWorkload === 'fp64') {
    els.cpuKernelBadge.textContent = t('workloadFp64');
    if (!cpuOn) els.cpuRateUnit.textContent = t('unitGflops');
  } else if (state.cpuWorkload === 'crypto') {
    els.cpuKernelBadge.textContent = t('workloadCrypto');
    if (!cpuOn) els.cpuRateUnit.textContent = t('unitMh');
  } else {
    els.cpuKernelBadge.textContent = t('workloadExtreme');
    if (!cpuOn) els.cpuRateUnit.textContent = t('unitMops');
  }

  // GPU module
  els.gpuCard.classList.toggle('is-running-gpu', gpuOn);
  if (state.gpuHasError && !gpuOn) {
    els.gpuLed.className = 'status-led is-error';
    els.gpuStatusText.textContent = t('statusError');
  } else {
    els.gpuLed.className = gpuOn ? 'status-led is-gpu-on' : 'status-led';
    els.gpuStatusText.textContent = gpuOn ? t('statusRunning') : t('statusIdle');
  }
  els.gpuToggleBtn.classList.toggle('is-running', gpuOn);
  els.gpuToggleLabel.textContent = gpuOn ? t('stop') : t('start');
  els.gpuEngineBadge.textContent =
    state.gpuEngine === 'webgpu' ? t('gpuEngineWebgpu') : t('gpuEngineWebgl2');

  const presetKeyMap = {
    auto: 'presetAuto',
    high: 'presetHigh',
    extreme: 'presetExtreme',
    overdrive: 'presetOverdrive'
  };
  els.gpuModeBadge.textContent = t(presetKeyMap[state.gpuPreset] || 'presetAuto');
  els.gpuOverdriveHint.classList.toggle('hidden', state.gpuPreset !== 'overdrive');
  els.gpuScaleReadout.textContent = `${state.gpuScale}×`;
  els.gpuDualPipeRow.classList.toggle('hidden', state.gpuEngine !== 'webgpu');

  // RAM module
  els.ramCard.classList.toggle('is-running-ram', ramOn);
  if (state.ramErrors > 0) {
    els.ramLed.className = 'status-led is-error';
  } else {
    els.ramLed.className = ramOn ? 'status-led is-ram-on' : 'status-led';
  }
  els.ramStatusText.textContent = !ramOn
    ? t('statusIdle')
    : state.ramAllocating
      ? t('statusAllocating')
      : t('statusRunning');
  els.ramToggleBtn.classList.toggle('is-running', ramOn);
  els.ramToggleLabel.textContent = ramOn ? t('stop') : t('start');

  const targetFmt = formatMemoryMB(state.ramTargetMB);
  els.ramSizeReadout.textContent = targetFmt.text;
  els.ramTargetSub.textContent = targetFmt.text;
  els.ramThreadsReadout.textContent = String(state.ramThreads);

  syncGpuCanvasVisibility();
}

function applyLanguage() {
  document.documentElement.lang = state.lang === 'zh' ? 'zh-CN' : 'en';
  els.langToggleBtn.textContent = state.lang === 'zh' ? 'EN' : '中';

  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    if (key) el.textContent = t(key);
  });

  els.hwCpuText.textContent = `${detectedCores} ${t('logicalCores')}${cpuEngine.simdSupported ? ' · SIMD128' : ''}`;
  els.cpuThreadsHint.textContent = t('cpuThreadsHint').replace('{cores}', String(detectedCores));

  const caps = gpuEngine.capabilities;
  const backendTag = caps.webgpu ? 'WebGPU' : caps.webgl2 ? `WebGL2 ${caps.mrtCount}MRT` : 'N/A';
  els.hwGpuText.textContent = caps.rendererName
    ? `${backendTag} · ${caps.rendererName}`
    : `${backendTag} · ${t('unknownGpu')}`;

  els.hwRamText.textContent = detectedDeviceMemGB
    ? t('ramDeviceApprox').replace('{gb}', String(detectedDeviceMemGB))
    : t('ramDetected');

  if (!caps.webgpu && caps.webgl2) {
    els.gpuEngineHint.textContent = t('gpuWebgpuUnavailable');
    els.gpuEngineHint.classList.remove('hidden');
  } else {
    els.gpuEngineHint.classList.add('hidden');
  }

  if (!ramEngine.running) {
    els.ramErrorsSub.textContent = t('ramErrorsZero');
  }

  if (!state.sessionActive) updateIdleTimerPreview();
  else tickSessionTimer();

  syncUIState();
}

// ==================== Event Wiring ====================
function setCpuThreads(value) {
  const clamped = Math.max(1, Math.min(256, Math.round(value) || 1));
  state.cpuThreads = clamped;
  const sliderMax = Number(els.cpuThreadsSlider.max) || 64;
  if (clamped > sliderMax) els.cpuThreadsSlider.max = String(clamped);
  els.cpuThreadsSlider.value = String(clamped);
  els.cpuThreadsInput.value = String(clamped);
  updateSliderFill(els.cpuThreadsSlider);
  cpuEngine.updateConfig({ threads: clamped });
  syncUIState();
}

function setRamTargetMB(mb) {
  const clamped = Math.max(256, Math.min(16384, Math.round(mb / 256) * 256));
  state.ramTargetMB = clamped;
  els.ramSizeSlider.value = String(clamped);
  updateSliderFill(els.ramSizeSlider);
  els.ramSizePresetBtns.forEach((btn) => {
    btn.classList.toggle('is-active', Number(btn.dataset.mb) === clamped);
  });
  ramEngine.updateConfig({ targetMB: clamped });
  syncUIState();
}

function wireEvents() {
  els.langToggleBtn.addEventListener('click', () => {
    state.lang = state.lang === 'zh' ? 'en' : 'zh';
    applyLanguage();
  });

  els.escStopBtn.addEventListener('click', () => stopAll(false));
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') stopAll(false);
  });

  // Duration presets
  els.durationBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      els.durationBtns.forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      state.durationPreset = btn.dataset.duration;
      els.customDurationRow.classList.toggle('hidden', state.durationPreset !== 'custom');

      if (state.sessionActive) {
        state.sessionTargetSec = getSelectedDurationSec();
        tickSessionTimer();
      } else {
        updateIdleTimerPreview();
      }
    });
  });

  const onCustomTimeChange = () => {
    if (state.durationPreset !== 'custom') return;
    if (state.sessionActive) {
      state.sessionTargetSec = getSelectedDurationSec();
      tickSessionTimer();
    } else {
      updateIdleTimerPreview();
    }
  };
  els.customMinutesInput.addEventListener('input', onCustomTimeChange);
  els.customSecondsInput.addEventListener('input', onCustomTimeChange);

  // Master toggle
  els.masterToggleBtn.addEventListener('click', () => {
    if (cpuEngine.running || gpuEngine.running || ramEngine.running) {
      stopAll(false);
    } else {
      startAll();
    }
  });

  // CPU controls
  els.cpuToggleBtn.addEventListener('click', () => {
    if (cpuEngine.running) stopCpu();
    else startCpu();
  });

  els.cpuThreadsSlider.addEventListener('input', () => {
    setCpuThreads(Number(els.cpuThreadsSlider.value));
  });
  els.cpuThreadsInput.addEventListener('change', () => {
    setCpuThreads(Number(els.cpuThreadsInput.value));
  });
  els.cpuThreadChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      if (chip.dataset.abs) setCpuThreads(Number(chip.dataset.abs));
      else if (chip.dataset.mult) setCpuThreads(Math.round(detectedCores * Number(chip.dataset.mult)));
    });
  });

  els.cpuWorkloadBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      els.cpuWorkloadBtns.forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      state.cpuWorkload = btn.dataset.workload;
      cpuEngine.updateConfig({ workload: state.cpuWorkload });
      syncUIState();
    });
  });

  els.cpuDutyBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      els.cpuDutyBtns.forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      state.cpuDuty = Number(btn.dataset.duty);
      cpuEngine.updateConfig({ duty: state.cpuDuty });
      syncUIState();
    });
  });

  // GPU controls
  els.gpuToggleBtn.addEventListener('click', () => {
    if (gpuEngine.running) stopGpu();
    else startGpu();
  });

  els.gpuPreviewToggleBtn.addEventListener('click', () => {
    state.gpuShowPreview = !state.gpuShowPreview;
    gpuEngine.updateConfig({ showPreview: state.gpuShowPreview });
    syncGpuCanvasVisibility();
  });

  els.gpuEngineBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      if (btn.disabled) return;
      els.gpuEngineBtns.forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      state.gpuEngine = btn.dataset.engine;
      gpuEngine.updateConfig({ engine: state.gpuEngine });
      syncUIState();
    });
  });

  els.gpuPresetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      els.gpuPresetBtns.forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      state.gpuPreset = btn.dataset.preset;
      gpuEngine.updateConfig({ preset: state.gpuPreset });
      syncUIState();
    });
  });

  els.gpuScaleSlider.addEventListener('input', () => {
    state.gpuScale = Number(els.gpuScaleSlider.value);
    updateSliderFill(els.gpuScaleSlider);
    gpuEngine.updateConfig({ scale: state.gpuScale });
    syncUIState();
  });

  els.gpuUncapCheckbox.addEventListener('change', () => {
    state.gpuUncap = els.gpuUncapCheckbox.checked;
    gpuEngine.updateConfig({ uncapVsync: state.gpuUncap });
  });

  els.gpuDualPipeCheckbox.addEventListener('change', () => {
    state.gpuDualPipe = els.gpuDualPipeCheckbox.checked;
    gpuEngine.updateConfig({ dualPipeline: state.gpuDualPipe });
  });

  // RAM controls
  els.ramToggleBtn.addEventListener('click', () => {
    if (ramEngine.running) stopRam();
    else startRam();
  });

  els.ramSizePresetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      setRamTargetMB(Number(btn.dataset.mb));
    });
  });

  els.ramSizeSlider.addEventListener('input', () => {
    // Update readout live during drag; commit reallocation on 'change' so we don't churn GBs per pixel
    const mb = Number(els.ramSizeSlider.value);
    state.ramTargetMB = mb;
    updateSliderFill(els.ramSizeSlider);
    els.ramSizePresetBtns.forEach((btn) => {
      btn.classList.toggle('is-active', Number(btn.dataset.mb) === mb);
    });
    syncUIState();
  });

  els.ramSizeSlider.addEventListener('change', () => {
    ramEngine.updateConfig({ targetMB: state.ramTargetMB });
  });

  els.ramModeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      els.ramModeBtns.forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      state.ramMode = btn.dataset.ramMode;
      ramEngine.updateConfig({ mode: state.ramMode });
      syncUIState();
    });
  });

  els.ramThreadsSlider.addEventListener('input', () => {
    state.ramThreads = Number(els.ramThreadsSlider.value);
    updateSliderFill(els.ramThreadsSlider);
    syncUIState();
  });

  els.ramThreadsSlider.addEventListener('change', () => {
    ramEngine.updateConfig({ threads: state.ramThreads });
  });

  // Summary & error banners
  els.summaryDismissBtn.addEventListener('click', () => {
    els.summaryStrip.classList.add('hidden');
  });
  els.errorDismissBtn.addEventListener('click', hideError);

  window.addEventListener('beforeunload', () => {
    cpuEngine.destroy();
    gpuEngine.destroy();
    ramEngine.destroy();
    releaseWakeLock();
  });
}

// ==================== Bootstrap ====================
async function init() {
  const maxCpuSlider = Math.max(64, Math.min(256, detectedCores * 2));
  els.cpuThreadsSlider.max = String(maxCpuSlider);
  els.cpuThreadsSlider.value = String(detectedCores);
  els.cpuThreadsInput.value = String(detectedCores);

  els.ramThreadsSlider.value = String(state.ramThreads);
  els.ramSizeSlider.value = String(state.ramTargetMB);

  updateSliderFill(els.cpuThreadsSlider);
  updateSliderFill(els.gpuScaleSlider);
  updateSliderFill(els.ramSizeSlider);
  updateSliderFill(els.ramThreadsSlider);

  wireEvents();

  const caps = await gpuEngine.init();
  state.gpuEngine = caps.preferredEngine === 'webgpu' ? 'webgpu' : 'webgl2';

  if (!caps.webgpu) {
    els.btnEngineWebgpu.disabled = true;
    els.btnEngineWebgpu.classList.remove('is-active');
    els.btnEngineWebgl2.classList.add('is-active');
  } else {
    els.btnEngineWebgpu.disabled = false;
    els.btnEngineWebgpu.classList.add('is-active');
    els.btnEngineWebgl2.classList.remove('is-active');
  }

  if (!caps.webgl2) {
    els.btnEngineWebgl2.disabled = true;
  }

  if (!caps.webgpu && !caps.webgl2) {
    showError(t('gpuUnsupported'));
  }

  applyLanguage();
  cpuSparkline.draw();
  ramSparkline.draw();
}

init();

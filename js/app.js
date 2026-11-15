import { I18N } from './i18n.js';
import { CpuStressEngine } from './cpu-stress.js';
import { GpuStressEngine } from './gpu-stress.js';
import { Sparkline } from './sparkline.js';

const detectedCores = Math.max(1, navigator.hardwareConcurrency || 8);

const state = {
  lang: 'zh',
  // Duration: 0 = infinite continuous, >0 = timed seconds
  durationPreset: '0', // '60' | '300' | '900' | '1800' | '3600' | '0' | 'custom'
  customMinutes: 10,
  customSeconds: 0,

  // Session runtime
  sessionActive: false,
  sessionStartMs: 0,
  sessionTargetSec: 0, // 0 means continuous
  ranCpuInSession: false,
  ranGpuInSession: false,
  timerInterval: null,
  wakeLock: null,

  // CPU config
  cpuThreads: detectedCores,
  cpuWorkload: 'extreme',
  cpuDuty: 1.0,

  // GPU config
  gpuEngine: 'webgl2',
  gpuPreset: 'auto',
  gpuScale: 4,
  gpuUncap: true,
  gpuShowPreview: true,
  gpuHasError: false
};

// DOM References
const $ = (id) => document.getElementById(id);

const els = {
  hwCpuText: $('hwCpuText'),
  hwGpuText: $('hwGpuText'),
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

  summaryStrip: $('summaryStrip'),
  summaryDurationVal: $('summaryDurationVal'),
  summaryCpuItem: $('summaryCpuItem'),
  summaryCpuVal: $('summaryCpuVal'),
  summaryGpuItem: $('summaryGpuItem'),
  summaryGpuVal: $('summaryGpuVal'),
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
  if (hrs > 0) {
    return `${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
  }
  return `${pad(mins)}:${pad(secs)}`;
}

function formatGflops(gflops) {
  if (gflops >= 1000) {
    return { val: (gflops / 1000).toFixed(2), unit: t('unitTflops') };
  }
  return { val: gflops.toFixed(1), unit: t('unitGflops') };
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

// ==================== Engines & Sparkline ====================
const cpuSparkline = new Sparkline(els.cpuSparkline, {
  maxPoints: 48,
  strokeColor: '#2F6FED',
  fillTop: 'rgba(47, 111, 237, 0.2)',
  fillBottom: 'rgba(47, 111, 237, 0.0)'
});

const cpuEngine = new CpuStressEngine((stats) => {
  const unit = stats.workload === 'crypto' ? t('unitMh') : t('unitMops');
  els.cpuRateVal.textContent = stats.rateMillions.toFixed(1);
  els.cpuRateUnit.textContent = unit;
  els.cpuPeakVal.textContent = `${stats.peakMillions.toFixed(1)} ${unit}`;
  els.cpuActiveVal.textContent = String(stats.activeWorkers);
  els.cpuTargetThreadsUnit.textContent = `/ ${state.cpuThreads}`;
  els.cpuSatVal.textContent = Math.round(stats.saturationPct).toString();
  cpuSparkline.push(stats.rateMillions);
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
  } catch (_) {
    // Ignore silently when unsupported or blocked
  }
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
  if (target > 0) {
    els.timerDisplay.textContent = formatDuration(target);
  } else {
    els.timerDisplay.textContent = '∞';
  }
  els.timerProgressFill.classList.remove('is-continuous');
  els.timerProgressFill.style.width = '0%';
  els.timerStateTag.textContent = t('timerReady');
  els.timerStateTag.className = 'timer-state-tag';
}

function checkSessionAutoEnd(completedByTimer = false) {
  if (!state.sessionActive) return;
  if (cpuEngine.running || gpuEngine.running) return;

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

  if (elapsedSec >= 1.0 && (state.ranCpuInSession || state.ranGpuInSession)) {
    renderSummary(elapsedSec);
  }
}

function renderSummary(elapsedSec) {
  els.summaryDurationVal.textContent = formatDuration(elapsedSec);

  if (state.ranCpuInSession) {
    const cpuSum = cpuEngine.getSummary();
    const unit = cpuSum.workload === 'crypto' ? t('unitMh') : t('unitMops');
    els.summaryCpuVal.textContent = `${cpuSum.avgMillions.toFixed(1)} / ${cpuSum.peakMillions.toFixed(1)} ${unit}`;
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
  if (state.sessionActive) {
    state.ranCpuInSession = true;
  } else {
    beginSessionIfNeeded();
  }
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
    showPreview: state.gpuShowPreview
  });
  if (state.sessionActive) {
    state.ranGpuInSession = true;
  } else {
    beginSessionIfNeeded();
  }
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

function startAll() {
  if (!cpuEngine.running) startCpu();
  if (!gpuEngine.running && (gpuEngine.capabilities.webgpu || gpuEngine.capabilities.webgl2)) {
    startGpu();
  }
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
  const anyOn = cpuOn || gpuOn;

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
  els.cpuRateUnit.textContent = state.cpuWorkload === 'crypto' ? t('unitMh') : t('unitMops');
  els.cpuDutySub.textContent = `${Math.round(state.cpuDuty * 100)}%`;

  const kernelKey =
    state.cpuWorkload === 'fp64'
      ? 'workloadFp64'
      : state.cpuWorkload === 'crypto'
        ? 'workloadCrypto'
        : 'workloadExtreme';
  els.cpuKernelBadge.textContent = t(kernelKey);

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

  syncGpuCanvasVisibility();
}

function applyLanguage() {
  document.documentElement.lang = state.lang === 'zh' ? 'zh-CN' : 'en';
  els.langToggleBtn.textContent = state.lang === 'zh' ? 'EN' : '中';

  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    if (key) {
      el.textContent = t(key);
    }
  });

  // Dynamic interpolated texts
  els.hwCpuText.textContent = `${detectedCores} ${t('logicalCores')}`;
  els.cpuThreadsHint.textContent = t('cpuThreadsHint').replace('{cores}', String(detectedCores));

  const caps = gpuEngine.capabilities;
  const backendTag = caps.webgpu ? 'WebGPU' : caps.webgl2 ? 'WebGL2' : 'N/A';
  els.hwGpuText.textContent = caps.rendererName
    ? `${backendTag} · ${caps.rendererName}`
    : `${backendTag} · ${t('unknownGpu')}`;

  if (!caps.webgpu && caps.webgl2) {
    els.gpuEngineHint.textContent = t('gpuWebgpuUnavailable');
    els.gpuEngineHint.classList.remove('hidden');
  } else {
    els.gpuEngineHint.classList.add('hidden');
  }

  if (!state.sessionActive) {
    updateIdleTimerPreview();
  } else {
    tickSessionTimer();
  }
  syncUIState();
}

// ==================== Event Wiring ====================
function setCpuThreads(value) {
  const clamped = Math.max(1, Math.min(256, Math.round(value) || 1));
  state.cpuThreads = clamped;
  const sliderMax = Number(els.cpuThreadsSlider.max) || 64;
  if (clamped > sliderMax) {
    els.cpuThreadsSlider.max = String(clamped);
  }
  els.cpuThreadsSlider.value = String(clamped);
  els.cpuThreadsInput.value = String(clamped);
  updateSliderFill(els.cpuThreadsSlider);
  cpuEngine.updateConfig({ threads: clamped });
  syncUIState();
}

function wireEvents() {
  // Language toggle
  els.langToggleBtn.addEventListener('click', () => {
    state.lang = state.lang === 'zh' ? 'en' : 'zh';
    applyLanguage();
  });

  // ESC emergency stop (button + global keyboard)
  els.escStopBtn.addEventListener('click', () => stopAll(false));
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      stopAll(false);
    }
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
    if (cpuEngine.running || gpuEngine.running) {
      stopAll(false);
    } else {
      startAll();
    }
  });

  // CPU toggle & controls
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
      if (chip.dataset.abs) {
        setCpuThreads(Number(chip.dataset.abs));
      } else if (chip.dataset.mult) {
        setCpuThreads(Math.round(detectedCores * Number(chip.dataset.mult)));
      }
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

  // GPU toggle & controls
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

  // Summary & error banners
  els.summaryDismissBtn.addEventListener('click', () => {
    els.summaryStrip.classList.add('hidden');
  });
  els.errorDismissBtn.addEventListener('click', hideError);

  // Clean shutdown on page unload
  window.addEventListener('beforeunload', () => {
    cpuEngine.destroy();
    gpuEngine.destroy();
    releaseWakeLock();
  });
}

// ==================== Bootstrap ====================
async function init() {
  // Configure CPU slider range smartly based on hardwareConcurrency
  const maxSlider = Math.max(64, detectedCores * 2);
  els.cpuThreadsSlider.max = String(Math.min(256, maxSlider));
  els.cpuThreadsSlider.value = String(detectedCores);
  els.cpuThreadsInput.value = String(detectedCores);
  updateSliderFill(els.cpuThreadsSlider);
  updateSliderFill(els.gpuScaleSlider);

  wireEvents();

  // Probe GPU capabilities
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
}

init();

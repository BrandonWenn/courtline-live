import * as ort from './ort.webgpu.min.mjs';
import {
  letterboxTransform, decodeBody, decodeRackets, selectPerson, selectRacket,
  squareCrop, decodeSimCC, validRacketAxis,
} from './vision-math.js';

// The worker has no upload endpoint: camera frames and model execution stay in this browser.
const SIZES = { body: 416, racket: 640, pose: 256 };
const MODEL_FILES = { body: 'body.onnx', racket: 'racket.onnx', pose: 'racket-pose.onnx' };
const MEAN = [123.675, 116.28, 103.53];
const STD = [58.395, 57.12, 57.375];
const imageCanvas = new OffscreenCanvas(1, 1);
const imageContext = imageCanvas.getContext('2d', { willReadFrequently: true, alpha: false });
const bodyCanvas = new OffscreenCanvas(SIZES.body, SIZES.body);
const racketCanvas = new OffscreenCanvas(SIZES.racket, SIZES.racket);
const poseCanvas = new OffscreenCanvas(SIZES.pose, SIZES.pose);
const bodyContext = bodyCanvas.getContext('2d', { willReadFrequently: true, alpha: false });
const racketContext = racketCanvas.getContext('2d', { willReadFrequently: true, alpha: false });
const poseContext = poseCanvas.getContext('2d', { willReadFrequently: true, alpha: false });
let sessions = null;
let modelBytes = {};
let backend = 'wasm';
let baseUrl = new URL('./', import.meta.url).href;
let initialized = false;
let busy = false;
let initializing = false;
let previousPersonBox = null;
let missedPeople = 0;
let fallbackUsed = false;
let sessionBackends = {};
let sessionPreferences = {};
const tensorBuffers = new Map();

function report(message, extra = {}) {
  self.postMessage({ type: 'progress', message, ...extra });
}

function backendLabel() {
  if (backend !== 'webgpu') return 'WASM / CPU';
  const cpuStages = Object.entries(sessionBackends).filter(([, value]) => value === 'wasm').map(([name]) => ({body:'人体',racket:'球拍检测',pose:'球拍关键点'}[name]));
  return cpuStages.length ? `混合加速（${cpuStages.join('、')}用CPU）` : 'WebGPU（含CPU算子回退）';
}

async function fetchModel(name, number) {
  const paths = name === 'pose' ? ["racket-pose.part01.bin", "racket-pose.part02.bin", "racket-pose.part03.bin"] : [MODEL_FILES[name]];
  const chunks = []; let size = 0;
  for (let i=0; i<paths.length; i++) {
    report(`正在下载模型 ${number}/3 · 文件 ${i+1}/${paths.length}`);
    const response = await fetch(new URL(paths[i], baseUrl));
    if (!response.ok) throw new Error(`模型下载失败：${paths[i]} HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer()); chunks.push(bytes); size += bytes.length;
  }
  const merged = new Uint8Array(size); let offset = 0;
  for (const bytes of chunks) { merged.set(bytes,offset); offset += bytes.length; }
  report(`模型 ${number}/3 下载完成 · ${(size/1e6).toFixed(1)} MB`);
  return merged;
}

async function releaseSessions() {
  const old = sessions;
  sessions = null;
  if (old) await Promise.all(Object.values(old).map(session => session.release().catch(() => {})));
}

async function createSessions(nextBackend) {
  await releaseSessions();
  const created = {};
  try {
    const assignments = {};
    for (const [name, bytes] of Object.entries(modelBytes)) {
      const stageBackend = nextBackend === 'webgpu' && sessionPreferences[name] !== 'wasm' ? 'webgpu' : 'wasm';
      assignments[name] = stageBackend;
      report(`正在准备${{ body: '人体', racket: '球拍检测', pose: '球拍关键点' }[name]}模型…`);
      created[name] = await ort.InferenceSession.create(bytes, {
        executionProviders: stageBackend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
        graphOptimizationLevel: 'all',
        // Keep readbacks deterministic; no native GPU tensors are passed to the page.
        enableCpuMemArena: true,
      });
    }
    sessions = created;
    sessionBackends = assignments;
    backend = Object.values(assignments).includes('webgpu') ? 'webgpu' : 'wasm';
  } catch (error) {
    await Promise.all(Object.values(created).map(session => session.release().catch(() => {})));
    throw error;
  }
}

async function initialize(message) {
  initializing = true;
  initialized = false;
  previousPersonBox = null;
  missedPeople = 0;
  fallbackUsed = false;
  try {
    baseUrl = new URL(message.baseUrl || './', import.meta.url).href;
    sessionPreferences = message.sessionBackends || {};
    ort.env.wasm.wasmPaths = {mjs: new URL('ort-wasm-simd-threaded.asyncify.mjs',baseUrl).href};
    if (!ort.env.wasm.wasmBinary) {
      report('正在准备浏览器推理运行文件…');
      if (typeof DecompressionStream === 'undefined') throw new Error('请升级到最新版Chrome或Edge后重试');
      const runtime = await fetch(new URL('runtime-gzip.bin',baseUrl));
      if (!runtime.ok) throw new Error('运行文件下载失败 HTTP '+runtime.status);
      const binary = new Uint8Array(await new Response(runtime.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
      if (binary[0]!==0 || binary[1]!==97 || binary[2]!==115 || binary[3]!==109) throw new Error('运行文件不完整，请刷新后重试');
      ort.env.wasm.wasmBinary = binary;
    }
    // Multi-threaded WASM requires cross-origin isolation. Static GitHub Pages doesn't provide it.
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.max(1, Math.min(4, self.navigator?.hardwareConcurrency || 1)) : 1;
    ort.env.wasm.proxy = false;
    ort.env.logLevel = 'error';
    for (const [index, name] of Object.keys(MODEL_FILES).entries()) {
      if (!modelBytes[name]) modelBytes[name] = await fetchModel(name, index + 1);
    }
    let useGpu = false;
    if (message.backend !== 'wasm' && self.navigator?.gpu) {
      try { useGpu = !!(await self.navigator.gpu.requestAdapter()); } catch { /* use WASM */ }
    }
    if (useGpu) {
      try {
        await createSessions('webgpu');
      } catch (error) {
        console.warn('WebGPU initialization failed; using WASM', error);
        fallbackUsed = true;
        report('当前显卡不支持全部模型，正在切换到CPU…');
        await createSessions('wasm');
      }
    } else {
      report('使用浏览器CPU模式，首次准备模型可能需要一些时间…');
      await createSessions('wasm');
    }
    initialized = true;
    self.postMessage({ type: 'ready', backend, backendLabel: backendLabel(), sizes: SIZES, sessionBackends });
  } finally {
    initializing = false;
  }
}

function pixelsToTensor(context, size, normalizePose = false) {
  const rgba = context.getImageData(0, 0, size, size).data;
  const plane = size * size;
  if (!tensorBuffers.has(size)) tensorBuffers.set(size, new Float32Array(plane * 3));
  const data = tensorBuffers.get(size);
  for (let i = 0, pixel = 0; i < plane; i++, pixel += 4) {
    if (normalizePose) {
      data[i] = (rgba[pixel] - MEAN[0]) / STD[0];
      data[plane + i] = (rgba[pixel + 1] - MEAN[1]) / STD[1];
      data[plane * 2 + i] = (rgba[pixel + 2] - MEAN[2]) / STD[2];
    } else {
      data[i] = rgba[pixel] / 255;
      data[plane + i] = rgba[pixel + 1] / 255;
      data[plane * 2 + i] = rgba[pixel + 2] / 255;
    }
  }
  return new ort.Tensor('float32', data, [1, 3, size, size]);
}

function prepareYolo(context, size) {
  const transform = letterboxTransform(imageCanvas.width, imageCanvas.height, size);
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.fillStyle = 'rgb(114,114,114)';
  context.fillRect(0, 0, size, size);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.drawImage(imageCanvas, 0, 0, imageCanvas.width, imageCanvas.height,
    transform.padLeft, transform.padTop, transform.resizedWidth, transform.resizedHeight);
  return { tensor: pixelsToTensor(context, size), transform };
}

function prepareRacketPose(box) {
  const crop = squareCrop(box);
  const scale = SIZES.pose / crop.side;
  poseContext.setTransform(1, 0, 0, 1, 0, 0);
  poseContext.fillStyle = '#000';
  poseContext.fillRect(0, 0, SIZES.pose, SIZES.pose);
  poseContext.imageSmoothingEnabled = true;
  poseContext.imageSmoothingQuality = 'high';
  poseContext.setTransform(scale, 0, 0, scale, SIZES.pose / 2 - crop.centerX * scale, SIZES.pose / 2 - crop.centerY * scale);
  poseContext.drawImage(imageCanvas, 0, 0);
  poseContext.setTransform(1, 0, 0, 1, 0, 0);
  return { tensor: pixelsToTensor(poseContext, SIZES.pose, true), crop };
}

function disposeOutputs(outputs) {
  if (!outputs) return;
  for (const tensor of Object.values(outputs)) {
    // Host tensors are garbage collected; dispose releases device resources when present.
    try { tensor.dispose?.(); } catch { /* host tensor */ }
  }
}

async function runYolo(session, prepared, decode, threshold) {
  let outputs;
  try {
    const start = performance.now();
    outputs = await session.run({ [session.inputNames[0]]: prepared.tensor });
    const runMs = performance.now() - start;
    const decodeStart = performance.now();
    const items = decode(outputs[session.outputNames[0]], prepared.transform, threshold);
    return { items, runMs, decodeMs: performance.now() - decodeStart };
  } finally {
    prepared.tensor.dispose?.();
    disposeOutputs(outputs);
  }
}

async function inferSource() {
  const width = imageCanvas.width, height = imageCanvas.height;
  const profile = {};
  const bodyStart = performance.now();
  const preparedBody = prepareYolo(bodyContext, SIZES.body);
  profile.bodyPreprocessMs = performance.now() - bodyStart;
  const bodyResult = await runYolo(sessions.body, preparedBody, decodeBody, 0.4);
  profile.bodyMs = performance.now() - bodyStart;
  profile.bodyRunMs = bodyResult.runMs;
  profile.bodyDecodeMs = bodyResult.decodeMs;
  const person = selectPerson(bodyResult.items, previousPersonBox);
  if (person) {
    previousPersonBox = person.box;
    missedPeople = 0;
  } else if (++missedPeople >= 3) previousPersonBox = null;
  const racketStart = performance.now();
  const preparedRacket = prepareYolo(racketContext, SIZES.racket);
  profile.racketPreprocessMs = performance.now() - racketStart;
  const racketResult = await runYolo(sessions.racket, preparedRacket, decodeRackets, 0.25);
  profile.racketMs = performance.now() - racketStart;
  profile.racketRunMs = racketResult.runMs;
  profile.racketDecodeMs = racketResult.decodeMs;
  const selected = selectRacket(racketResult.items, person, width, height);
  let racketPoints = null, racketScores = null, axis = null;
  profile.poseMs = 0; profile.poseRunMs = 0; profile.posePreprocessMs = 0; profile.poseDecodeMs = 0;
  if (selected) {
    const poseStart = performance.now();
    const prepared = prepareRacketPose(selected.box);
    profile.posePreprocessMs = performance.now() - poseStart;
    let outputs;
    try {
      const runStart = performance.now();
      outputs = await sessions.pose.run({ [sessions.pose.inputNames[0]]: prepared.tensor });
      profile.poseRunMs = performance.now() - runStart;
      const decodeStart = performance.now();
      const x = outputs.simcc_x || outputs[sessions.pose.outputNames[0]];
      const y = outputs.simcc_y || outputs[sessions.pose.outputNames[1]];
      const decoded = decodeSimCC(x, y, prepared.crop);
      racketPoints = decoded.points;
      racketScores = decoded.scores;
      axis = validRacketAxis(racketPoints, racketScores, selected.box, width, height);
      profile.poseDecodeMs = performance.now() - decodeStart;
    } finally {
      prepared.tensor.dispose?.();
      disposeOutputs(outputs);
    }
    profile.poseMs = performance.now() - poseStart;
  }
  return {
    person: person?.keypoints || null,
    personBox: person?.box || null,
    bbox: selected?.box || null,
    racketConfidence: selected?.score || null,
    racketPoints, racketScores, axis, profile,
  };
}

async function processFrame(message) {
  if (!initialized || initializing || busy) {
    message.bitmap?.close?.();
    self.postMessage({ type: 'error', frameId: message.frameId, fatal: false, message: '模型尚未就绪或正在处理上一帧' });
    return;
  }
  busy = true;
  const started = performance.now();
  try {
    const bitmap = message.bitmap;
    if (!bitmap?.width || !bitmap?.height) throw new Error('无效的视频帧');
    if (imageCanvas.width !== bitmap.width || imageCanvas.height !== bitmap.height) {
      imageCanvas.width = bitmap.width; imageCanvas.height = bitmap.height;
    }
    imageContext.drawImage(bitmap, 0, 0);
    bitmap.close();
    const sourceDrawMs = performance.now() - started;
    let result;
    try {
      result = await inferSource();
    } catch (error) {
      if (backend !== 'webgpu' || fallbackUsed) throw error;
      fallbackUsed = true;
      console.warn('WebGPU execution failed; retrying frame on WASM', error);
      report('显卡执行遇到兼容问题，切换到CPU模式…');
      await createSessions('wasm');
      self.postMessage({ type: 'backend', backend, backendLabel: backendLabel(), sessionBackends });
      result = await inferSource();
    }
    self.postMessage({ type: 'result', frameId: message.frameId, timestamp: message.timestamp,
      width: imageCanvas.width, height: imageCanvas.height, ...result,
      profile: { ...result.profile, sourceDrawMs, totalMs: performance.now() - started },
      processingMs: performance.now() - started, backend, backendLabel: backendLabel(), sessionBackends });
  } catch (error) {
    message.bitmap?.close?.();
    self.postMessage({ type: 'error', frameId: message.frameId, fatal: true,
      message: `识别失败：${error.message || String(error)}` });
  } finally {
    busy = false;
  }
}

self.onmessage = event => {
  const message = event.data || {};
  if (message.type === 'init') {
    if (initializing || busy) return;
    initialize(message).catch(error => self.postMessage({ type: 'error', fatal: true,
      message: `模型初始化失败：${error.message || String(error)}` }));
  } else if (message.type === 'frame') {
    processFrame(message);
  } else if (message.type === 'reset') {
    previousPersonBox = null;
    missedPeople = 0;
  }
};

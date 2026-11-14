const $ = (id) => document.getElementById(id);
const video = $('input-video'), canvas = $('output'), ctx = canvas.getContext('2d');
const capture = $('capture'), captureCtx = capture.getContext('2d', { alpha: false });
const bones = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
let worker = null, stream = null, fileUrl = null, epoch = 0, inFlight = false, running = false;
let sourceType = null, frames = 0, lastSentTime = -1, previousResultTime = 0, measurements = [];
let pendingInit = null, frameCounter = 0, activeFrame = 0, sending = false;
const stats = { results: 0, body: 0, axis: 0, errors: 0, profile: null };
// Exposes only aggregate diagnostics, never video pixels, for local compatibility checks.
window.courtlineDiagnostics = () => ({ ...stats, running, sourceType, backend: $('backend-tag').textContent, fps: $('fps').textContent, notice: $('notice').textContent });

function notice(message, error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
}
function setControls(busy) {
  $('start-camera').disabled = busy;
  $('camera-select').disabled = busy;
  $('backend-select').disabled = busy;
  $('stop').disabled = !busy;
  $('video-file').disabled = busy;
  document.body.classList.toggle('running', busy);
}
function loading(show, message) {
  $('loading-overlay').hidden = !show;
  if (message) $('loading-message').textContent = message;
}
function stop({ message = '已停止识别，摄像头已释放。', keepFrame = true } = {}) {
  epoch++;
  running = false; sending = false; inFlight = false;
  if (pendingInit) { pendingInit.reject(new Error('识别已取消')); pendingInit = null; }
  if (worker) { worker.terminate(); worker = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.pause(); video.srcObject = null;
  if (fileUrl) { video.removeAttribute('src'); video.load(); URL.revokeObjectURL(fileUrl); fileUrl = null; }
  loading(false); setControls(false);
  $('live-pill').classList.remove('running'); $('source-status').textContent = frames ? '已停止' : '尚未开始';
  if (!keepFrame) { ctx.clearRect(0,0,canvas.width,canvas.height); $('empty-state').hidden = false; }
  notice(message);
}
function friendlyError(error) {
  if (error?.name === 'NotAllowedError') return '摄像头访问未获允许。请在地址栏的相机设置中允许访问后重试。';
  if (error?.name === 'NotFoundError') return '没有找到摄像头。请连接摄像头后刷新列表。';
  if (error?.name === 'NotReadableError') return '摄像头无法读取，可能正在被其他程序占用。请关闭其他相机程序。';
  return error?.message || '启动失败，请尝试 CPU 兼容模式，或使用最新版 Chrome / Edge。';
}
async function refreshCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const items = (await navigator.mediaDevices.enumerateDevices()).filter(x => x.kind === 'videoinput');
    const previous = $('camera-select').value;
    $('camera-select').replaceChildren(new Option('默认摄像头', ''));
    items.forEach((x, i) => $('camera-select').add(new Option(x.label || `摄像头 ${i+1}`, x.deviceId)));
    if ([...$('camera-select').options].some(x => x.value === previous)) $('camera-select').value = previous;
  } catch { /* Permission may hide labels until the first successful stream. */ }
}
async function initModels(token) {
  worker = new Worker(new URL('./inference-worker.js', import.meta.url), { type: 'module' });
  return new Promise((resolve, reject) => {
    pendingInit = { resolve, reject };
    worker.onmessage = (event) => {
      if (token !== epoch) return;
      const data = event.data;
      if (data.type === 'progress') { $('loading-message').textContent = data.message; return; }
      if (data.type === 'ready') {
        $('backend-tag').textContent = data.backendLabel || data.backend || '本地推理';
        pendingInit = null; resolve(data); return;
      }
      if (data.type === 'backend') {
        $('backend-tag').textContent = data.backendLabel || data.backend || 'CPU / WASM';
        notice(data.message || '已切换到兼容运行方式。'); return;
      }
      if (data.type === 'result') { receiveResult(data, token); return; }
      if (data.type === 'error') {
        stats.errors++;
        if (pendingInit) { pendingInit = null; reject(new Error(data.message)); }
        else { stop(); notice(data.message || '模型运行失败，请切换到 CPU 兼容模式。', true); }
      }
    };
    worker.onerror = (event) => {
      stats.errors++;
      const message = event.message || '识别线程启动失败。请检查模型文件是否完整。';
      if (pendingInit) { pendingInit = null; reject(new Error(message)); }
      else { stop(); notice(message, true); }
    };
    worker.postMessage({ type: 'init', backend: $('backend-select').value, baseUrl: new URL('./', location.href).href });
  });
}
function waitForVideo() {
  if (video.readyState >= 2 && video.videoWidth) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { cleanup(); reject(new Error('视频未能加载。请尝试 H.264 编码的 MP4，或重新连接摄像头。')); }, 15000);
    const cleanup = () => { clearTimeout(timer); video.removeEventListener('loadeddata', good); video.removeEventListener('error', bad); };
    const good = () => { cleanup(); resolve(); };
    const bad = () => { cleanup(); reject(new Error('浏览器不支持这段视频的编码。建议转换为 H.264 MP4。')); };
    video.addEventListener('loadeddata', good); video.addEventListener('error', bad);
  });
}
async function start(kind, file) {
  stop({message:'正在准备…', keepFrame:false});
  const token = epoch;
  frames = 0; previousResultTime = 0; measurements = []; lastSentTime = -1;
  Object.assign(stats, { results:0, body:0, axis:0, errors:0 });
  sourceType = kind; setControls(true);
  $('body-status').textContent = '待识别'; $('racket-status').textContent = '待识别';
  $('body-status').classList.remove('observed'); $('racket-status').classList.remove('observed');
  $('fps').innerHTML = '—<small> FPS</small>'; $('latency').innerHTML = '—<small> ms</small>';
  try {
    if (!window.isSecureContext) throw new Error('摄像头需要 HTTPS 或 localhost。请使用正式部署网址，不要直接双击 HTML 文件。');
    if (kind === 'camera') {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('此浏览器不支持摄像头，请使用最新版 Chrome 或 Edge。');
      notice('请在浏览器提示中允许摄像头访问。');
      const id = $('camera-select').value;
      const nextStream = await navigator.mediaDevices.getUserMedia({ audio:false, video:{ width:{ideal:1280}, height:{ideal:720}, frameRate:{ideal:30}, ...(id ? {deviceId:{exact:id}} : {}) } });
      if (token !== epoch) { nextStream.getTracks().forEach(t=>t.stop()); return; }
      stream = nextStream; video.srcObject = stream;
      stream.getVideoTracks().forEach(t => t.addEventListener('ended', () => { if(token===epoch) stop({message:'摄像头连接已结束。'}); }));
      await refreshCameras();
    } else {
      if (!file) return;
      fileUrl = URL.createObjectURL(file); video.src = fileUrl;
    }
    await video.play();
    await waitForVideo();
    if (token !== epoch) return;
    // Pause files during initialization so their first frames are not silently skipped.
    if (kind === 'file') video.pause();
    loading(true, '首次下载约 110 MB 模型与运行文件，后续由浏览器缓存。');
    notice('正在加载人体与球拍模型，视频画面不会上传。');
    await initModels(token);
    if (token !== epoch) return;
    const ratio = Math.min(1, 1280/Math.max(video.videoWidth, video.videoHeight));
    capture.width = canvas.width = Math.round(video.videoWidth*ratio);
    capture.height = canvas.height = Math.round(video.videoHeight*ratio);
    if (kind === 'file') { video.currentTime=0; await video.play(); }
    loading(false); $('empty-state').hidden = true;
    $('source-status').textContent = kind==='camera' ? '实时识别中' : '本地视频识别中';
    $('live-pill').classList.add('running');
    running = true;
    notice('识别已启动。遮挡或低置信度时会隐藏球拍线；帧率为分析帧率，不是摄像头采集帧率。');
    schedule(token);
  } catch (error) {
    if (token !== epoch) return;
    stop({keepFrame:false}); notice(friendlyError(error), true);
  }
}
function schedule(token) {
  if (!running || token !== epoch) return;
  if (video.ended) { stop({message:'视频播放完成。可重新选择视频，或打开摄像头。'}); return; }
  const callback = () => sendFrame(token);
  // Worker completion drives the next frame. Do not depend on compositor callbacks:
  // Chromium may throttle them for a hidden source video or occluded window.
  queueMicrotask(callback);
}
async function sendFrame(token) {
  if (!running || token!==epoch || inFlight || sending) return;
  if (video.ended) { stop({message:'视频播放完成。'}); return; }
  if (video.readyState<2 || video.currentTime===lastSentTime) { setTimeout(()=>schedule(token),20); return; }
  sending = true;
  try {
    captureCtx.drawImage(video,0,0,capture.width,capture.height);
    const bitmap = await createImageBitmap(capture);
    if (!running || token!==epoch) { bitmap.close(); return; }
    lastSentTime = video.currentTime; inFlight = true;
    activeFrame = ++frameCounter;
    worker.postMessage({type:'frame',frameId:activeFrame,bitmap,timestamp:performance.now()},[bitmap]);
  } catch (error) {
    if (token===epoch) { stop(); notice(friendlyError(error),true); }
  } finally { sending=false; }
}
function receiveResult(data, token) {
  if (!running || token!==epoch || data.frameId!==activeFrame) return;
  inFlight = false; frames++; stats.results++; stats.profile = data.profile || null;
  stats.body += !!data.person; stats.axis += !!data.axis;
  const now = performance.now();
  if (previousResultTime) { measurements.push(now-previousResultTime); if(measurements.length>30)measurements.shift(); }
  previousResultTime=now;
  const fps=measurements.length ? 1000/(measurements.reduce((a,b)=>a+b,0)/measurements.length) : null;
  $('fps').innerHTML=(fps ? fps.toFixed(1) : '—')+'<small> FPS</small>';
  $('latency').innerHTML=Math.round(data.processingMs)+'<small> ms</small>';
  $('body-status').textContent=data.person ? '已识别' : '未检出';
  $('racket-status').textContent=data.axis ? '已识别' : '暂未定位';
  $('body-status').classList.toggle('observed',!!data.person);
  $('racket-status').classList.toggle('observed',!!data.axis);
  paint(data);
  schedule(token);
}
function point(p) { return p && p.length>=2 && Number.isFinite(p[0]) && Number.isFinite(p[1]); }
function line(a,b,color,width) {
  if(!point(a)||!point(b))return;
  ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
}
function paint(data) {
  ctx.save(); ctx.clearRect(0,0,canvas.width,canvas.height);
  if($('mirror').checked) {ctx.translate(canvas.width,0);ctx.scale(-1,1);}
  ctx.drawImage(capture,0,0);
  const thickness=Math.max(1.5,canvas.width/550);
  if(data.person) {
    const p=data.person;
    for(const [a,b] of bones)if(p[a]?.[2]>.5&&p[b]?.[2]>.5)line(p[a],p[b],'#77e6d0',thickness);
    for(const q of p.slice(5))if(point(q)&&q[2]>.5) {ctx.beginPath();ctx.fillStyle='#f1fcf5';ctx.arc(q[0],q[1],thickness*1.6,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#48a58c';ctx.lineWidth=1;ctx.stroke();}
  }
  if(data.axis) {
    line(data.axis[0],data.axis[1],'#ffb45c',thickness*1.6);
    for(const q of data.axis)if(point(q)){ctx.beginPath();ctx.fillStyle='#ffd496';ctx.arc(q[0],q[1],thickness*1.8,0,Math.PI*2);ctx.fill();}
  }
  ctx.restore();
}
$('start-camera').addEventListener('click',()=>start('camera'));
$('stop').addEventListener('click',()=>stop());
$('cancel-load').addEventListener('click',()=>stop());
$('video-file').addEventListener('change',e=>{const file=e.target.files?.[0];if(file)start('file',file);e.target.value='';});
$('refresh-cameras').addEventListener('click',refreshCameras);
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('viewer').requestFullscreen();}catch{notice('当前浏览器暂不支持全屏。');}});
video.addEventListener('ended',()=>{if(running&&sourceType==='file')stop({message:'视频播放完成。可重新选择视频继续。'});});
window.addEventListener('beforeunload',()=>{if(stream)stream.getTracks().forEach(t=>t.stop());worker?.terminate();});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&running&&!document.fullscreenElement)stop();});
if(location.protocol==='file:')notice('请通过 HTTPS 网站或 localhost 打开，直接双击文件无法运行识别。',true);
else if(!window.Worker||!window.OffscreenCanvas)notice('此浏览器缺少本地推理所需能力，请使用最新版 Chrome 或 Edge。',true);
refreshCameras();

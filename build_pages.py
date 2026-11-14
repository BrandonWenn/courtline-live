"""Build a flat GitHub-Pages directory; split the large pose model for browser uploads."""
from pathlib import Path
import shutil,re,json,hashlib,gzip
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'dist';dst=ROOT/'pages'
if dst.exists():shutil.rmtree(dst)
dst.mkdir()
(dst/'.nojekyll').write_text('')
for f in src.iterdir():
 if f.is_file():shutil.copy2(f,dst/f.name)
for n in ('ort.webgpu.min.mjs','ort-wasm-simd-threaded.asyncify.mjs'):
 shutil.copy2(src/'vendor'/n,dst/n)
(dst/'runtime-gzip.bin').write_bytes(gzip.compress((src/'vendor/ort-wasm-simd-threaded.asyncify.wasm').read_bytes(),compresslevel=9,mtime=0))
for f in (src/'licenses').iterdir():shutil.copy2(f,dst/f.name)
for n in ('body.onnx','racket.onnx'):shutil.copy2(src/'models'/n,dst/n)
parts=[];data=(src/'models/racket-pose.onnx').read_bytes();chunk=18*1024*1024
for i,offset in enumerate(range(0,len(data),chunk),1):
 name=f'racket-pose.part{i:02d}.bin';(dst/name).write_bytes(data[offset:offset+chunk]);parts.append(name)
worker=(dst/'inference-worker.js').read_text().replace("'./vendor/ort.webgpu.min.mjs'","'./ort.webgpu.min.mjs'").replace("new URL('vendor/', baseUrl)","new URL('./', baseUrl)")
a=worker.index('async function fetchModel(');b=worker.index('async function releaseSessions()',a)
loader='''async function fetchModel(name, number) {
  const paths = name === 'pose' ? PARTS : [MODEL_FILES[name]];
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

'''.replace('PARTS',json.dumps(parts))
worker=worker[:a]+loader+worker[b:]
worker=worker.replace("ort.env.wasm.wasmPaths = new URL('./', baseUrl).href;", '''ort.env.wasm.wasmPaths = {mjs: new URL('ort-wasm-simd-threaded.asyncify.mjs',baseUrl).href};
    if (!ort.env.wasm.wasmBinary) {
      report('正在准备浏览器推理运行文件…');
      if (typeof DecompressionStream === 'undefined') throw new Error('请升级到最新版Chrome或Edge后重试');
      const runtime = await fetch(new URL('runtime-gzip.bin',baseUrl));
      if (!runtime.ok) throw new Error('运行文件下载失败 HTTP '+runtime.status);
      const binary = new Uint8Array(await new Response(runtime.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
      if (binary[0]!==0 || binary[1]!==97 || binary[2]!==115 || binary[3]!==109) throw new Error('运行文件不完整，请刷新后重试');
      ort.env.wasm.wasmBinary = binary;
    }''')
(dst/'inference-worker.js').write_text(worker)
for n in ('LICENSE','THIRD_PARTY_NOTICES.md'):shutil.copy2(ROOT/n,dst/n)
(dst/'README.md').write_text('''# CourtLine 网页实时人体与球拍识别

在浏览器本地运行，无视频上传。Chrome/Edge优先，支持摄像头和本地视频。
GitHub Pages使用main分支根目录发布本目录，无需Python或付费推理服务器。

本目录含完整浏览器源码与模型：body.onnx、racket.onnx、racket-pose.part*.bin。
球拍关键点大模型按序拼接后加载；切勿删除分块文件。
算法YOLO26人体＋YOLO11球拍检测＋RacketVision RTMPose五关键点。
默认416/640/256输入；低置信度或遮挡时不画球拍轴线；非真实三维测量。
首次下载约85MB资源，后续由浏览器缓存。速度取决于电脑，后台标签页会被浏览器限速。

源码按AGPL-3.0发布，第三方许可证见对应文件。模型来源、转换方法见THIRD_PARTY_NOTICES.md及export_models.py。
模型不会将画面发送到服务器，没有账号或分析日志收集。
''')
for f in (ROOT/'tools').glob('*.py'):shutil.copy2(f,dst/f.name)
(src/'models/manifest.json').exists() and shutil.copy2(src/'models/manifest.json',dst/'model-manifest.json')
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dst.iterdir() if f.is_file()}
(dst/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
print('Pages files:',len(manifest),'MB:',sum(f.stat().st_size for f in dst.iterdir())/1e6)
print('Pose parts:',parts)

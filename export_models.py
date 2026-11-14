"""Export existing local weights to fixed-shape, browser-ready ONNX models."""
from pathlib import Path
import json,shutil,time
import cv2,numpy as np,torch,onnx,onnxruntime as ort
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEST=HERE.parent/'dist/models'
TEMP=HERE/'export-inputs'
TEMP.mkdir(exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
for source,name,size in [
 (ROOT/'realtime-tennis/yolo26n-pose.pt','body',416),
 (ROOT/'realtime-tennis/yolo11n.pt','racket',640)]:
 copy=TEMP/source.name
 if not copy.exists():shutil.copy2(source,copy)
 output=DEST/f'{name}.onnx'
 if not output.exists():
  result=YOLO(str(copy)).export(format='onnx',imgsz=size,opset=17,simplify=False,dynamic=False,device='cpu',nms=False,verbose=False)
  shutil.copy2(result,output)
 print('EXPORTED',name,output.stat().st_size,flush=True)
model=DEST/'racket-pose.onnx'
if not model.exists():
 net=torch.jit.load(str(ROOT/'racketvision-live/models/rtmpose_m_cpu.ts'),map_location='cpu').eval()
 with torch.inference_mode():
  torch.onnx.export(net,torch.zeros(1,3,256,256),str(model),input_names=['images'],output_names=['simcc_x','simcc_y'],opset_version=17,dynamo=False)
 print('EXPORTED racket-pose',model.stat().st_size,flush=True)
contracts={}
for name in ('body','racket','racket-pose'):
 path=DEST/f'{name}.onnx';net=onnx.load(path);onnx.checker.check_model(net)
 session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
 info=dict(file=path.name,bytes=path.stat().st_size,ir_version=net.ir_version,opset=[(x.domain,x.version) for x in net.opset_import],inputs=[dict(name=x.name,shape=x.shape,type=x.type) for x in session.get_inputs()],outputs=[dict(name=x.name,shape=x.shape,type=x.type) for x in session.get_outputs()],ops=sorted(set(x.op_type for x in net.graph.node)))
 size=256 if name=='racket-pose' else (416 if name=='body' else 640)
 result=session.run(None,{'images':np.zeros((1,3,size,size),np.float32)})
 info['inference_shapes']=[list(x.shape) for x in result];info['finite']=all(np.isfinite(x).all() for x in result)
 contracts[name]=info
 print('CONTRACT',name,json.dumps(info),flush=True)
(HERE/'model-contract.json').write_text(json.dumps(contracts,indent=2))

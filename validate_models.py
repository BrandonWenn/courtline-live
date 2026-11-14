from pathlib import Path
import json,time,hashlib,argparse
import cv2,numpy as np,torch,onnxruntime as ort
from ultralytics import YOLO
from ultralytics.nn.modules import Detect,SPPF
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;MODELS=HERE.parent/'dist/models'
torch.set_num_threads(2)
parser=argparse.ArgumentParser();parser.add_argument('--video',required=True);parser.add_argument('--frame',type=int,default=180);parser.add_argument('--reference-boxes',type=Path,required=True);args=parser.parse_args()
cap=cv2.VideoCapture(args.video);cap.set(cv2.CAP_PROP_POS_FRAMES,args.frame);ok,frame=cap.read();cap.release()
if not ok:raise RuntimeError('No validation image')

def letterbox(im,size):
 h,w=im.shape[:2];ratio=min(size/w,size/h);nw,nh=round(w*ratio),round(h*ratio);left=round((size-nw)/2-.1);top=round((size-nh)/2-.1)
 resized=cv2.resize(im,(nw,nh));out=np.full((size,size,3),114,np.uint8);out[top:top+nh,left:left+nw]=resized
 tensor=np.ascontiguousarray(out[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
 return tensor,ratio,left,top

report={}
for name,size,source in [('body',416,ROOT/'realtime-tennis/yolo26n-pose.pt'),('racket',640,ROOT/'realtime-tennis/yolo11n.pt')]:
 x,ratio,left,top=letterbox(frame,size)
 model=YOLO(str(source)).model.cpu().eval();model.end2end=True;model.fuse(verbose=False,imgsz=size)
 for m in model.modules():
  if isinstance(m,SPPF):m.export=True
  if isinstance(m,Detect):
   m.dynamic=False;m.export=True;m.format='onnx';m.max_det=300;m.agnostic_nms=False;m.xyxy=False;m.shape=None
 with torch.inference_mode():ref=model(torch.from_numpy(x)).numpy()
 session=ort.InferenceSession(str(MODELS/f'{name}.onnx'),providers=['CPUExecutionProvider']);actual=session.run(None,{'images':x})[0]
 # E2E TopK tie ordering may permute low-confidence candidates. Compare meaningful rows only.
 if name=='body':
  high=actual[0,actual[0,:,4]>.4]
  refhigh=ref[0,ref[0,:,4]>.4]
  errors=[]
  for row in high:
   nearest=refhigh[np.argmin(np.linalg.norm(refhigh[:,:4]-row[:4],axis=1))]
   errors.append(float(np.abs(row-nearest).max()))
  result=dict(shape=list(actual.shape),detections=len(high),high_confidence_max_abs_error=max(errors,default=0),top_rows=high[:3].tolist())
  assert len(high)>0 and len(high)==len(refhigh)
  assert max(errors)<.02
 else:
  error=float(np.abs(actual-ref).max());result=dict(shape=list(actual.shape),raw_max_abs_error=error)
  cls=38;scores=actual[0,4+cls];inds=np.flatnonzero(scores>.25)
  result['tennis_racket_raw_candidates']=len(inds);result['top_racket_score']=float(scores.max());result['top_racket_xywh']=actual[0,:4,int(scores.argmax())].tolist()
  assert error<.03
 report[name]=result
 print(name,json.dumps(result),flush=True)
# Reference crop from the known frame's exported racket bbox.
data=json.load(open(args.reference_boxes))
bbox=np.array(data[args.frame]['rackets'][0]['box'],np.float32)
center=(bbox[:2]+bbox[2:])/2;side=float(max(bbox[2:]-bbox[:2])*1.25)
src=np.array([center,center+[-side/2,0],center+[-side/2,-side/2]],np.float32);dst=np.array([[128,128],[0,128],[0,0]],np.float32)
crop=cv2.warpAffine(frame,cv2.getAffineTransform(src,dst),(256,256),flags=cv2.INTER_LINEAR)
x=np.ascontiguousarray(((crop[:,:,::-1].astype(np.float32)-[123.675,116.28,103.53])/[58.395,57.12,57.375]).transpose(2,0,1)[None],dtype=np.float32)
net=torch.jit.load(str(ROOT/'racketvision-live/models/rtmpose_m_cpu.ts'),map_location='cpu').eval()
with torch.inference_mode():refs=[v.numpy() for v in net(torch.from_numpy(x))]
session=ort.InferenceSession(str(MODELS/'racket-pose.onnx'),providers=['CPUExecutionProvider']);actual=session.run(None,{'images':x})
errors=[float(np.max(np.abs(a-b))) for a,b in zip(actual,refs)]
assert max(errors)<.001
xx,yy=actual[0][0],actual[1][0];coords=np.stack([xx.argmax(-1),yy.argmax(-1)],-1)/2/256*side+center-side/2
scores=np.minimum(xx.max(-1),yy.max(-1))
report['racket-pose']=dict(max_abs_errors=errors,reference_bbox=bbox.tolist(),keypoint_order=['top','bottom','handle','left','right'],coords=coords.tolist(),scores=scores.tolist())
report['validation_source']=f'Local video frame {args.frame}; image not included in public package'
for name in ('body','racket','racket-pose'):
 b=(MODELS/f'{name}.onnx').read_bytes();report[name]['sha256']=hashlib.sha256(b).hexdigest()
(HERE/'validation-report.json').write_text(json.dumps(report,indent=2));print('VALIDATION PASS',flush=True)

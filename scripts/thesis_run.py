"""Fresh fixed holdout experiment; never overwrites prior experiments."""
import os, sys, csv, json, hashlib, random, time, subprocess, platform, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor_packages'),str(ROOT)]
import torch, torchvision
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import build_transforms
from src.model import build_model

def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 os.chdir(ROOT)
 run=ROOT/'outputs'/('thesis_'+time.strftime('%Y%m%d_%H%M%S'))
 run.mkdir(); data=run/'dataset'; out=run/'training'; figs=run/'figures'; figs.mkdir()
 dump(ROOT/'outputs'/'latest_thesis_run.json',{'run':str(run)})
 assert torch.cuda.is_available()
 import ctypes, winreg
 class Mem(ctypes.Structure):
  _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','avail','page','availpage','virtual','availvirtual','extended']]
 m=Mem(); m.length=ctypes.sizeof(m); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
 with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k: cpu=winreg.QueryValueEx(k,'ProcessorNameString')[0]
 dump(run/'environment.json',{'cpu':cpu,'ram_bytes':m.total,'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'os':platform.platform(),'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)})
 # Decode before hashing so exact pixels with different encodings stay together.
 groups={}; invalid=[]; source=ROOT/'data/domestic_garbage'
 for split in ['train','val']:
  for cls in ['dry','hazardous','recyclable','wet']:
   for f in sorted((source/split/cls).rglob('*')):
    if f.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}: continue
    try:
     with Image.open(f) as im:
      im=ImageOps.exif_transpose(im).convert('RGB'); digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
      thumb=im.resize((8,8)).convert('L'); vals=list(thumb.getdata()); avg=sum(vals)/64
      ah=sum((v>=avg)<<i for i,v in enumerate(vals))
     groups.setdefault(digest,[]).append((f,cls,ah))
    except Exception as e: invalid.append([str(f),str(e)])
 print('Decoded unique images',len(groups),flush=True)
 # Conservative near-duplicate groups: average hash within 4 bits, same label.
 records=[]; excluded=[]; rng=random.Random(20261102)
 for cls in ['dry','hazardous','recyclable','wet']:
  items=[]
  for digest,entries in groups.items():
   if len({e[1] for e in entries})>1:
    if entries[0][1]==cls: excluded.append([digest,'conflicting labels'])
    continue
   if entries[0][1]==cls: items.append((digest,entries[0]))
  parent=list(range(len(items)))
  def find(i):
   while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
   return i
  for i,(_,(_,_,h)) in enumerate(items):
   for j in range(i):
    if bin(h^items[j][1][2]).count('1')<=4: parent[find(i)]=find(j)
  buckets={}
  for i,x in enumerate(items): buckets.setdefault(find(i),[]).append(x)
  buckets=list(buckets.values()); rng.shuffle(buckets); counts={'train':0,'val':0,'test':0}; targets={'train':.7,'val':.15,'test':.15}
  for bucket in sorted(buckets,key=len,reverse=True):
   split=min(counts,key=lambda s:counts[s]/targets[s]); counts[split]+=len(bucket)
   for digest,(f,_,ah) in bucket:
    dest=data/split/cls/(digest+f.suffix.lower()); dest.parent.mkdir(parents=True,exist_ok=True)
    try: os.link(f,dest)
    except OSError: shutil.copy2(f,dest)
    records.append({'source':str(f),'image':str(dest),'class':cls,'split':split,'pixel_sha256':digest,'average_hash':str(ah)})
  print(cls,counts,'groups',len(buckets),flush=True)
 with (run/'dataset_manifest.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
 dump(run/'split_audit.json',{'seed':20261102,'method':'decoded RGB SHA256 dedup; within-class average hash Hamming <=4 connected components assigned together; heuristic cannot exclude every crop or semantic duplicate','invalid':invalid,'conflicting_labels':excluded,'counts':{s:{c:sum(r['split']==s and r['class']==c for r in records) for c in ['dry','hazardous','recyclable','wet']} for s in ['train','val','test']},'source':'existing local data/domestic_garbage; original download provenance requires further verification; hazardous covers batteries only','test_policy':'fixed before this fresh training; never used for model selection; source images may have appeared in historical experiments; this is a fresh-run holdout, not a previously unseen external collection'})
 for c in ['dry','hazardous','recyclable','wet']: assert all(any(r['class']==c and r['split']==s for r in records) for s in ['train','val','test'])
 subprocess.run([sys.executable,'-u','scripts/train_with_gpu_monitor.py','--epochs','12','--data-dir',str(data),'--out-dir',str(out)],check=True)
 checkpoint=torch.load(out/'best_model.pth',map_location='cpu',weights_only=False)
 classes=checkpoint['class_names']; model=build_model(checkpoint['model_name'],4,False).cuda().eval(); model.load_state_dict(checkpoint['model_state']); torch.set_num_threads(4)
 tf=build_transforms(False); tests=[r for r in records if r['split']=='test']; rows=[]
 with torch.inference_mode():
  for _ in range(5): model(torch.zeros(1,3,224,224,device='cuda'))
  torch.cuda.synchronize(); start=time.perf_counter()
  for r in tests:
   t=time.perf_counter()
   with Image.open(r['image']) as im: x=tf(im.convert('RGB')).unsqueeze(0).cuda()
   probs=model(x).softmax(1)[0].cpu().tolist(); torch.cuda.synchronize(); elapsed=time.perf_counter()-t; pred=classes[max(range(4),key=lambda i:probs[i])]
   rows.append({'image':r['image'],'true_class':r['class'],'pred_class':pred,'confidence':max(probs),'correct':int(pred==r['class']),'seconds':elapsed,**dict(zip(classes,probs))})
 total_time=time.perf_counter()-start
 with (run/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 dump(run/'test_conditions.json',{'checkpoint_sha256':hashlib.sha256((out/'best_model.pth').read_bytes()).hexdigest(),'epoch':checkpoint['epoch'],'batch_size':1,'precision':'float32','warmup':5,'seconds':total_time,'mean_seconds':total_time/len(rows),'count':len(rows),'scope':'wall time includes image read, preprocessing, transfer, forward, softmax and synchronization; excludes model load and warmup'})
 subprocess.run([sys.executable,'scripts/analyze_results.py','--eval-csv',str(run/'test_predictions.csv'),'--out-dir',str(run/'analysis'),'--split-name','Fresh-run independent holdout'],check=True)
 for name in ['loss_curve.png','accuracy_curve.png','gpu_monitor.png']: shutil.copy2(out/name,figs/name)
 shutil.copy2(run/'analysis/confusion_matrix.png',figs/'confusion_matrix.png')
 def montage(name,selected,title):
  fig,axs=plt.subplots(2,4,figsize=(13,7)); axs=list(axs.flat)
  for ax in axs: ax.axis('off')
  for ax,r in zip(axs,selected[:8]):
   with Image.open(r['image']) as im: ax.imshow(im.convert('RGB'))
   ax.set_title(f"True: {r['true_class']}\nPred: {r['pred_class']} ({float(r['confidence']):.2f})",fontsize=9)
  fig.suptitle(title); fig.tight_layout(); fig.savefig(figs/name,dpi=170); plt.close(fig)
 montage('correct_examples.png',[r for c in classes for r in [next((x for x in rows if x['true_class']==c and x['correct']),None)] if r]+[r for r in rows if r['correct']][:4],'Correct predictions (fixed manifest order)')
 montage('wrong_examples.png',[r for r in rows if not r['correct']],'Misclassified test images (first eight in manifest order)')
 montage('dataset_examples.png',[r for c in classes for r in records if r['class']==c and r['split']=='train'][:0] if False else [{'image':r['image'],'true_class':r['class'],'pred_class':'label','confidence':1} for c in classes for r in [next(x for x in records if x['class']==c and x['split']=='train')]],'Training dataset examples')
 report=ROOT.parent/'毕业论文'/'实验记录'; report.mkdir(exist_ok=True)
 (report/(run.name+'.md')).write_text('# 本次真实训练与测试\n\n原始记录：'+str(run)+'\n\n'+(run/'analysis/experiment_report.md').read_text(encoding='utf-8')+'\n\n数据为本次从头训练的固定holdout，曾参与历史实验的图片不属于全新外部数据。去重为启发式检查；原始来源许可和四分类映射仍需人工核验，有害类仅覆盖电池。\n\n'+ '\n\n'.join('!['+f.stem+']('+f.as_posix()+')' for f in figs.glob('*.png')),encoding='utf-8')
 print('COMPLETE',run,flush=True)
if __name__=='__main__': main()

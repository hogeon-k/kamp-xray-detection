from pathlib import Path
import sys,json,os,time,hashlib
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
root=Path(__file__).resolve().parents[1]
out=root/'report_audit'
os.environ['IPYTHONDIR']=str(out/'ipython')
spec=out/'kernels'/'report-audit';spec.mkdir(parents=True,exist_ok=True)
(spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Report audit','language':'python'}),encoding='utf-8')
def run(folder,label):
    p=folder/'KAMP_Xray_Final_Analysis.ipynb';before=hashlib.sha256(p.read_bytes()).hexdigest()
    nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
    km=KernelManager(kernel_name='report-audit',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(out/'kernels')]))
    client=NotebookClient(nb,km=km,timeout=120,resources={'metadata':{'path':str(folder)}})
    t=time.time();client.execute();nbformat.write(nb,out/f'{label}_executed.ipynb')
    result=dict(label=label,code_cells=sum(c.cell_type=='code' for c in nb.cells),errors=[o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error'],seconds=round(time.time()-t,3),original_unchanged=before==hashlib.sha256(p.read_bytes()).hexdigest(),python=sys.version.split()[0])
    print(result);return result
results=[run(root,'root'),run(root/'KAMP_Xray_Submission','submission')]
(out/'kernel_verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

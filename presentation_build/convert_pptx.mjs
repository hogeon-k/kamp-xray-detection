import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='C:/workspace/Kamp_Xray';
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(root,'KAMP_Xray_Presentation.pptx')));
const out=path.join(root,'presentation_build','pdf_from_pptx');await fs.mkdir(out,{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:2.5});
 await fs.writeFile(path.join(out,`page-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));console.log('PPTX converted page',i+1);
}

import {Presentation} from '@oai/artifact-tool';
const p=Presentation.create();p.slides.add();
for(const format of ['pdf','svg']){
try {const b=await p.export({slide:p.slides.items[0],format});console.log(format,b.type,b.size);}catch(e){console.log(format,e.message);}
}

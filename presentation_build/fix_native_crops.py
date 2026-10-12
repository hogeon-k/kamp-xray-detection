"""Preserve original image bytes and repair native PPTX crop metadata only.

Artifact Tool 2.8.51 recomputes a centered fit crop on export, replacing the
explicit panel crop. This fixes a:srcRect after export without raster editing.
"""
import sys,zipfile,xml.etree.ElementTree as E
from pathlib import Path
p=Path(sys.argv[1])
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
for k,v in ns.items(): E.register_namespace(k,v)
E.register_namespace('r','http://schemas.openxmlformats.org/officeDocument/2006/relationships')
crops={
 3:[(i*.2,.04,1-(i+1)*.2,0) for i in range(3)],
 7:[(360/1400,95/720,700/1400,290/720),(360/1400,95/720,700/1400,284/720)],
 8:[(0,0,.5,.5),(.5,0,0,.5),(380/576,135/444,142/576,249/444)]
}
if len(sys.argv)>2 and sys.argv[2]=='submission':
 crops={n+1:value for n,value in crops.items()}
with zipfile.ZipFile(p) as z: entries=[(i,z.read(i.filename)) for i in z.infolist()]
stage=p.with_suffix('.cropped.pptx')
with zipfile.ZipFile(stage,'w',zipfile.ZIP_DEFLATED) as z:
 for info,data in entries:
  for n,cropvalues in crops.items():
   if info.filename!=f'ppt/slides/slide{n}.xml': continue
   root=E.fromstring(data);pics=root.findall('.//p:pic',ns)
   assert len(pics)==len(cropvalues)
   for pic,crop in zip(pics,cropvalues):
    fill=pic.find('p:blipFill',ns)
    rect=fill.find('a:srcRect',ns)
    if rect is None:
     rect=E.Element('{'+ns['a']+'}srcRect');fill.insert(1,rect)
    rect.attrib={k:str(round(v*100000)) for k,v in zip(('l','t','r','b'),crop)}
   data=E.tostring(root,encoding='utf-8',xml_declaration=True)
  z.writestr(info,data)
stage.replace(p)
print('Native crop metadata repaired; source image bytes unchanged.')

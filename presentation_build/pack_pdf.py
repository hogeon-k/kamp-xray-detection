from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
from pypdf import PdfReader
r=Path.cwd(); out=r/'KAMP_Xray_Presentation.pdf'
c=canvas.Canvas(str(out),pagesize=(960,540),pageCompression=1)
c.setTitle('KAMP X-ray Presentation');c.setAuthor('KAMP Xray Project')
c.setSubject('Converted from KAMP_Xray_Presentation.pptx at 240 DPI')
for i in range(1,11):
 p=r/f'presentation_build/pdf_from_pptx/page-{i}.png';im=Image.open(p);assert im.size==(3200,1800)
 c.drawImage(ImageReader(im),0,0,width=960,height=540);c.showPage()
c.save();pdf=PdfReader(out);assert len(pdf.pages)==10
assert all(tuple(map(float,p.mediabox[2:]))==(960.,540.) for p in pdf.pages)
print('PDF converted: 10 pages, 16:9, 240 DPI, bytes:',out.stat().st_size)

import fs from 'node:fs/promises';
import {Presentation} from '@oai/artifact-tool';
const p=Presentation.create();
console.log(p.help('*',{search:'text style anchor verticalAlignment',include:['notes'],maxChars:3000}).text ?? JSON.stringify(p.help('*',{search:'text style anchor verticalAlignment',include:['notes'],maxChars:3000})));

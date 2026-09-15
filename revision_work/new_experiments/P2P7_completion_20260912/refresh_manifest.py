"""Refresh the package manifest after adding reports; never changes scientific files."""
from pathlib import Path
import hashlib,json,os
HERE=Path(__file__).resolve().parent
files=[p for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts]
configured=[v.encode() for name in ('DEEPSEEK_API_KEY','DASHSCOPE_API_KEY','QWEN_API_KEY') if (v:=os.environ.get(name,''))]
assert all(all(value not in p.read_bytes() for value in configured) for p in files),'A configured credential appears in an artifact'
manifest=[{'file':p.relative_to(HERE).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files]
(HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest),'configured_credential_values_absent':True}))

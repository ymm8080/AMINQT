import os
files = []
for root, dirs, fnames in os.walk('.'):
    if '.git' in root or 'node_modules' in root or '__pycache__' in root:
        continue
    for f in fnames:
        if f.endswith('.py'):
            files.append(os.path.join(root, f))
bom_files = []
for f in files:
    try:
        with open(f, 'rb') as fh:
            if fh.read(3) == b'\xef\xbb\xbf':
                bom_files.append(f)
    except:
        pass
if bom_files:
    print('Files with BOM:')
    for f in bom_files:
        print(f'  {f}')
else:
    print('No BOM files found')

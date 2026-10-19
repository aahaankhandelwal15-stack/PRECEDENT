import os
base = 'D:/Projects/PRECEDENT'
# Search all .py files for function definitions
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as fh:
                    content = fh.read()
                # Check for function definitions
                if 'def retrieve_chunks' in content:
                    print(f'{path}: has def retrieve_chunks')
                if 'def decompose_question' in content:
                    print(f'{path}: has def decompose_question')
            except:
                pass
import os
base = 'D:/Projects/PRECEDENT'
functions = ['def decompose_question', 'def retrieve_chunks', 'def BoundedIterator',
             'def get_refusal_message', 'MAX_ITERATIONS', 'DEFAULT_REFUSAL_MESSAGES',
             'def _compute_final_confidence', 'def _build_refusal_response',
             'def _should_short_circuit_to_refusal', 'def _determine_refusal_category',
             'def compute_confidence_from_inputs', 'def compute_confidence_from_retrieval',
             'def get_confidence_scorer_model', 'def set_confidence_scorer_model']

found = {}
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as fh:
                    content = fh.read()
                for func in functions:
                    if func in content:
                        if path not in found:
                            found[path] = []
                        if func not in found[path]:
                            found[path].append(func)
            except:
                pass

for path, funcs in sorted(found.items()):
    print(f'{path}:')
    for func in funcs:
        print(f'  - {func}')
    print()
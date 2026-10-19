import ast
with open('D:\\Projects\\PRECEDENT\\app\\core\\retrieval.py', 'r') as f:
    code = f.read()
try:
    ast.parse(code)
    print('Syntax OK')
except SyntaxError as e:
    print(f'Syntax error at line {e.lineno}: {e.msg}')
import sys, traceback
sys.path.insert(0, 'src')
from inputs import load_input

try:
    load_input({'type': 'GoogleASRInput'})
    print('Loaded GoogleASRInput successfully')
except Exception as e:
    print('EXC:', type(e), e)
    traceback.print_exc()

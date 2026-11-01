"""Narrow portfolio on a stage-6 round output before plateau. usage: early_narrow.py <arm> <round>"""
import sys
from pathlib import Path
from common import *
sys.path.insert(0,str(CMP/'py'))
from attacks_port import attack_circuit
arm,r=sys.argv[1],int(sys.argv[2]);name=f'{arm}_r{r}'
F=RUN/'stage6'/arm/'v2'/f'round{r}'/'output.mpmct1';source=RUN/'stages35'/arm/'source.mpmct'
save(RUN/'early_narrow'/name/'input.json',dict(circuit=str(F),gates=size(F),sha256=digest(F)))
fails=attack_circuit(name,F,source,2)
save(RUN/'early_narrow'/name/'DONE.json',dict(failures=fails));print('EARLY NARROW DONE',name,fails,flush=True)

import sys
def load(path):
    with open(path) as f:
        hdr=f.readline().split(); w=int(hdr[1])
        tg=[]
        for line in f:
            tg.append(int(line.split()[0]))
    return w, tg
for path in sys.argv[1:]:
    w, tg = load(path)
    print(f"{path.split('/')[-1]:>22} wires={w} gates={len(tg)}")
    for W in (2000, 10000, 50000):
        # slide in strides, count wires never written in the window
        counts=[]
        for start in range(0, max(1,len(tg)-W), max(1,(len(tg)-W)//20 or 1)):
            seen=set(tg[start:start+W])
            counts.append(w-len(seen))
        if counts:
            print(f"    window {W:>6}: never-written wires  min {min(counts):3d}  median {sorted(counts)[len(counts)//2]:3d}  max {max(counts):3d}")

import sys, statistics
def census(path, n_data):
    w=None
    writes=None; reads=None
    with open(path) as f:
        hdr=f.readline().split()
        w=int(hdr[1]); m=int(hdr[2])
        writes=[0]*w; reads=[0]*w
        for line in f:
            t=line.split()
            tgt=int(t[0]); k=int(t[2])
            writes[tgt]+=1
            for i in range(k):
                reads[int(t[3+2*i])]+=1
    return w, writes, reads

for path in sys.argv[1:]:
    w, writes, reads = census(path, 256)
    carriers=list(range(0,512)); band=list(range(512,w))
    cw=[writes[i] for i in carriers]; bw=[writes[i] for i in band]
    cr=[reads[i] for i in carriers]; br=[reads[i] for i in band]
    zero=[i for i in range(w) if writes[i]==0]
    # how separable is the band by write count? min carrier write vs max band write
    print(f"{path.split('/')[-1]:>22}  wires={w}")
    print(f"   writes  carrier: min {min(cw)} med {int(statistics.median(cw))} max {max(cw)}   band: min {min(bw)} med {int(statistics.median(bw))} max {max(bw)}")
    print(f"   reads   carrier: min {min(cr)} med {int(statistics.median(cr))} max {max(cr)}   band: min {min(br)} med {int(statistics.median(br))} max {max(br)}")
    print(f"   never-written wires: {len(zero)}   band separable by a write-count threshold: {max(bw) < min(cw)}")

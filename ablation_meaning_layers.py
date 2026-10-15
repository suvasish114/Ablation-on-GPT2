"""Ablate representation depth, holding mean pooling fixed."""
from para_common import *
if __name__ == '__main__':
    args=parser(__doc__, 'plot_meaning').parse_args(); args.experiment='layers'
    frame,tok,model=setup(args)
    emb=representations(frame,tok,model)
    meaning_report(emb,frame,args.output,'layers',[(f'layer_{i:02d}',emb[:,:,i,0]) for i in range(emb.shape[2])])

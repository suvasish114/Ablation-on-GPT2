"""Ablate mean, last-token and max pooling, holding the final layer fixed."""
from para_common import *
if __name__ == '__main__':
    args=parser(__doc__, 'plot_meaning').parse_args(); args.experiment='pooling'
    frame,tok,model=setup(args)
    emb=representations(frame,tok,model)
    meaning_report(emb,frame,args.output,'pooling',[(name,emb[:,:,-1,i]) for i,name in enumerate(['mean','last_token','max'])])

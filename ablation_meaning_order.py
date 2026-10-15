"""Ablate word order with fixed final-layer mean pooling. Perturb whitespace words."""
from para_common import *
if __name__ == '__main__':
    args=parser(__doc__, 'plot_meaning').parse_args(); args.experiment='order'
    frame,tok,model=setup(args); rng=np.random.default_rng(args.seed); conditions=[]; transformed=[]
    for condition in ['intact','reverse_words','shuffle_words']:
        edited=frame.copy()
        for idx in frame.index:
            for col in COLUMNS:
                words=frame.at[idx,col].split()
                if condition=='reverse_words': words=words[::-1]
                elif condition=='shuffle_words': words=list(rng.permutation(words))
                edited.at[idx,col]=frame.at[idx,col] if condition=='intact' else ' '.join(words)
                transformed.append(dict(row_index=idx,condition=condition,column=col,text=edited.at[idx,col]))
        conditions.append((condition,representations(edited,tok,model)[:,:,-1,0]))
    pd.DataFrame(transformed).to_csv(args.output/'order_inputs.csv',index=False)
    meaning_report(None,frame,args.output,'order',conditions)

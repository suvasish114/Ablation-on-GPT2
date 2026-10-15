"""Save four attention heatmaps for a zero-based CSV data-row index."""
from para_common import *

@torch.inference_mode()
def run(args):
    frame,tok,model=setup(args)
    if not 0 <= args.row < len(frame): raise ValueError(f'--row must be between 0 and {len(frame)-1}')
    if not -model.config.n_layer <= args.layer < model.config.n_layer: raise ValueError('Invalid attention layer')
    if args.head is not None and not 0 <= args.head < model.config.n_head: raise ValueError('Invalid attention head')
    for col in COLUMNS:
        text=frame.iloc[args.row][col]; ids=encode(tok,model,text)
        attention=model(ids,output_attentions=True,use_cache=False).attentions[args.layer][0]
        matrix=(attention.mean(0) if args.head is None else attention[args.head]).cpu().numpy()
        labels=[f'{i}: {tok.decode([int(t)])!r}' for i,t in enumerate(ids[0])]
        layer=args.layer%model.config.n_layer; head='mean_heads' if args.head is None else f'head_{args.head}'
        stem=f'row_{args.row}_{col}_layer_{layer}_{head}'
        save_matrix(matrix,labels,f'{col}\n{text}\nlayer {layer}, {head}',args.output/f'{stem}.png',xlabel='Key token (attended to)',ylabel='Query token',vmin=0,vmax=1)
        np.save(args.output/f'{stem}.npy',matrix)
        (args.output/f'{stem}_tokens.json').write_text(json.dumps(dict(text=text,tokens=labels),indent=2))
        print(args.output/f'{stem}.png')
if __name__=='__main__':
    p=parser(__doc__,'plot_attention'); p.add_argument('--row',type=int,required=True); p.add_argument('--layer',type=int,default=-1); p.add_argument('--head',type=int)
    args=p.parse_args(); args.experiment=f'attention_row_{args.row}'; run(args)

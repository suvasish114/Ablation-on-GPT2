"""Teacher-forced next-token accuracy, sentence perplexity, BPE and word surprisal."""
import re
import math
from para_common import *

@torch.inference_mode()
def run(args):
    frame,tok,model=setup(args); sentences=[]; tokens=[]; words=[]
    for idx,row in frame.iterrows():
        for col in COLUMNS:
            text=row[col]; ids=encode(tok,model,text,bos=True)
            hidden=model(ids,use_cache=False).last_hidden_state
            logits=F.linear(hidden[:,:-1],model.get_input_embeddings().weight)[0].float()
            targets=ids[0,1:]; nll=F.cross_entropy(logits,targets,reduction='none'); pred=logits.argmax(-1)
            offsets=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
            losses=nll.cpu().numpy(); correct=(pred==targets).cpu().numpy()
            sentences.append(dict(row_index=idx,column=col,text=text,tokens=len(losses),nll_sum=float(losses.sum()),mean_nll=float(losses.mean()),perplexity=math.exp(float(losses.mean())),next_token_accuracy=float(correct.mean())))
            for j,((start,end),loss,hit) in enumerate(zip(offsets,losses,correct)):
                tokens.append(dict(row_index=idx,column=col,token_index=j,token_id=int(targets[j]),token=tok.decode([int(targets[j])]),start=start,end=end,surprisal_nats=float(loss),surprisal_bits=float(loss/math.log(2)),correct=bool(hit),predicted_token=tok.decode([int(pred[j])])))
            for j,match in enumerate(re.finditer(r'\S+',text)):
                # Assign each BPE token exactly once by the last nonspace character in its span.
                selected=[k for k,(s,e) in enumerate(offsets) if match.start() <= e-1 < match.end()]
                words.append(dict(row_index=idx,column=col,word_index=j,word=match.group(),bpe_tokens=len(selected),surprisal_nats=float(losses[selected].sum()),surprisal_bits=float(losses[selected].sum()/math.log(2))))
        if idx < args.sample_plots:
            row_words=[w for w in words if w['row_index']==idx]
            max_words=max(w['word_index'] for w in row_words)+1
            fig,ax=plt.subplots(figsize=(max(12,max_words*.45),6))
            for col in COLUMNS:
                entries=[w for w in row_words if w['column']==col]
                ax.plot([w['word_index']+1 for w in entries],
                        [w['surprisal_bits'] for w in entries],
                        marker='o', markersize=4, linewidth=1.5, label=col)
            ax.set(title=f'Row {idx}: word surprisal across sentence variants',
                   xlabel='Word position (1-based; positions are not semantic alignments)',
                   ylabel='Word surprisal (bits)')
            ax.set_xticks(range(1,max_words+1))
            ax.grid(alpha=.25); ax.legend()
            fig.tight_layout()
            fig.savefig(args.output/f'row_{idx}_words.png',dpi=160)
            plt.close(fig)
        if (idx+1)%50==0: print(f'Scored {idx+1}/{len(frame)} rows',flush=True)
    df=pd.DataFrame(sentences); df.to_csv(args.output/'sentence_metrics.csv',index=False)
    pd.DataFrame(tokens).to_csv(args.output/'token_surprisal.csv',index=False)
    pd.DataFrame(words).to_csv(args.output/'word_surprisal.csv',index=False)
    summary=[]
    for col,group in df.groupby('column',sort=False):
        summary.append(dict(column=col,corpus_perplexity=math.exp(group.nll_sum.sum()/group.tokens.sum()),token_accuracy=float((group.next_token_accuracy*group.tokens).sum()/group.tokens.sum())))
    pd.DataFrame(summary).to_csv(args.output/'summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,metric in zip(axes,['perplexity','next_token_accuracy']):
        ax.boxplot([df.loc[df.column==c,metric] for c in COLUMNS],tick_labels=COLUMNS)
        ax.tick_params(axis='x',rotation=35); ax.set_ylabel(metric)
        if metric=='perplexity': ax.set_yscale('log')
    fig.tight_layout(); fig.savefig(args.output/'sentence_metrics.png',dpi=160); plt.close(fig)

if __name__=='__main__':
    p=parser(__doc__,'plot_surprisal'); p.add_argument('--sample-plots',type=int,default=3)
    args=p.parse_args(); args.experiment='surprisal'; run(args)

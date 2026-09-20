import json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=Path('report_assets'); out.mkdir(exist_ok=True)
stats=json.load(open('merge_domain_graph/delta_stats.json')); order=['v1','v2','v3','v6','v7','v8','v9']
nodes=[stats[k]['nodes'] for k in order]; edges=[stats[k]['edges'] for k in order]; delta=[stats[k]['delta_total'] for k in order]
fig,ax=plt.subplots(2,1,figsize=(11,8),sharex=True)
ax[0].plot(order,nodes,'o-',lw=2.5,label='Nodes',color='#1f77b4'); ax[0].plot(order,edges,'o-',lw=2.5,label='Edges',color='#ff7f0e')
for xs,ys in [(order,nodes),(order,edges)]:
 for x,y in zip(xs,ys): ax[0].annotate(str(y),(x,y),xytext=(0,7 if ys is nodes else -15),textcoords='offset points',ha='center',fontsize=9)
ax[0].set_title('Graph size: v1 to v9',loc='left',weight='bold'); ax[0].set_ylabel('Count'); ax[0].grid(axis='y',alpha=.25); ax[0].legend(frameon=False,ncol=2)
ax[1].plot(order,delta,'o-',lw=2.5,label='Integration delta edges',color='#2ca02c')
for x,y in zip(order,delta): ax[1].annotate(str(y),(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
ax[1].set_title('Integration delta: coverage to sealed tree',loc='left',weight='bold'); ax[1].set_ylabel('Delta edges'); ax[1].set_xlabel('Merge version'); ax[1].grid(axis='y',alpha=.25); ax[1].legend(frameon=False)
fig.suptitle('v9 minibatch merge progress (5 papers; BOHR_ID 20732204)',fontsize=15,weight='bold',y=.98); fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(out/'v9_progress_line.png',dpi=180,bbox_inches='tight'); plt.close(fig)
sel=['v1','v6','v8','v9']; names=['Nodes','Edges','Delta edges','Deduction %','Abduction %','Contradiction %']; raw=[]
for k in sel:
 d=stats[k]['delta']; t=stats[k]['delta_total']; raw.append([stats[k]['nodes'],stats[k]['edges'],t,100*d.get('deduction',0)/t,100*d.get('abduction',0)/t,100*d.get('contradiction',0)/t])
a=np.array(raw,float); scaled=a/np.max(a,axis=0)*100; ang=np.linspace(0,2*np.pi,len(names),endpoint=False).tolist(); ang += ang[:1]
fig=plt.figure(figsize=(9,8)); ax=plt.subplot(111,polar=True); colors=['#4c78a8','#f58518','#54a24b','#e45756']
for i,k in enumerate(sel):
 vals=scaled[i].tolist()+[scaled[i,0]]; ax.plot(ang,vals,lw=2,color=colors[i],label=f'{k} (Nodes {int(raw[i][0])} / Edges {int(raw[i][1])})'); ax.fill(ang,vals,color=colors[i],alpha=.08)
ax.set_thetagrids(np.degrees(ang[:-1]),names,fontsize=10); ax.set_ylim(0,100); ax.set_yticks([25,50,75,100]); ax.set_yticklabels(['25','50','75','100'],fontsize=8,color='gray'); ax.grid(alpha=.25); ax.set_title('Radar: selected merge versions (normalized)',pad=24,weight='bold'); ax.legend(loc='lower center',bbox_to_anchor=(.5,-.23),frameon=False,ncol=2,fontsize=9); fig.text(.5,.02,'Share axes use delta-edge composition; count axes are normalized only for plotting.',ha='center',fontsize=8,color='dimgray'); fig.tight_layout(rect=[0,0,1,.92]); fig.savefig(out/'v9_radar.png',dpi=180,bbox_inches='tight'); plt.close(fig)

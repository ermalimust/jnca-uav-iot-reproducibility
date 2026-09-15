"""Exact rational finite-pattern solver; independent scalar reference included."""
import itertools
from fractions import Fraction as F
import numpy as np

PATTERNS=np.array(list(itertools.product([False,True],repeat=5)),dtype=bool)
THETA=np.array([840000,560000,700000,1400000,680000],dtype=np.int64)
INF_N,INF_D=1,0

def lt(an,ad,bn,bd):
    return (ad!=0)&((bd==0)|(an*bd<bn*ad))
def eq(an,ad,bn,bd):
    return ((ad==0)&(bd==0))|((ad!=0)&(bd!=0)&(an*bd==bn*ad))

def values_bits(q):
    risk=2*q[:,0]+2*q[:,1]+2*q[:,2]+q[:,3]
    z=np.column_stack([risk,2*q[:,1],2*q[:,3],risk,2*q[:,2]])
    bits=z>=THETA;bits[:,2:]=z[:,2:]<THETA[2:]
    return z,bits

def admission(bits,flags):
    bits=np.broadcast_to(bits,(len(flags),5))
    s,r,v,vr,e=bits.T;sf,rf,vf,ef=flags.T
    ok=np.ones((len(flags),6),bool)
    ok[:,0]=~((sf&s)|(rf&r));ok[:,1]=~(rf&r)
    ok[:,3]=~((rf&r)|(ef&e));ok[:,4]=~(rf&r)
    ok[:,5]=~(vf&v&vr)
    return ok

def choose(scores,ok,exposed,rank):
    admitted=ok&exposed;has=admitted.any(axis=1)
    # Stable ties are determined by first occurrence in the original string list.
    huge=np.iinfo(np.int64).max//4
    best=np.min(np.where(admitted,scores,huge),axis=1)
    tied=admitted&(scores==best[:,None])
    a=np.argmin(np.where(tied,rank,999999),axis=1)
    a[~has]=np.where(ok[~has,5],5,6)
    return a,admitted

def guard_pattern(z,bits,pattern):
    n=len(z);gn=np.zeros(n,np.int64);gd=np.ones(n,np.int64);ga=np.ones(n,bool)
    reachable=np.ones(n,bool)
    for j in range(5):
        flip=bits[:,j]!=pattern[j]
        reachable &= ~(flip&(z[:,j]==0))
        pn=np.where(flip,np.abs(z[:,j]-THETA[j]),0);pd=np.full(n,THETA[j],np.int64)
        pa=np.ones(n,bool)
        # GE false / LT true requires theta > evidence (open boundary).
        open_atom=(not pattern[j]) if j<2 else bool(pattern[j])
        if open_atom:pa[flip]=False
        bigger=lt(gn,gd,pn,pd);same=eq(gn,gd,pn,pd)
        ga=np.where(bigger,pa,np.where(same,ga&pa,ga));gn=np.where(bigger,pn,gn);gd=np.where(bigger,pd,gd)
    gn[~reachable]=1;gd[~reachable]=0;ga[~reachable]=False
    return gn,gd,ga

def cost_crossings(effects,overhead,rank,winner):
    n=len(effects);ix=np.arange(n);safe=np.minimum(winner,5)
    A=effects[ix,safe];Ha=overhead[ix,safe];S=A[:,None]+effects
    gap=effects+overhead-(A+Ha)[:,None]
    earlier=rank<rank[ix,safe][:,None]
    cn=np.ones((n,6),np.int64);cd=np.zeros((n,6),np.int64);ca=np.zeros((n,6),bool)
    neg=gap<0;tie=gap==0
    zero=neg|(tie&(earlier|(S>0)))
    cn[zero]=0;cd[zero]=1;ca[zero]=neg[zero]|earlier[zero]
    first=(gap>0)&(S>0)&(gap<=S)
    # At epsilon=1 an A=0 winner cannot be beaten strictly if b only reaches a tie.
    impossible_plateau=first&(gap==S)&(A[:,None]==0)&~earlier
    first &= ~impossible_plateau
    cn[first]=gap[first];cd[first]=S[first];ca[first]=earlier[first]
    beyond=(gap>0)&~first&~impossible_plateau&(A[:,None]>0)
    # Valid beyond-one crossing follows H_b-H_a-A(1+epsilon)=0.
    numerator=overhead-Ha[:,None]-A[:,None]
    cn[beyond]=numerator[beyond];cd[beyond]=np.broadcast_to(A[:,None],cn.shape)[beyond];ca[beyond]=earlier[beyond]
    selfmask=np.arange(6)[None,:]==winner[:,None]
    cn[selfmask]=1;cd[selfmask]=0;ca[selfmask]=False
    return cn,cd,ca

def solve(q,flags,effects,overhead,exposed,rank,winner):
    n=len(q);ix=np.arange(n);safe=np.minimum(winner,5)
    z,bits=values_bits(q);cn,cd,ca=cost_crossings(effects,overhead,rank,winner)
    maxv=max(int(np.abs(z-THETA).max()),int(cn.max()),int(cd.max()),int(THETA.max()))
    assert maxv*maxv < np.iinfo(np.int64).max,('rational cross-product overflow',maxv)
    rn=np.ones(n,np.int64);rd=np.zeros(n,np.int64);ra=np.zeros(n,bool)
    pattern_index=np.full(n,-1,np.int16);challenger=np.full(n,-1,np.int8)
    winning_gn=np.ones(n,np.int64);winning_gd=np.zeros(n,np.int64)
    winning_cn=np.ones(n,np.int64);winning_cd=np.zeros(n,np.int64)
    for pi,pattern in enumerate(PATTERNS):
        gn,gd,ga=guard_pattern(z,bits,pattern)
        ok=admission(pattern,flags);S=ok&exposed;has=S.any(axis=1)
        contains=(winner<6)&S[ix,safe]
        empty_result=np.where(ok[:,5],5,6)
        direct=(has&~contains)|(~has&(empty_result!=winner))
        pc_n=np.ones(n,np.int64);pc_d=np.zeros(n,np.int64);pc_a=np.zeros(n,bool);pc_b=np.full(n,-1,np.int8)
        for b in range(6):
            valid=has&contains&S[:,b]&(winner!=b)
            lower=lt(cn[:,b],cd[:,b],pc_n,pc_d)
            same=eq(cn[:,b],cd[:,b],pc_n,pc_d)
            update=valid&(lower|(same&ca[:,b]&~pc_a))
            pc_n[update]=cn[update,b];pc_d[update]=cd[update,b];pc_a[update]=ca[update,b];pc_b[update]=b
        pc_n[direct]=0;pc_d[direct]=1;pc_a[direct]=True;pc_b[direct]=-1
        g_less=lt(gn,gd,pc_n,pc_d);c_less=lt(pc_n,pc_d,gn,gd)
        pn=np.where(g_less,pc_n,gn);pd=np.where(g_less,pc_d,gd)
        pa=(pd!=0)&(g_less|ga)&(c_less|pc_a)
        lower=lt(pn,pd,rn,rd);same=eq(pn,pd,rn,rd)
        update=lower|(same&pa&~ra)
        rn[update]=pn[update];rd[update]=pd[update];ra[update]=pa[update]
        pattern_index[update]=pi;challenger[update]=pc_b[update]
        winning_gn[update]=gn[update];winning_gd[update]=gd[update]
        winning_cn[update]=pc_n[update];winning_cd[update]=pc_d[update]
    gcd=np.gcd(rn,rd);rn//=gcd;rd//=gcd
    assert max(int(rn.max()),int(rd.max()))**2<np.iinfo(np.int64).max
    return {'num':rn,'den':rd,'attained':ra,'pattern':pattern_index,'challenger':challenger,
            'gnum':winning_gn,'gden':winning_gd,'cnum':winning_cn,'cden':winning_cd,
            'comparison_integer_bound':maxv,'cost_num':cn,'cost_den':cd,'cost_attained':ca}

def feasible_boundary_patterns(q,num,den):
    z,_=values_bits(q)
    lower=THETA[None,:]*np.maximum(den-num,0)[:,None]
    upper=THETA[None,:]*(den+num)[:,None]
    evidence=z*den[:,None]
    le=(z>0)&(lower<=evidence);gt=upper>evidence
    for pattern in PATTERNS:
        wanted_le=np.r_[pattern[:2],~pattern[2:]]
        yield pattern,np.where(wanted_le[None,:],le,gt).all(axis=1)

def boundary_flip_possible(q,flags,effects,overhead,exposed,rank,winner,num,den):
    n=len(q);ix=np.arange(n);safe=np.minimum(winner,5)
    # All score comparisons are exact integer cross-multiplications.
    bound=int(effects.max())*int((num+den).max())+int(overhead.max())*int(den.max())
    assert bound<np.iinfo(np.int64).max,('score numerator overflow',bound)
    original_max=effects[ix,safe]*(den+num)+overhead[ix,safe]*den
    rival_min=effects*np.maximum(den-num,0)[:,None]+overhead*den[:,None]
    earlier=rank<rank[ix,safe][:,None]
    beats=(rival_min<original_max[:,None])|((rival_min==original_max[:,None])&earlier)
    beats &= np.arange(6)[None,:]!=winner[:,None]
    found=np.zeros(n,bool)
    for pattern,feasible in feasible_boundary_patterns(q,num,den):
        ok=admission(pattern,flags);S=ok&exposed;has=S.any(axis=1);contains=(winner<6)&S[ix,safe]
        different=(has&~contains)|(~has&(np.where(ok[:,5],5,6)!=winner))|(has&contains&(beats&S).any(axis=1))
        found |= feasible&different&(den!=0)
    return found,bound

def scalar_admission(bits,flags):
    s,r,v,vr,e=bits;sf,rf,vf,ef=flags
    return [not(sf and s or rf and r),not(rf and r),True,not(rf and r or ef and e),not(rf and r),not(vf and v and vr)]

def scalar_cost_crossing(A,B,Ha,Hb,earlier):
    """Independent piecewise interval search over all algebraic breakpoints."""
    points={F(0),F(1)}
    if A+B:points.add(F(B+Hb-A-Ha,A+B))
    if A:points.add(F(Hb-Ha-A,A))
    points=sorted(x for x in points if x>=0)
    def accept(x):
        gap=B*max(F(0),1-x)+Hb-A*(1+x)-Ha
        return gap<0 or (gap==0 and earlier)
    for j,x in enumerate(points):
        if accept(x):return x,True
        nextx=points[j+1] if j+1<len(points) else x+2
        if accept((x+nextx)/2):return x,False
    return None,False

def scalar_solve(q,flags,effects,overhead,exposed,rank,winner):
    q=list(map(int,q));ef=list(map(int,effects));h=list(map(int,overhead));rank=list(map(int,rank))
    risk=2*q[0]+2*q[1]+2*q[2]+q[3];z=[risk,2*q[1],2*q[3],risk,2*q[2]]
    theta=list(map(int,THETA));base=[z[j]>=theta[j] if j<2 else z[j]<theta[j] for j in range(5)]
    best=None;attained=False
    for pattern in itertools.product([False,True],repeat=5):
        flips=[j for j in range(5) if pattern[j]!=base[j]]
        if any(z[j]==0 for j in flips):continue
        g=max([F(abs(z[j]-theta[j]),theta[j]) for j in flips]+[F(0)])
        ok=scalar_admission(pattern,list(map(bool,flags)))
        S=sorted([a for a in range(6) if exposed[a] and ok[a]],key=lambda a:rank[a])
        if not S:
            if (5 if ok[5] else 6)==winner:continue
            c=F(0)
        elif winner not in S:c=F(0)
        else:
            choices=[scalar_cost_crossing(ef[winner],ef[b],h[winner],h[b],rank[b]<rank[winner])[0] for b in S if b!=winner]
            choices=[v for v in choices if v is not None]
            if not choices:continue
            c=min(choices)
        radius=max(g,c)
        # Derive exact feasibility at this radius from the threshold interval,
        # rather than copying the vector solver's open-boundary flags.
        feasible=True
        for j in range(5):
            want_le=pattern[j] if j<2 else not pattern[j]
            low=max(F(0),theta[j]*(1-radius));high=theta[j]*(1+radius)
            feasible &= (z[j]>0 and low<=z[j]) if want_le else high>z[j]
        if not S:flip=(5 if ok[5] else 6)!=winner
        elif winner not in S:flip=True
        else:
            higha=ef[winner]*(1+radius)+h[winner]
            flip=any(ef[b]*max(F(0),1-radius)+h[b]<higha or (ef[b]*max(F(0),1-radius)+h[b]==higha and rank[b]<rank[winner]) for b in S if b!=winner)
        att=bool(feasible and flip)
        if best is None or radius<best:best=radius;attained=att
        elif radius==best:attained |= att
    return best,attained

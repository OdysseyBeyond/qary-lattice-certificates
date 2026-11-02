#!/usr/bin/env python3
"""Rigorous lower certificate for the fixed-parameter SVP time objective.

This is a branch-and-bound certificate, not a numerical optimizer.  Float
calculations decide subdivisions only; every discarded box receives an
outward-rounded rational interval lower bound.  A successful run covers all
fixed recursion depths.  The accompanying analytical reduction is required.
"""
import argparse, heapq, json, math, sys, time
from fractions import Fraction as Q
from pathlib import Path
from functools import lru_cache
import global_parameter_certificate as cvp
import svp_parameter_selection as svp
F=cvp.FixedInterval; SCALE=cvp.SCALE
C=Q('.11201367')
OUTER_X=(Q('.144478403'),)

def h(x):
    x=Q(x)
    if x==0:return F(0)
    if x==Q(1,2):return F(1)
    return cvp.fixed_entropy(x)

def hf(x):
    if x<=0:return 0.
    if x>=.5:return 1.
    return -2*x*math.log2(x)-(1-2*x)*math.log2(1-2*x)

def r(p,x):
    p,x=Q(p),Q(x)
    if p==Q(1,2):return F(1) # feasible x=1/4
    z=x-p/2;v=1-p-2*x;u=1-2*p
    assert z>=0 and v>=0 and u>0,(p,x,z,v)
    def xl(a):return F(0) if a==0 else F(a)*cvp.fixed_log2(a)
    return 2*F(p)-2*xl(z)-xl(v)+xl(u)

def rf(p,x):
    z=x-p/2;v=1-p-2*x;u=1-2*p
    if min(z,v,u)<-1e-12:return -1e6
    def xl(a):return a*math.log2(a) if a>0 else 0.
    return 2*p-2*xl(z)-xl(v)+xl(u)

def ell(p,x):return h(x)-r(p,x)
def clip(x,l,u):return max(l,min(u,x))
def min_h(l,u):return min(h(l).lo,h(u).lo)
def min_ell(p,l,u):
    l=max(l,p/2);u=min(u,(1-p)/2)
    if l>u:return None
    return ell(p,clip(Q(1,4),l,u)).lo

def max_r(p,l,u):
    l=max(l,p/2);u=min(u,(1-p)/2)
    if l>u:return None
    return r(p,clip((2-p)/6,l,u)).hi

def psi(p,budget):
    # The elementary ell>=p bound is valid also at endpoint and budget cases.
    base=F(p).lo
    if p<=0 or p>=Q(1,2) or budget<=0 or budget>=h(Q(1,4)).lo:return base
    try:return max(base,cvp.certified_last_merge(F(p),F(budget,budget,raw=True)))
    except AssertionError:return base

def psi_float(p,b):
    if p<=0:return 0.
    if p>=.5:return p
    if b>=1.5:return p
    return cvp.float_last_merge(p,b)

def outer_data():
    data=[]
    for x in OUTER_X:
        q=svp.law_bounds(x)
        data.append(tuple(F(v.lo,v.hi) for v in (q['density'],q['w'],q['large_entropy'])))
    # The last sample is genuinely inside the density range and certifies C.
    assert data[-1][0].hi<SCALE
    assert data[-1][0].lo>F('.99').hi
    assert (data[-1][1]/2/data[-1][0]).lo>F(C).hi
    return data
DATA=outer_data()

def subset_lower(rho,D,W):
    # alpha=D/rho-2 C D.  The cost decreases with alpha.  At fixed alpha
    # it is concave in w until the zero plateau; its minimum on an interval
    # is attained at an endpoint (or is zero if the plateau is reached).
    A=D*(1/F(rho)-2*C)
    if A.hi+W.hi>=SCALE:return 0
    def at(w):
        a=F(A.hi,A.hi,raw=True)
        u=F(w,w,raw=True)
        return (a*cvp.fixed_log2(a)-(a+u)*cvp.fixed_log2(a+u)
                -(1-u)*cvp.fixed_log2(1-u)).lo
    return max(0,min(at(W.lo),at(W.hi)))

def normalized_outer(lower_t,rl,ru):
    return max(R.lo+subset_lower(rl,D,W)
               +(F(lower_t,lower_t,raw=True)*D/F(ru)).lo for D,W,R in DATA)

def bound(box,depth):
    rl,ru,xl,xu,*tail=box
    al=C*rl
    xl=max(xl,al/2);xu=min(xu,(1-al)/2)
    if xl>xu:return None
    e1=min_ell(al,xl,xu)
    c1=e1+max(0,min_h(xl,xu)-F(ru).hi)
    base=max(c1,(h(C*rl)-F(ru)).lo)
    if depth==1:
        return normalized_outer(max(base,min_h(xl,xu)//2),rl,ru)
    yl,yu=tail
    yl=max(yl,xl/2);yu=min(yu,(1-xl)/2)
    if yl>yu:return None
    r1=max_r(al,xl,xu)
    c2=min_ell(xl,yl,yu)+max(0,min_h(yl,yu)-r1)
    base=max(base,c2)
    if depth==2:base=max(base,min_h(yl,yu)//2)
    else:base=max(base,psi(yl,max_r(xl,yl,yu)))
    return normalized_outer(base,rl,ru)

def float_bound(box,depth):
    rl,ru,xl,xu,*tail=map(float,box);c=float(C);al=c*rl
    xl=max(xl,al/2);xu=min(xu,(1-al)/2)
    if xl>xu:return 9.
    def hm(l,u):return min(hf(l),hf(u))
    def em(p,l,u):return hf(clip(.25,l,u))-rf(p,clip(.25,l,u))
    def rm(p,l,u):return rf(p,clip((2-p)/6,l,u))
    b=max(em(al,xl,xu)+max(0,hm(xl,xu)-ru),hf(al)-ru)
    if depth==1:b=max(b,hm(xl,xu)/2)
    else:
        yl,yu=tail;yl=max(yl,xl/2);yu=min(yu,(1-xl)/2)
        if yl>yu:return 9.
        b=max(b,em(xl,yl,yu)+max(0,hm(yl,yu)-rm(al,xl,xu)))
        b=max(b,hm(yl,yu)/2 if depth==2 else psi_float(yl,rm(xl,yl,yu)))
    result=0
    for D,W,R in DATA:
        d,w,rr=D.lo/SCALE,W.lo/SCALE,R.lo/SCALE
        a=d/rl-2*c*d
        g=0 if a+w>=1 else (a*math.log2(a)-(a+w)*math.log2(a+w)-(1-w)*math.log2(1-w))
        result=max(result,rr+g+b*d/ru)
    return result

def split(box,depth):
    # Coordinate widths are normalized to their typical sensitive scales.
    scales=(Q('.5'),Q('.2'),Q('.1'))
    j=max(range(len(box)//2),key=lambda j:(box[2*j+1]-box[2*j])/scales[j])
    l,u=box[2*j:2*j+2];mid=(l+u)/2
    a=list(box);b=list(box);a[2*j+1]=mid;b[2*j]=mid
    return tuple(a),tuple(b)


LOCAL_CENTER=tuple(map(Q,('.09796686677105','.06606292369075','.03479412735314')))
LOCAL_RAD=Q('.001')
LOCAL_RHO=(Q('.995'),Q('1.005'))
LOCAL_XY_RAD=Q('.0004')

def asq(raw):return Q(raw,SCALE)
def fabs(v):return max(abs(v.lo),abs(v.hi))
def point(raw):return F(raw,raw,raw=True)

def local_certificate(threshold):
    D,W,R=DATA[-1]; dl=asq(D.lo);rl=asq(R.lo)
    b=1-W+2*C*D
    cusp=D/b; cusp_upper=asq(cusp.hi)
    assert LOCAL_RHO[0]<cusp_upper<LOCAL_RHO[1]
    hinge=-cvp.fixed_log2(1-W)*b; hinge_lower=asq(hinge.lo)
    lam=F('.417633157509395');v=F('.515519165806019');u=1-lam-v
    assert min(lam.lo,u.lo,v.lo)>0
    x,y,z=[F(a-LOCAL_RAD,a+LOCAL_RAD) for a in LOCAL_CENTER]
    rp=F(LOCAL_RHO[0],LOCAL_RHO[1]);a=C*rp
    root_parent=2+cvp.fixed_log2((x-a/2)*(1-a-2*x)/((1-2*a)*(1-2*a)))
    derivative=-dl*lam*C*root_parent+rl-threshold
    assert derivative.hi<0 and (derivative+hinge_lower).lo>0
    # The supported hinge has its minimum at cusp_upper throughout the box.
    root=C*cusp_upper
    def representations(p,ch):
        added=ch-p/2;remaining=1-p-2*ch;zero=1-2*p
        return 2*p-2*added*cvp.fixed_log2(added)-remaining*cvp.fixed_log2(remaining)+zero*cvp.fixed_log2(zero)
    def ent(t):return -2*t*cvp.fixed_log2(t)-(1-2*t)*cvp.fixed_log2(1-2*t)
    def derivative_h(t):return 2*cvp.fixed_log2((1-2*t)/t)
    def derivative_r(p,ch):
        ad=ch-p/2;re=1-p-2*ch;ze=1-2*p
        return 2+cvp.fixed_log2(ad*re/(ze*ze)),2*cvp.fixed_log2(re/ad)
    def value(xx,yy,zz,root_value=None):
        if root_value is None:root_value=F(root)
        return 2*lam*ent(yy)+(u+2*v)*ent(zz)-lam*representations(root_value,xx)-(lam+v)*representations(xx,yy)-(u+v)*representations(yy,zz)
    assert value(x,y,z,C*rp).lo>0
    z1=x-root/2;v1=1-root-2*x
    z2=y-x/2;v2=1-x-2*y;u2=1-2*x
    z3=z-y/2;v3=1-y-2*z;u3=1-2*y
    coefficients=[lam*(2/z1+4/v1)-4*(lam+v)/u2,
                  -4*lam/(y*(1-2*y))-4*(1-lam)/u3,
                  -2*(u+2*v)/(z*(1-2*z)),
                  2*(lam+v)/z2,(lam+v)/v2,2*(1-lam)/z3,(1-lam)/v3]
    coarse=[Q(f.lo*10000//SCALE,10000) for f in coefficients]
    ca,cc0,ce,ca2,cb2,ca3,cb3=coarse
    ca+=ca2/4+cb2;cb=-ca2/2+2*cb2
    cc0+=ca2+4*cb2+ca3/4+cb3;cd=-ca3/2+2*cb3;ce+=ca3+4*cb3
    cp2=cc0-cb*cb/ca;cp3=ce-cd*cd/cp2
    assert ca>0 and cp2>0 and cp3>0
    aa,cc,ee,a2,b2,a3,b3=map(lambda f:asq(f.lo),coefficients)
    aa+=a2/4+b2;bb=-a2/2+2*b2;cc+=a2+4*b2+a3/4+b3;dd=-a3/2+2*b3;ee+=a3+4*b3
    pp2=cc-bb*bb/aa;pp3=ee-dd*dd/pp2
    assert aa>0 and pp2>0 and pp3>0,(float(aa),float(pp2),float(pp3))
    xp,yp,zp=map(F,LOCAL_CENTER)
    a1,b1=derivative_r(F(root),xp);a2,b2=derivative_r(xp,yp);a3,b3=derivative_r(yp,zp)
    gradient=(-lam*b1-(lam+v)*a2,
              2*lam*derivative_h(yp)-(lam+v)*b2-(u+v)*a3,
              (u+2*v)*derivative_h(zp)-(u+v)*b3)
    v0=value(xp,yp,zp)
    tangent=asq(v0.lo)-LOCAL_RAD*sum(asq(fabs(g)) for g in gradient)
    final=dl*tangent+cusp_upper*(rl-threshold)
    assert final>0, float(final)
    # Every psi minimizer for the confined x,y range lies in this z box.
    xl,xu=LOCAL_CENTER[0]-LOCAL_XY_RAD,LOCAL_CENTER[0]+LOCAL_XY_RAD
    yl,yu=LOCAL_CENTER[1]-LOCAL_XY_RAD,LOCAL_CENTER[1]+LOCAL_XY_RAD
    zl,zu=LOCAL_CENTER[2]-LOCAL_RAD,LOCAL_CENTER[2]+LOCAL_RAD
    assert yu<(2-xu)/6
    rrlo=r(xu,yl).lo;rrhi=r(xl,yu).hi
    assert h(zl).hi<rrlo and h(zu).lo>rrhi
    assert cvp.stationary_polynomial(F(yl,yu),F(zl)).hi<0
    assert cvp.stationary_polynomial(F(yl,yu),F(zu)).lo>0
    return dict(cusp_upper=str(cusp_upper),hinge_slope_lower=str(hinge_lower),
                coefficient_lower_bounds=list(map(str,coarse)),
                tridiagonal_lower_matrix=list(map(str,(ca,cb,cc0,cd,ce))),
                rounded_matrix_pivots=list(map(str,(ca,cp2,cp3))),
                gradient_bounds=[[str(asq(g.lo)),str(asq(g.hi))] for g in gradient],
                center_value_bounds=[str(asq(v0.lo)),str(asq(v0.hi))],
                rho_derivative=[str(asq(derivative.lo)),str(asq(derivative.hi))],
                positive_pivots=[str(aa),str(pp2),str(pp3)],
                tangent_lower=str(tangent),positive_gap=str(final),
                weights=[str(asq(a.lo)) for a in (lam,u,v)],
                center=list(map(str,LOCAL_CENTER)),radius=str(LOCAL_RAD),
                confined_rho=list(map(str,LOCAL_RHO)),confined_xy_radius=str(LOCAL_XY_RAD))

def local_contains(box,depth):
    if depth!=3:return False
    rl,ru,xl,xu,yl,yu=box
    return (LOCAL_RHO[0]<=rl<=ru<=LOCAL_RHO[1]
            and LOCAL_CENTER[0]-LOCAL_XY_RAD<=xl<=xu<=LOCAL_CENTER[0]+LOCAL_XY_RAD
            and LOCAL_CENTER[1]-LOCAL_XY_RAD<=yl<=yu<=LOCAL_CENTER[1]+LOCAL_XY_RAD)


def run(threshold,max_boxes,output):
    threshold=Q(threshold);target=F(threshold).hi
    assert threshold<Q('.278')
    # Root parity gives rho<2.  For rho<=.4, h(C rho)/rho-1
    # decreases in rho and excludes the entire interval by this check.
    small=h(C*Q('.4'))/Q('.4')-1
    assert max((F(small.lo,small.lo,raw=True)*D+R).lo for D,W,R in DATA)>target
    assert all(D.lo>2*target for D,W,R in DATA) # d<2 threshold; conservative
    local=local_certificate(threshold)
    local_leaves=0
    initial={1:(Q('.4'),Q(2),Q(0),Q('.5')),
             2:(Q('.4'),Q(2),Q(0),Q('.5'),Q(0),Q('.5')),
             3:(Q('.4'),Q(2),Q(0),Q('.5'),Q(0),Q('.5'))}
    queue=[];counter=0;seen=0;leaves={1:0,2:0,3:0};infeasible=0;start=time.time();lowest=2*SCALE
    for dep,b in initial.items():
        heapq.heappush(queue,(float_bound(b,dep),counter,dep,b));counter+=1
    while queue and seen<max_boxes:
        _,_,dep,box=heapq.heappop(queue);seen+=1
        if local_contains(box,dep):
            local_leaves+=1
            continue
        lb=bound(box,dep)
        if lb is None:infeasible+=1
        elif lb>target:
            leaves[dep]+=1;lowest=min(lowest,lb)
        else:
            a,b=split(box,dep)
            for ch in (a,b):
                heapq.heappush(queue,(float_bound(ch,dep),counter,dep,ch));counter+=1
        if seen%1000==0:
            print(f'boxes {seen} pending {len(queue)} leaves {leaves} elapsed {time.time()-start:.1f}s; min proposal {queue[0][0] if queue else None}',flush=True)
    if not queue:
        assert 2*(sum(leaves.values())+local_leaves+infeasible)==seen+3
        assert counter==seen
    result=dict(success=not queue,threshold=str(threshold),boxes=seen,created_nodes=counter,pending=len(queue),leaves=leaves,local_leaves=local_leaves,local_certificate=local,infeasible=infeasible,seconds=time.time()-start,
                scope='All fixed recursion depths and all admissible parameter choices at the certified density; consequently this also bounds policies that choose parameters separately at each density. Requires the analytical reduction.',
                lowest_certified=str(Q(lowest,SCALE)),next_box=([str(x) for x in queue[0][3]] if queue else None),
                next_depth=(queue[0][2] if queue else None),next_bound=(queue[0][0] if queue else None))
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--threshold',default='.2740094');p.add_argument('--max-boxes',type=int,default=1000000);p.add_argument('--output',default='/private/tmp/svp-global-lower.json');a=p.parse_args()
    run(a.threshold,a.max_boxes,a.output)

  

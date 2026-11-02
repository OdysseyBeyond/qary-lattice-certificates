#!/usr/bin/env python3
"""Outward integer-interval certificate for a complete SVP parameter witness.

This certifies feasibility and the full density supremum, not global minimality.
It uses the fixed-point interval implementation from the independent CVP
certificate.  Adaptive bisection covers a rational x=exp(-a) interval containing
all input densities.  A mean-value bound, including one-sided derivatives at
the zero subset-cost boundary, certifies each leaf.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import time
from functools import lru_cache

from global_parameter_certificate import (
    SCALE, FixedInterval as I, fixed as iv, fixed_max as imax,
    fixed_log as ln, fixed_log2 as log2, fixed_entropy as ternary,
    fixed_representations as representations,
)


def ipow(x,n):
    out=I(1)
    while n:
        if n&1:out=out*x
        n//=2
        if n:x=x*x
    return out


def positive(x):return imax(0,x)
def maximum(v):
    v=list(v)
    return I(max(x.lo for x in v),max(x.hi for x in v),raw=True)
def hull(a,b):return I(min(a.lo,b.lo),max(a.hi,b.hi),raw=True)
def upper_abs(x):return Fraction(max(abs(x.lo),abs(x.hi)),SCALE)
def xlog(x):
    if x.lo==x.hi==0:return I(0)
    assert x.lo>0
    return x*log2(x)
def entropy(p):return -sum((xlog(iv(x)) for x in p),I(0))
def out(x):
    if isinstance(x,I):return {'lo':str(Fraction(x.lo,SCALE)), 'hi':str(Fraction(x.hi,SCALE)),
                              'lo_decimal':float(Fraction(x.lo,SCALE)), 'hi_decimal':float(Fraction(x.hi,SCALE))}
    if isinstance(x,Fraction):return str(x)
    if isinstance(x,dict):return {k:out(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [out(v) for v in x]
    return x


def law(x):
    """Theta law, merged entropy, and derivatives with respect to log x.

    All sums include |j|<=5.  For |j|>=6 the successive-term ratios
    are bounded by x^13, (49/36)x^13, and (2401/1296)x^13 for the
    zeroth, second, and fourth moments, respectively.
    """
    x=iv(x)
    assert 0<x.lo<=x.hi<SCALE//4
    power=[ipow(x,j*j) for j in range(2,6)]
    small=[2*sum((j**r*p for j,p in zip(range(2,6),power)),I(0)) for r in (0,2,4)]
    tail=[2*(6**r)*ipow(x,36)/(1-Fraction(7**r,6**r)*ipow(x,13)) for r in (0,2,4)]
    sums=[I(s.lo,(s+t).hi,raw=True) for s,t in zip(small,tail)]
    z=1+2*x+sums[0]
    p,mt,nt=[s/z for s in sums]
    u=2*x/z
    mu=u+mt
    fourth=u+nt
    lx=ln(x)
    h=(ln(z)-mu*lx)/ln(I(2))
    w=(-(1-p)*ln(1-p)+p*ln(z)-mt*lx)/ln(I(2))
    dh=-lx*(fourth-mu*mu)/ln(I(2))
    du=u*(1-mu)
    dw=((mt-mu*p)*ln(1+2*x)-lx*(nt-mu*mt))/ln(I(2))
    return dict(h=h,w=w,u=u,dh=dh,dw=dw,du=du)


def subset(b,u):
    """Exact-argument interval evaluation; callers use point law enclosures."""
    if (b+u).lo>=SCALE:return I(0)
    value=xlog(b)-xlog(b+u)-xlog(1-u)
    if (b+u).hi<SCALE:return value
    return hull(I(0),value)


def inner(t,rho,eta):
    t=list(map(iv,t));rho=iv(rho);eta=iv(eta)
    h=list(map(ternary,t));r=[representations(x,y) for x,y in zip(t,t[1:])]
    d=[rho]+[x-eta for x in r]
    margins={'rho':rho,'eta':eta,'last_equations':d[-1],
             'root_parity':entropy((2*t[0],1-2*t[0]))-rho/2}
    for i,x in enumerate(t):
        margins[f'coordinate_positive_{i}']=x
        margins[f'coordinate_below_half_{i}']=I(Fraction(1,2))-x
    for i,(x,y) in enumerate(zip(t,t[1:]),1):
        margins[f'added_sign_{i}']=y-x/2
        margins[f'zero_coordinates_{i}']=1-x-2*y
        margins[f'equation_gap_{i}']=d[i-1]-d[i]
        margins[f'list_parity_{i}']=entropy((2*y,1-2*y))-d[i-1]/2
        margins[f'representation_parity_{i}']=r[i-1]-d[i]/2-2*y+x
    assert all(x.lo>0 for x in margins.values()), [(k,out(v)) for k,v in margins.items() if v.lo<=0]
    ell=[positive(x-y) for x,y in zip(h,d)]
    time_terms=[h[-1]/2]+[ell[i]+positive(h[i]-d[i-1]) for i in range(1,len(t))]
    space_terms=[h[-1]/2]+ell[:-1]+[positive(h[-1]-d[-2])]
    return dict(time=maximum(time_terms),space=maximum(space_terms),
                time_terms=time_terms,space_terms=space_terms,margins=margins,h=h,r=r)


def bound_leaf(left,right,alpha,beta,epsilon):
    center=(left+right)/2
    mid=law(I(center));wide=law(I(left,right))
    bm=alpha*mid['h'];um=mid['u']
    center_value=mid['w']+subset(bm,um)+beta*mid['h']
    b=alpha*wide['h'];u=wide['u']
    # Derivative with respect to log x.  At b+u=1, enclose both
    # the active-side derivative and the plateau derivative zero.
    if (b+u).lo>=SCALE:
        dg=I(0)
    else:
        dg=log2(b/(b+u))*alpha*wide['dh']+log2((1-u)/(b+u))*wide['du']
        if (b+u).hi>=SCALE:dg=hull(dg,I(0))
    derivative=wide['dw']+dg+beta*wide['dh']
    # dx derivative is (d/d log x)/x.  Using a centered mean-value
    # enclosure gives a quadratic excess near a stationary maximum.
    dx=derivative/I(left,right)
    meanvalue_upper=Fraction(center_value.hi,SCALE)+upper_abs(dx)*(right-left)/2
    # Count tolerance: g is Lipschitz in its second argument.  This
    # also covers the nondifferentiable zero-cost boundary.
    expanded=I(Fraction(u.lo,SCALE)-2*epsilon,Fraction(u.hi,SCALE)+2*epsilon)
    assert expanded.lo>0 and expanded.hi<SCALE
    lipschitz=upper_abs(log2((1-expanded)/(b+expanded)))
    tolerance=epsilon*(1+2*lipschitz)
    # A direct interval bound is often sharper away from stationary
    # maxima, and requires no differentiability assumptions.
    direct=wide['w']+subset(b,u)+beta*wide['h']
    ub=min(meanvalue_upper,Fraction(direct.hi,SCALE))+tolerance
    return ub,Fraction(center_value.lo,SCALE),center


def certify(config):
    start=time.monotonic()
    rho,t=Fraction(config['rho']),list(map(Fraction,config['t']))
    eta,eps,cutoff=map(Fraction,(config['eta'],config['epsilon'],config['cutoff']))
    target=Fraction(config['threshold'])
    densitylo=Fraction(config.get('densitylo',cutoff))
    densityhi=Fraction(config.get('densityhi',1))
    if 'xlo' in config and 'xhi' in config:
        left,right=map(Fraction,(config['xlo'],config['xhi']))
    else:
        left=density_bracket(densitylo)[0]
        right=density_bracket(densityhi)[1]
    assert Fraction(1,2)<cutoff<1 and 0<eps and eta>0 and cutoff/2<target
    inside=inner(t,rho,eta)
    first,last=law(I(left)),law(I(right))
    assert first['h'].hi<=int(densitylo*SCALE)
    assert last['h'].lo>=I(densityhi).hi, out(last['h'])
    # The analytic monotonicity proof of p_x(1)/H(p_x) for x<=1/4
    # reduces every density's padding constraint to this endpoint.
    # For 0<x<=1/4, write A=2 sum_{j>=2}(j^2-1)x^{j^2}
    # and B=2 sum_{j>=2}j^2(j^2-1)x^{j^2}.  The sign of
    # d[p_x(1)/H(p_x)]/dx is the sign of
    # (1-A) log Z - B log(1/x).  Successive-term bounds give
    # A<1/16, B<25x^4.  Moreover log Z>=4x/3 and
    # x^3 log(1/x)<=log(4)/64<1/32.  The sign is therefore
    # greater than 5x/4-25x/32=15x/32>0.
    quarter=Fraction(1,4)
    assert 8*quarter**4/(1-Fraction(9,4)*quarter**5)<Fraction(1,16)
    assert 24/(1-6*quarter**5)<25
    assert ln(I(4)).hi<2*SCALE
    padding=I(t[0]/rho)-last['u']/(2*last['h'])-eps/densitylo
    assert padding.lo>0,out(padding)
    nu=int(config.get('nu',16))
    assert nu**3*(cutoff/rho)**2>1
    alpha=I((1-2*t[0])/rho)
    beta=inside['time']/rho
    stack=[(left,right,0)]
    leaves=[];visited=0;best_lower=Fraction(0);best_x=None
    while stack:
        lo,hi,depth=stack.pop()
        upper,lower,x=bound_leaf(lo,hi,alpha,beta,eps)
        visited+=1
        if lower>best_lower:best_lower,best_x=lower,x
        assert lower<target, f'Threshold below a certified point value: {float(lower):.12f}, x={float(x):.12f}'
        if upper<target:
            leaves.append((lo,hi,upper))
        else:
            assert depth<70,'Subdivision failed to converge'
            middle=(lo+hi)/2
            stack.extend(((middle,hi,depth+1),(lo,middle,depth+1)))
        if visited%500==0:
            print(f'Checked {visited} intervals; {len(leaves)} accepted, {len(stack)} pending; max point {float(best_lower):.12f}',flush=True)
    # Independent exact coverage check: no gaps, overlaps, or omitted leaf.
    leaves.sort()
    assert leaves[0][0]==left and leaves[-1][1]==right
    assert all(a[1]==b[0] for a,b in zip(leaves,leaves[1:]))
    uniform_upper=max(v[2] for v in leaves)
    time_upper=max(cutoff/2,uniform_upper)
    space=maximum((I(cutoff/4),inside['space']*densityhi/rho))
    return out(dict(scope='A rigorous feasible upper bound for the complete SVP time exponent; no global lower bound is asserted by this file.',
        parameters=config,inner=inside,padding_margin=padding,
        density_endpoints=(first['h'],last['h']),
        time_upper=time_upper,time_upper_decimal=float(time_upper),space_upper=space,
        largest_certified_point=best_lower,largest_point_x=best_x,
        partition=dict(visited=visited,leaves=len(leaves),interval=(left,right),
                       exact_coverage_verified=True,method='adaptive interval mean-value and direct bounds'),
        seconds=time.monotonic()-start))


@lru_cache(maxsize=1000)
def density_bracket(density):
    """Certified rational inverse-entropy bracket, using exact comparisons."""
    density=Fraction(density)
    assert Fraction(1,2)<=density<=1
    left,right=Fraction('.04'),Fraction('.16')
    assert law(I(left))['h'].hi<I(density).lo
    assert law(I(right))['h'].lo>I(density).hi
    for _ in range(75):
        middle=(left+right)/2
        h=law(I(middle))['h']
        if h.hi<I(density).lo:left=middle
        elif h.lo>I(density).hi:right=middle
        else:raise ArithmeticError('The inverse entropy needs more interval precision')
    return left,right


def certify_policy(config):
    """Certify finitely many rational tuples covering all input densities."""
    start=time.monotonic()
    cutoff=Fraction(config['cutoff']);threshold=Fraction(config['threshold'])
    pieces=config['pieces']
    assert Fraction(pieces[0]['densitylo'])==cutoff
    assert Fraction(pieces[-1]['densityhi'])==1
    assert all(Fraction(a['densityhi'])==Fraction(b['densitylo']) for a,b in zip(pieces,pieces[1:]))
    certificates=[]
    base={k:v for k,v in config.items() if k!='pieces'}
    for i,piece in enumerate(pieces):
        item=dict(base,**piece)
        assert Fraction(item['densitylo'])<Fraction(item['densityhi'])
        result=certify(item)
        certificates.append(result)
        print(f'Certified policy interval {i+1}/{len(pieces)}: densities [{item["densitylo"]}, {item["densityhi"]}], time < {result["time_upper_decimal"]:.12f}',flush=True)
    upper=max(Fraction(c['time_upper']) for c in certificates)
    space=max(Fraction(c['space_upper']['hi']) for c in certificates)
    padding=min(Fraction(c['padding_margin']['lo']) for c in certificates)
    assert upper<threshold
    return out(dict(scope='A rigorous feasible upper bound using a finite list of density-dependent rational SVP parameter choices; no global lower bound is asserted by this file.',
        parameters=config,time_upper=upper,time_upper_decimal=float(upper),
        space_upper=space,space_upper_decimal=float(space),padding_margin=padding,
        policy_intervals=len(pieces),exact_density_coverage_verified=True,
        certificates=certificates,seconds=time.monotonic()-start))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,default=Path(__file__).with_name('svp_global_candidate.json'))
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('svp_global_upper.json'))
    args=p.parse_args()
    config=json.loads(args.candidate.read_text())
    result=certify_policy(config) if 'pieces' in config else certify(config)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('time_upper','space_upper','padding_margin','seconds')},indent=2))

if __name__=='__main__':main()

  

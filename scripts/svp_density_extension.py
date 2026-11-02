#!/usr/bin/env python3
"""Outward rational interval certificates for SVP entropy densities above one.

The existing fixed-point interval arithmetic and representation constraints are
reused without modification.  The theta enclosure here is independent of the
old x<1/4 restriction: it is valid for 0<x<=3/4.  Every parameter profile is
checked on an entire inverse-entropy interval, including its endpoints.
This is a feasible upper-bound certificate, not an optimality certificate.
"""
import argparse
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import time

from svp_global_upper import SCALE, I, ipow, inner, out, ln

GAMMA=Fraction(99999999,100000000)
OFFSET=Fraction(1,10**9)
ROUNDING=Fraction(1,10**12)
ETA=EPSILON=Fraction(1,10**10)
NU=16
PROFILES=[
    dict(densityhi='1.05',time='.286',space='.270',
         ratios=['1','.8939052281','.6177516046','.3386702762']),
    dict(densityhi='1.10',time='.298',space='.276',
         ratios=['1','.8678977552','.5915218016','.3209603588']),
    dict(densityhi='1.20',time='.326',space='.286',
         ratios=['1','.8656984966','.5982906774','.3350929465']),
    dict(densityhi='1.30',time='.361',space='.309',
         ratios=['1','.8501069734','.5873979844','.3331341199']),
    dict(densityhi='1.40',time='.408',space='.320',
         ratios=['1','.8191452039','.5562355082','.3124046192']),
    dict(densityhi='1.50',time='.468',space='.329',
         ratios=['1','.8022369518','.5405699407','.3037290770']),
]


def law(x):
    """Enclose theta-law and merged entropies for 0<x<=3/4.

    Terms |j|<=16 are included.  For j>=17, the ratio of consecutive
    j^r*x^(j*j) terms is at most (18/17)^r*x^35 (r=0,2), since both
    (1+1/j)^r and x^(2*j+1) are nonincreasing.  A geometric series
    therefore encloses each nonnegative tail.  No floating point is used.
    """
    assert 0<x.lo<=x.hi<=3*SCALE//4
    powers=[ipow(x,j*j) for j in range(2,17)]
    finite=[2*sum((j**r*p for j,p in zip(range(2,17),powers)),I(0))
            for r in (0,2)]
    tails=[2*17**r*ipow(x,289)/(1-Fraction(18**r,17**r)*ipow(x,35))
           for r in (0,2)]
    sums=[I(s.lo,(s+t).hi,raw=True) for s,t in zip(finite,tails)]
    z=1+2*x+sums[0]
    p,mt=[s/z for s in sums]
    u=2*x/z
    mu=u+mt
    lx=ln(x)
    h=(ln(z)-mu*lx)/ln(I(2))
    w=(-(1-p)*ln(1-p)+p*ln(z)-mt*lx)/ln(I(2))
    return dict(h=h,w=w,u=u,mu=mu)


@lru_cache(maxsize=100)
def density_bracket(density):
    """Certified inverse of H(p_x), which increases strictly with x.

    Its derivative with respect to log x equals
    -log(x)*Var_p(j^2)/log(2)>0 on (0,1).
    """
    density=Fraction(density)
    assert 1<=density<=2
    left,right=Fraction('.1'),Fraction('.7')
    assert law(I(left))['h'].hi<I(density).lo
    assert law(I(right))['h'].lo>I(density).hi
    for _ in range(75):
        middle=(left+right)/2
        h=law(I(middle))['h']
        if h.hi<I(density).lo:left=middle
        elif h.lo>I(density).hi:right=middle
        else:raise ArithmeticError('More interval precision required')
    return left,right


def certify(profile):
    start=time.monotonic()
    densityhi=Fraction(profile['densityhi'])
    threshold,space_target=map(Fraction,(profile['time'],profile['space']))
    ratios=tuple(map(Fraction,profile['ratios']))
    assert 1<=densityhi<=Fraction(3,2)
    assert len(ratios)==4 and ratios[0]==1, 'The padding proof requires c0=1 and this certificate uses three decomposition steps'
    left=density_bracket(1)[0]
    right=density_bracket(densityhi)[1]
    first,last=law(I(left)),law(I(right))
    assert first['h'].hi<I(1).lo
    assert last['h'].lo>I(densityhi).hi
    # mu increases with log x by Var_p(j^2)>0.  mu<1 implies
    # d p_x(1)/d log x=p_x(1)*(1-mu)>0 on this entire interval.
    assert last['mu'].hi<SCALE
    pmax=last['u']/2
    retain=(1-GAMMA)*(1-2*pmax)/GAMMA-2*(OFFSET+ROUNDING)/GAMMA-2*EPSILON
    assert retain.lo>0
    padding=OFFSET/GAMMA-EPSILON
    assert padding>0
    assert NU**3*(1/GAMMA)**2>1
    stack=[(left,right,0)]
    leaves=[];visited=0
    min_margins={};worst_time=Fraction(0);worst_space=Fraction(0)
    while stack:
        lo,hi,depth=stack.pop();visited+=1
        distribution=law(I(lo,hi))
        t0=distribution['u']/2+I(OFFSET,OFFSET+ROUNDING)
        try:
            rep=inner([c*t0 for c in ratios],GAMMA*distribution['h'],ETA)
            cost=distribution['w']+EPSILON+rep['time']/GAMMA
            space=rep['space']/GAMMA
            accepted=cost.hi<I(threshold).lo and space.hi<I(space_target).lo
        except AssertionError:
            accepted=False
        if accepted:
            upper=Fraction(cost.hi,SCALE);sp=Fraction(space.hi,SCALE)
            worst_time=max(worst_time,upper);worst_space=max(worst_space,sp)
            for key,value in rep['margins'].items():
                min_margins[key]=min(min_margins.get(key,Fraction(100)),Fraction(value.lo,SCALE))
            leaves.append((lo,hi,upper,sp))
        else:
            assert depth<70, f'Certificate does not converge at x={float(lo)}..{float(hi)}'
            middle=(lo+hi)/2
            stack.extend(((middle,hi,depth+1),(lo,middle,depth+1)))
        if visited%500==0:
            print(f'density<={densityhi}: checked {visited} intervals; {len(leaves)} accepted',flush=True)
    leaves.sort()
    assert leaves[0][0]==left and leaves[-1][1]==right
    assert all(a[1]==b[0] for a,b in zip(leaves,leaves[1:]))
    endpoint_law=law(I(*density_bracket(densityhi)))
    endpoint_t0=endpoint_law['u']/2+I(OFFSET,OFFSET+ROUNDING)
    endpoint_rep=inner([c*endpoint_t0 for c in ratios],GAMMA*endpoint_law['h'],ETA)
    return out(dict(parameters=dict(gamma=GAMMA,offset=OFFSET,rounding_error=ROUNDING,
                      ratios=ratios,eta=ETA,epsilon=EPSILON,nu=NU,densitylo=1,densityhi=densityhi),
        claimed_time=threshold,claimed_space=space_target,
        time_upper=worst_time,time_upper_decimal=float(worst_time),
        space_upper=worst_space,space_upper_decimal=float(worst_space),
        endpoint_time=endpoint_law['w']+EPSILON+endpoint_rep['time']/GAMMA,
        endpoint_space=endpoint_rep['space']/GAMMA,
        endpoint_parameters=[c*endpoint_t0 for c in ratios],
        minimum_feasibility_margins=min_margins,retained_coordinate_margin=retain,
        padding_margin=padding,density_endpoint_bounds=(first['h'],last['h']),
        unit_probability_monotonicity=last['mu'],
        partition=dict(visited=visited,leaves=len(leaves),interval=(left,right),
                       exact_coverage_verified=True,method='outward rational interval subdivision'),
        seconds=time.monotonic()-start))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    parser.add_argument('--profiles',type=Path,help='Optional JSON list of rational profiles')
    args=parser.parse_args()
    profiles=json.loads(args.profiles.read_text()) if args.profiles else PROFILES
    results=[]
    for profile in profiles:
        result=certify(profile);results.append(result)
        print(json.dumps({k:result[k] for k in ('parameters','time_upper_decimal','space_upper_decimal','partition','seconds')},indent=2))
    at_two=law(I(*density_bracket(2)))
    assert at_two['w'].lo>I('.62508').hi
    assert at_two['w'].hi<I('.62509').lo
    output=dict(scope='Rigorous feasible upper bounds; no optimality or algorithmic lower bound is claimed.',
                certificates=results,density_two_merged_entropy=out(at_two['w']))
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(density_two_merged_entropy=out(at_two['w'])),indent=2))


if __name__=='__main__':main()

  

#!/usr/bin/env python3
"""Rigorous uniform certificate for density-dependent SVP parameters.

For delta in [.54,1], use n0=ceil(n/gamma), gamma=1-10^-8.
The limiting congruence rate is rho=gamma*delta.  Approximate the
unit-coordinate probability p(1) from above within 10^-12, add 10^-9,
and multiply by the three fixed rational ratios below.  All interval
checks allow this full approximation error.  The output certificate
covers every entropy density, rather than testing sampled densities.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import time

from svp_global_upper import (SCALE,I,law,inner,out,density_bracket)

GAMMA=Fraction(99999999,100000000)
OFFSET=Fraction(1,10**9)
ROUNDING=Fraction(1,10**12)
RATIOS=tuple(map(Fraction,('1','.874597414466','.589775545104','.310623936690')))
ETA=Fraction(1,10**10)
EPSILON=Fraction(1,10**10)
CUTOFF=Fraction('.54')
NU=16
THRESHOLD=Fraction('.27400949')
SPACE=Fraction('.26139204')


def certify():
    start=time.monotonic()
    left=density_bracket(CUTOFF)[0];right=density_bracket(1)[1]
    # p_x(1) increases: d p_x(1)/d log x=p_x(1)(1-mu),
    # and Z(1-mu)=1-2 sum_{j>=2}(j^2-1)x^{j^2}>0.
    # The following geometric bound holds throughout x<=1/4.
    quarter=Fraction(1,4)
    assert 8*quarter**4/(1-Fraction(9,4)*quarter**5)<1
    pmax=law(I(right))['u']/2
    # For every allowed count u in [2p(1)-2epsilon,2p(1)+2epsilon],
    # b+u >= 1+[(1-gamma)(1-2p(1))-2(offset+rounding)]/gamma-2epsilon.
    # Thus all coordinates can be retained and the subset cost is zero.
    retain=(1-GAMMA)*(1-2*pmax)/GAMMA-2*(OFFSET+ROUNDING)/GAMMA-2*EPSILON
    assert retain.lo>0
    # H*t0/rho-p(1)-epsilon=t0/gamma-p(1)-epsilon.
    padding=OFFSET/GAMMA-EPSILON
    assert padding>0
    assert NU**3*(1/GAMMA)**2>1
    stack=[(left,right,0)];leaves=[];visited=0
    min_margins={};worst_time=Fraction(0);worst_space=Fraction(0)
    endpoint=None
    while stack:
        lo,hi,depth=stack.pop();visited+=1
        distribution=law(I(lo,hi))
        t0=distribution['u']/2+I(OFFSET,OFFSET+ROUNDING)
        try:
            rep=inner([c*t0 for c in RATIOS],GAMMA*distribution['h'],ETA)
            cost=distribution['w']+EPSILON+rep['time']/GAMMA
            space=rep['space']/GAMMA
            accepted=cost.hi<I(THRESHOLD).lo and space.hi<I(SPACE).lo
        except AssertionError:
            accepted=False
        if accepted:
            upper=Fraction(cost.hi,SCALE);sp=Fraction(space.hi,SCALE)
            worst_time=max(worst_time,upper);worst_space=max(worst_space,sp)
            for key,value in rep['margins'].items():
                min_margins[key]=min(min_margins.get(key,Fraction(100)),Fraction(value.lo,SCALE))
            leaves.append((lo,hi))
        else:
            assert depth<90, f'Certificate failed to converge near x={float(lo)}..{float(hi)}'
            middle=(lo+hi)/2
            stack.extend(((middle,hi,depth+1),(lo,middle,depth+1)))
        if visited%500==0:
            print(f'Checked {visited} intervals, {len(leaves)} accepted, {len(stack)} pending',flush=True)
    leaves.sort()
    assert leaves[0][0]==left and leaves[-1][1]==right
    assert all(a[1]==b[0] for a,b in zip(leaves,leaves[1:]))
    endpoint_law=law(I(*density_bracket(1)))
    endpoint_t0=endpoint_law['u']/2+I(OFFSET,OFFSET+ROUNDING)
    endpoint_rep=inner([c*endpoint_t0 for c in RATIOS],GAMMA*endpoint_law['h'],ETA)
    endpoint_time=endpoint_law['w']+EPSILON+endpoint_rep['time']/GAMMA
    return out(dict(scope='A rigorous feasible uniform upper bound for the stated density-dependent SVP parameter policy; no global lower bound is asserted by this file.',
        parameters=dict(gamma=GAMMA,offset=OFFSET,rounding_error=ROUNDING,ratios=RATIOS,
                        eta=ETA,epsilon=EPSILON,cutoff=CUTOFF,nu=NU),
        time_upper=max(CUTOFF/2,worst_time),time_upper_decimal=float(max(CUTOFF/2,worst_time)),
        space_upper=max(CUTOFF/4,worst_space),space_upper_decimal=float(max(CUTOFF/4,worst_space)),
        endpoint_time=endpoint_time,endpoint_parameters=[c*endpoint_t0 for c in RATIOS],
        minimum_feasibility_margins=min_margins,retained_coordinate_margin=retain,
        padding_margin=padding,density_endpoint_bounds=(law(I(left))['h'],law(I(right))['h']),
        partition=dict(visited=visited,leaves=len(leaves),interval=(left,right),
                       exact_coverage_verified=True,method='outward rational interval subdivision'),
        seconds=time.monotonic()-start))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('svp_adaptive_upper.json'))
    args=parser.parse_args()
    result=certify()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('time_upper_decimal','space_upper_decimal','endpoint_time','retained_coordinate_margin','partition','seconds')},indent=2))

if __name__=='__main__':main()

  

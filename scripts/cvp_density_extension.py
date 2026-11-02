#!/usr/bin/env python3
"""Certify a continuous parameter recipe for CVP densities 1<=delta<=3/2.

For q>=3, let h(x)=H(x,x,1-2x), choose the inverse x_delta in (0,1/3),
and put t_i=c_i*(x_delta+10^-7), c=(1,.7,.47,.24).  The certificate permits
an additional upward rounding error of at most 10^-12 in t_0.  The correction
radius coefficient is 2*t_0+2*10^-5.  The representation rate is rho=delta;
unlike the density<=1 support reduction, the support is the ambient dimension.

This proves numerical feasibility and cost bounds for this parameter recipe.
The mathematical high-density reduction is supplied separately in the paper.
No optimality or extension beyond delta=3/2 is asserted.
"""
import argparse
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import time

from svp_global_upper import SCALE, I, inner, maximum, out
from global_parameter_certificate import fixed_entropy as h, fixed_log2 as log2
from parameter_selection import (
    Interval, SELECTED_A, exact_bounds, exact_certificate, ln_interval,
)

OFFSET = Fraction(1,10**7)
ROUNDING = Fraction(1,10**12)
ETA = Fraction(1,10**8)
RADIUS_SLACK = Fraction(2,10**5)
RATIOS = tuple(map(Fraction,('1','.7','.47','.24')))
TIME_TARGET = Fraction('.56')
SPACE_TARGET = Fraction('.325')


@lru_cache(maxsize=100)
def inverse_entropy(density):
    """Outward rational inverse of h; h'(x)=2*log2((1-2x)/x)>0."""
    density = Fraction(density)
    if not 1 <= density <= Fraction(3,2):
        raise ValueError('the certified extension requires density in [1,3/2]')
    left,right = Fraction('.1'),Fraction('.26')
    target = I(density)
    assert h(I(left)).hi < target.lo
    assert h(I(right)).lo > target.hi
    for _ in range(75):
        middle = (left+right)/2
        value = h(I(middle))
        if value.hi < target.lo:
            left = middle
        elif value.lo > target.hi:
            right = middle
        else:
            # In particular h(1/4)=3/2 exactly.  A small symmetric rational
            # bracket handles an exactly attained bisection midpoint.
            radius = Fraction(1,10**24)
            left,right = middle-radius,middle+radius
            assert h(I(left)).hi < target.lo and h(I(right)).lo > target.hi
            break
    assert left < right < Fraction(1,3)
    return left,right


@lru_cache(maxsize=1)
def exact_coefficients():
    witness = exact_certificate()
    # The shifted-theta condition concerns a and gamma only, so the
    # original certificate remains valid at every positive radius c.
    slope = Interval(SELECTED_A)*Fraction(9,25)/(2*ln_interval(2))
    intercept = exact_bounds(correction=0)['time']
    return I(slope.lo,slope.hi),I(intercept.lo,intercept.hi),witness


def recipe(x,density):
    """Enclose a point or an entire inverse-entropy interval."""
    t0 = x+I(OFFSET,OFFSET+ROUNDING)
    coordinates = [ratio*t0 for ratio in RATIOS]
    representation = inner(coordinates,density,ETA)
    correction = 2*t0+RADIUS_SLACK
    slope,intercept,_ = exact_coefficients()
    exact = slope*correction+intercept
    return dict(time=maximum((representation['time'],exact)),
                space=maximum((representation['space'],exact/2)),
                representation=representation,exact_time=exact,
                correction=correction,beta=2*t0,frequencies=coordinates)


def certify_point(density):
    density = Fraction(density)
    bracket = inverse_entropy(density)
    value = recipe(I(*bracket),I(density))
    surplus = h(value['frequencies'][0])-density
    assert surplus.lo > 0
    return out(dict(density=density,inverse_entropy_bracket=bracket,
                    root_entropy_surplus=surplus,**value))


def certify():
    start = time.monotonic()
    left = inverse_entropy(1)[0]
    right = inverse_entropy(Fraction(3,2))[1]
    # h is increasing up to 1/3 and h' is decreasing.  The mean-value
    # theorem proves a uniform root entropy surplus despite correlation
    # between rho=h(x) and t_0=x+OFFSET+rounding.  Direct subtraction over
    # a broad interval would lose this correlation.
    largest_t0 = I(right)+OFFSET+ROUNDING
    assert largest_t0.hi < I(Fraction(1,3)).lo
    derivative = 2*log2((1-2*largest_t0)/largest_t0)
    assert derivative.lo > I('1.99').hi
    root_surplus = OFFSET*derivative
    assert root_surplus.lo > 0
    stack = [(left,right,0)]
    leaves = []
    visited = 0
    min_margins = {}
    worst_time = worst_space = Fraction(0)
    while stack:
        lo,hi,depth = stack.pop()
        visited += 1
        try:
            value = recipe(I(lo,hi),h(I(lo,hi)))
            accepted = value['time'].hi < I(TIME_TARGET).lo and value['space'].hi < I(SPACE_TARGET).lo
        except AssertionError:
            accepted = False
        if accepted:
            time_upper = Fraction(value['time'].hi,SCALE)
            space_upper = Fraction(value['space'].hi,SCALE)
            worst_time = max(worst_time,time_upper)
            worst_space = max(worst_space,space_upper)
            for key,bound in value['representation']['margins'].items():
                lower = Fraction(bound.lo,SCALE)
                min_margins[key] = min(min_margins.get(key,Fraction(100)),lower)
            leaves.append((lo,hi,time_upper,space_upper))
        else:
            assert depth < 70, f'Certificate did not converge on [{lo},{hi}]'
            middle = (lo+hi)/2
            stack.extend(((middle,hi,depth+1),(lo,middle,depth+1)))
    leaves.sort()
    assert leaves[0][0] == left and leaves[-1][1] == right
    assert all(a[1] == b[0] for a,b in zip(leaves,leaves[1:]))
    assert h(I(left)).hi < I(1).lo
    assert h(I(right)).lo > I(Fraction(3,2)).hi
    _,_,theta_witness = exact_coefficients()
    return out(dict(
        scope='Numerical feasibility and asymptotic exponent upper bounds for one continuous '
              'q>=3 CVP parameter recipe on 1<=delta<=3/2; no optimality claim.',
        parameters=dict(approximation_factor='5/3',a=SELECTED_A,densitylo=1,densityhi='3/2',
                        offset=OFFSET,rounding_error=ROUNDING,ratios=RATIOS,eta=ETA,
                        radius_slack=RADIUS_SLACK,representation_rate='rho=delta'),
        formulas=dict(inverse_entropy='H(x_delta,x_delta,1-2*x_delta)=delta, x_delta<1/3',
                      frequencies='t_i=c_i*(x_delta+offset+rounding)',
                      correction='c_delta=2*t_0+radius_slack',
                      time='max(t_inner(delta), (a*c_delta*9/25+ln(theta(a)))/(2*ln(2)))',
                      space='max(s_inner(delta), (a*c_delta*9/25+ln(theta(a)))/(4*ln(2)))'),
        claimed_time=TIME_TARGET,claimed_space=SPACE_TARGET,
        time_upper=worst_time,time_upper_decimal=float(worst_time),
        space_upper=worst_space,space_upper_decimal=float(worst_space),
        root_entropy_surplus=root_surplus,root_entropy_derivative_lower_bound=derivative,
        minimum_feasibility_margins=min_margins,
        shifted_theta_certificate=theta_witness,
        partition=dict(visited=visited,leaves=len(leaves),interval=(left,right),
                       exact_coverage_verified=True,method='outward rational interval subdivision'),
        samples=[certify_point(d) for d in ('1','1.1','1.2','1.3','1.4','1.5')],
        seconds=time.monotonic()-start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density',type=Fraction,help='evaluate one exact density in [1,3/2]')
    parser.add_argument('--output',type=Path,
                        help='JSON destination; the full certificate defaults to the script stem, '
                             'while a point query writes no file unless requested')
    args = parser.parse_args()
    try:
        result = certify() if args.density is None else certify_point(args.density)
    except ValueError as error:
        parser.error(str(error))
    output = args.output
    if output is None and args.density is None:
        output = Path(__file__).with_suffix('.json')
    if output is not None:
        output.write_text(json.dumps(result,indent=2)+'\n')
    summary = ({key:result[key] for key in ('time_upper_decimal','space_upper_decimal','partition','seconds')}
               if args.density is None else {key:result[key] for key in ('density','time','space')})
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

  

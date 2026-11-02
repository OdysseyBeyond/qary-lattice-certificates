#!/usr/bin/env python3
"""Certify a feasible parameter choice for the complete randomized SVP algorithm.

Only rational arithmetic and explicit series remainders enter certification.
This is a feasible-witness certificate, not a global-optimality certificate.
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import parameter_selection as common

# Every interval operation rounds outward to this rational grid.
common.GRID = 10**30
Interval = common.Interval
as_interval = common.as_interval
SELECTED_T = ('.11016', '.09648', '.06502', '.03416')
SELECTED_RHO = '.9823'
SELECTED_ETA = '.00000001'
EPSILON = '.0001'
CUTOFF = '.55'
NU = 16
X_LO = '0.052'
X_HI = '0.14447841'
PARTITIONS = 4096
LN_TERMS = 24


def ln_point(x):
    """Natural logarithm, with exact power-of-two reduction and a tail bound."""
    x = Fraction(x)
    assert x > 0
    shift = 0
    while x < 1:
        x *= 2
        shift -= 1
    while x > 2:
        x /= 2
        shift += 1
    def reduced(y):
        z = (Interval(y)-1)/(Interval(y)+1)
        term = z
        total = Interval(0)
        square = z*z
        for j in range(LN_TERMS):
            total += 2*term/(2*j+1)
            term *= square
        tail = 2*term/((2*LN_TERMS+1)*(1-square))
        return Interval(total.lo, (total+tail).hi)
    value = reduced(x)
    if shift:
        value += shift*LN2
    return value


LN2 = ln_point(2)


def ln(x):
    x = as_interval(x)
    return Interval(ln_point(x.lo).lo, ln_point(x.hi).hi)


def entropy(values):
    out = Interval(0)
    for x in values:
        x = as_interval(x)
        if x.lo == x.hi == 0:
            continue
        assert x.lo > 0
        out -= x*ln(x)/LN2
    return out


def maximum(values):
    return Interval(max(v.lo for v in values), max(v.hi for v in values))


def minimum(values):
    return Interval(min(v.lo for v in values), min(v.hi for v in values))


def positive(x):
    return maximum((Interval(0), x))


def representation_certificate(t=SELECTED_T, rho=SELECTED_RHO, eta=SELECTED_ETA):
    t = list(map(Interval,t))
    rho,eta=Interval(rho),Interval(eta)
    k=len(t)-1
    h=[entropy((u,u,1-2*u)) for u in t]
    r=[]
    margins={'eta':eta,'rho':rho}
    for i,u in enumerate(t):
        margins[f'positive_coordinate_{i}']=u
        margins[f'coordinate_below_half_{i}']=Fraction(1,2)-u
    for i,(u,v) in enumerate(zip(t,t[1:]),1):
        z=v-u/2
        zero=1-2*u
        margins[f'added_signs_{i}']=z
        margins[f'unused_zeros_{i}']=zero-2*z
        r.append(2*u+zero*entropy((z/zero,z/zero,1-2*z/zero)))
    d=[rho]+[u-eta for u in r]
    margins['last_equations_positive']=d[-1]
    for i in range(1,k+1):
        margins[f'equation_gap_{i}']=d[i-1]-d[i]
        margins[f'list_parity_{i}']=entropy((2*t[i],1-2*t[i]))-d[i-1]/2
        margins[f'decomposition_parity_{i}']=r[i-1]-d[i]/2-2*t[i]+t[i-1]
    margins['root_parity']=entropy((2*t[0],1-2*t[0]))-rho/2
    assert all(u.lo>0 for u in margins.values())
    ell=[positive(u-v) for u,v in zip(h,d)]
    time_terms=[h[-1]/2]+[ell[i]+positive(h[i]-d[i-1]) for i in range(1,k+1)]
    space_terms=[h[-1]/2]+ell[:-1]+[positive(h[-1]-d[-2])]
    time,space=maximum(time_terms),maximum(space_terms)
    assert time.hi<Fraction('.26343')
    assert (space/rho).hi<Fraction('.2644')
    return dict(depth=k,rho=rho,t=t,eta=eta,h=h,r=r,d=d,
                time=time,space=space,time_terms=time_terms,space_terms=space_terms,
                margins=margins,root_entropy_minus_rho=h[0]-rho)


def law_bounds(x):
    """Enclose theta, second moment, entropy and large-coordinate entropy.

    Sum j=1,...,5 and bound the tail from j=6. The ratio of consecutive
    summands is at most x^13; for second moments it is at most
    (49/36)*x^13. This also bounds the tail in the large-coordinate sum.
    """
    x=as_interval(x)
    assert 0<x.lo<=x.hi<Fraction(1,4)
    tail_sum=2*x**36/(1-x**13)
    tail_moment=72*x**36/(1-Fraction(49,36)*x**13)
    large_sum=2*sum((x**(j*j) for j in range(2,6)),Interval(0))
    large_energy=2*sum((j*j*x**(j*j) for j in range(2,6)),Interval(0))
    large_sum=Interval(large_sum.lo,(large_sum+tail_sum).hi)
    large_energy=Interval(large_energy.lo,(large_energy+tail_moment).hi)
    theta=1+2*x+large_sum
    mu=(2*x+large_energy)/theta
    p=large_sum/theta
    nu=large_energy/theta
    logx=ln(x)
    logtheta=ln(theta)
    density=(logtheta-mu*logx)/LN2
    w=2*x/theta
    # This form avoids a large cancellation between entropy terms.
    large_entropy=(-(1-p)*ln(1-p)+p*logtheta-nu*logx)/LN2
    return dict(theta=theta,mu=mu,density=density,w=w,
                large_probability=p,large_entropy=large_entropy,
                tail_sum=tail_sum,tail_moment=tail_moment)


def outer_bound(A,ulo,uhi):
    """An upper enclosure of sup G(A,u) on the given interval.

    A is a rational lower bound, since G decreases with A. Before
    A+u reaches 1, its maximum occurs at u=(1-A)/2. Beyond that point
    the subset cost is zero. Only one rational u needs evaluation.
    """
    A,ulo,uhi=map(Fraction,(A,ulo,uhi))
    assert 0<A and 0<ulo<=uhi<1
    u=max(ulo,min(uhi,(1-A)/2))
    if A+u>=1:
        return Interval(0)
    assert A+u<1
    return (A*ln(A)-(A+u)*ln(A+u)-(1-u)*ln(1-u))/LN2


def certify(partitions=PARTITIONS):
    assert partitions>=1
    rep=representation_certificate()
    rho=Fraction(SELECTED_RHO)
    t0=Fraction(SELECTED_T[0])
    assert NU**3*(Fraction(CUTOFF)/rho)**2>1000
    beta=2*t0
    epsilon=Fraction(EPSILON)
    left,right=map(Fraction,(X_LO,X_HI))
    endpoints=[left+(right-left)*i/partitions for i in range(partitions+1)]
    prev=law_bounds(endpoints[0])
    first=prev
    upper_time=Fraction(0)
    min_padding=Fraction(1)
    max_shape_ratio=Fraction(0)
    max_large_entropy=Fraction(0)
    worst_time_index=None
    xquarter=Fraction(1,4)
    derivative_tail=72*xquarter**36/(1-Fraction(49,36)*xquarter**13)
    assert 2*sum((j*j-1)*xquarter**(j*j) for j in range(2,6))+derivative_tail<1
    for i in range(partitions):
        nxt=law_bounds(endpoints[i+1])
        interval=law_bounds(Interval(endpoints[i],endpoints[i+1]))
        # Entropy increases: d H(P_x)/d log x=-log(x) Var(X^2)/ln2 >0.
        density=Interval(prev['density'].lo,nxt['density'].hi)
        # w increases on this range: Z-x Z'=1-2 sum (j^2-1)x^(j^2)>0.
        # The following coarse bound proves this for every x <= 1/4.
        w=Interval(prev['w'].lo,nxt['w'].hi)
        A=(1-beta)*density/rho
        subset=outer_bound(A.lo,w.lo-2*epsilon,w.hi+2*epsilon)
        total=interval['large_entropy']+epsilon+subset+Interval('.26343')*density/rho
        padding=t0*density/rho-w/2-epsilon
        ratio=interval['mu']/density
        upper_time=max(upper_time,total.hi)
        if upper_time==total.hi:worst_time_index=i
        min_padding=min(min_padding,padding.lo)
        max_shape_ratio=max(max_shape_ratio,ratio.hi)
        max_large_entropy=max(max_large_entropy,interval['large_entropy'].hi)
        prev=nxt
        if (i+1)%512==0:
            print(f'Certified {i+1}/{partitions} intervals; time upper {float(upper_time):.9f}',flush=True)
    assert first['w'].lo/2>Fraction('.0471')>Fraction(EPSILON)
    assert first['density'].hi<Fraction(CUTOFF)
    assert prev['density'].lo>1
    assert min_padding>0
    assert max_shape_ratio<Fraction('.23')
    assert max_large_entropy<Fraction('.009')
    assert upper_time<Fraction('.277045'), float(upper_time)
    low_time=Fraction(CUTOFF)/2
    low_space=Fraction(CUTOFF)/4
    full_time=max(low_time,upper_time)
    full_space=max(low_space,(rep['space']/rho).hi)
    assert full_time<Fraction('.2771')
    assert full_space<Fraction('.264385')
    def serial(v):
        if isinstance(v,Interval):return v.display()
        if isinstance(v,Fraction):return Interval(v).display()
        if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [serial(x) for x in v]
        return v
    return serial(dict(
        scope='A feasible parameter choice for the complete SVP time and space bounds; no global-optimality claim.',
        selected=dict(depth=3,rho=SELECTED_RHO,t=SELECTED_T,eta=SELECTED_ETA,
                      epsilon=EPSILON,cutoff=CUTOFF,nu=NU),
        numerical_method=dict(partitions=partitions,interval=[X_LO,X_HI],
                              grid='10^-30',logarithm_terms=LN_TERMS,
                              theta_last_index=5),
        representation=rep,
        endpoint_density=[first['density'],prev['density']],
        recovery_margin_cubed=NU**3*(Fraction(CUTOFF)/rho)**2,
        unit_probability_lower=first['w'].lo/2,
        padding_margin=min_padding,
        squared_radius_per_density_upper=max_shape_ratio,
        large_coordinate_entropy_upper=max_large_entropy,
        randomized_time_upper=upper_time,
        worst_time_interval=[endpoints[worst_time_index],endpoints[worst_time_index+1]],
        low_density_time=low_time,low_density_space=low_space,
        time_upper=full_time,space_upper=full_space,
        main_time_bound='.277045',total_space_bound='.264385',
        claimed_time='.2771',claimed_space='.2644'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certify',action='store_true',help='run the rational interval certificate (default)')
    parser.add_argument('--partitions',type=int,default=PARTITIONS)
    args=parser.parse_args()
    result=certify(args.partitions)
    target=Path(__file__).with_name('svp_parameter_certificate.json')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('time_upper','space_upper','padding_margin','squared_radius_per_density_upper','large_coordinate_entropy_upper')},indent=2))


if __name__=='__main__':main()

  

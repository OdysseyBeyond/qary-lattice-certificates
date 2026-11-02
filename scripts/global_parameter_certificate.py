#!/usr/bin/env python3
"""Certify the numerical global CVP lower bound (standard library only).

The manuscript reduces the global representation-cost lower bound to a
two-variable rectangle.  This program certifies that reduction's remaining
numerical inequality, with threshold 0.26675186.  It also checks the feasible
rational upper-bound witness from parameter_selection.py.  It does not infer
global optimality from a numerical optimizer.

The initial rectangle has exact dyadic endpoints.  Each subdivision uses an
exact rational midpoint.  Floating-point calculations only propose root
brackets, whose endpoints and signs are checked by integer-interval arithmetic.
Every feasible rectangle is tested with a rigorous lower bound.  Leaves are
classified separately as infeasible, externally bounded, or contained in a
rational rectangle where a
convex weighted lower bound is certified.  An independent exact sweep checks
that the resulting leaves partition the complete initial rectangle.

Usage:
    python3 global_parameter_certificate.py --output global_parameter_certificate.json

The analytical domain reduction, monotonicity facts, and relation between the
weighted function and the full cost are proved in the manuscript.  This script
checks their numerical bounds; its JSON records that scope explicitly.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import time

from parameter_selection import (
    Interval, entropy_interval, ln_interval, representation_certificate,
)


THRESHOLD = Fraction('.26675186')
INITIAL = (Fraction(5, 64), Fraction(9, 64),
           Fraction(5, 128), Fraction(9, 64))
LOCAL = tuple(map(Fraction, ('.098674', '.099674', '.066409', '.067409')))
ROOT_LO = Fraction('.11354609760967409')
ROOT_HI = Fraction('.11354609760967411')
CENTER = tuple(map(Fraction, ('.09917396755431343', '.06690925652273322',
                              '.03531462090790641')))
RADIUS = Fraction('.001')


# The large subdivision certificate uses fixed-point integer endpoints.  All
# arithmetic operations round outwards; the smaller local certificate uses
# the independently implemented Fraction intervals in parameter_selection.py.
SCALE = 10**35


class FixedInterval:
    __slots__ = ('lo', 'hi')

    def __init__(self, lower=0, upper=None, *, raw=False):
        if raw:
            assert isinstance(lower, int) and isinstance(upper, int)
            self.lo, self.hi = lower, upper
        else:
            lower = Fraction(lower)
            upper = lower if upper is None else Fraction(upper)
            self.lo = lower.numerator * SCALE // lower.denominator
            self.hi = -((-upper.numerator * SCALE) // upper.denominator)
        assert self.lo <= self.hi

    def __add__(self, other):
        other = fixed(other)
        return FixedInterval(self.lo + other.lo, self.hi + other.hi, raw=True)

    __radd__ = __add__

    def __neg__(self):
        return FixedInterval(-self.hi, -self.lo, raw=True)

    def __sub__(self, other):
        return self + (-fixed(other))

    def __rsub__(self, other):
        return fixed(other) + (-self)

    def __mul__(self, other):
        other = fixed(other)
        products = [x*y for x in (self.lo, self.hi)
                    for y in (other.lo, other.hi)]
        return FixedInterval(min(products)//SCALE, -(-max(products)//SCALE), raw=True)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = fixed(other)
        assert other.lo * other.hi > 0
        ratios = [(x*SCALE, y) for x in (self.lo, self.hi)
                  for y in (other.lo, other.hi)]
        return FixedInterval(min(x//y for x, y in ratios),
                             max(-(-x//y) for x, y in ratios), raw=True)

    def __rtruediv__(self, other):
        return fixed(other) / self


def fixed(value):
    return value if isinstance(value, FixedInterval) else FixedInterval(value)


def fixed_max(left, right):
    left, right = fixed(left), fixed(right)
    return FixedInterval(max(left.lo, right.lo), max(left.hi, right.hi), raw=True)


@lru_cache(maxsize=400000)
def log_at_grid(integer):
    """Enclose log(integer/SCALE) by a positive atanh series.

    After reduction to [1,2], z=(x-1)/(x+1) lies in [0,1/3].  Forty
    terms end at z^79/79; the omitted tail is at most
    2*z^81/(81*(1-z^2)).  Range reduction and the tail are rounded outwards.
    """
    assert integer > 0
    x = FixedInterval(integer, integer, raw=True)
    shift = 0
    while x.lo < SCALE:
        x = 2*x
        shift -= 1
    while x.hi > 2*SCALE:
        x = x/2
        shift += 1
    assert SCALE <= x.lo <= x.hi <= 2*SCALE
    z = (x-1)/(x+1)
    z_squared = z*z
    power, total = z, FixedInterval(0)
    for j in range(40):
        total += power/(2*j+1)
        power *= z_squared
    partial = 2*total
    tail = 2*power/(81*(1-z_squared))
    result = FixedInterval(partial.lo, (partial+tail).hi, raw=True)
    if shift:
        result += shift*FIXED_LN2
    return result


# This invocation has shift zero, so it does not reference FIXED_LN2 itself.
FIXED_LN2 = log_at_grid(2*SCALE)


def fixed_log(value):
    value = fixed(value)
    return FixedInterval(log_at_grid(value.lo).lo, log_at_grid(value.hi).hi, raw=True)


def fixed_log2(value):
    return fixed_log(value)/FIXED_LN2


def fixed_entropy(value):
    """h(t)=H(t,t,1-2t), on the domains checked below."""
    value = fixed(value)
    if value.lo == value.hi == 0:
        return FixedInterval(0)
    if value.lo == value.hi == SCALE//2:
        return FixedInterval(1)
    assert 0 < value.lo and 2*value.hi < SCALE
    return -2*value*fixed_log2(value)-(1-2*value)*fixed_log2(1-2*value)


def fixed_representations(parent, child):
    parent, child = fixed(parent), fixed(child)
    added, remaining, zero = child-parent/2, 1-parent-2*child, 1-2*parent
    assert added.lo >= 0 and remaining.lo > 0 and zero.lo > 0
    # We evaluate exact grid points here.  An interval touching zero without
    # being exactly zero would require a separate x*log(x) enclosure.
    assert added.lo > 0 or added.hi == 0
    added_term = FixedInterval(0) if added.hi == 0 else 2*added*fixed_log2(added)
    return 2*parent-added_term-remaining*fixed_log2(remaining)+zero*fixed_log2(zero)


def stationary_polynomial(parent, child):
    return child*(1-2*child)*(1-3*child)-parent/2*(1-4*child+2*child*child)


def float_entropy(value):
    if value == 0:
        return 0.
    if value == .5:
        return 1.
    return -2*value*math.log2(value)-(1-2*value)*math.log2(1-2*value)


def float_representations(parent, child):
    added, remaining, zero = child-parent/2, 1-parent-2*child, 1-2*parent
    assert added >= -1e-14 and remaining > 0 and zero > 0
    added_term = 2*added*math.log2(added) if added > 0 else 0.
    return 2*parent-added_term-remaining*math.log2(remaining)+zero*math.log2(zero)


def float_inverse_entropy(value, lower=0., upper=.25):
    if value <= float_entropy(lower):
        return lower
    if value >= float_entropy(upper):
        return upper
    for _ in range(45):
        middle = (lower+upper)/2
        if float_entropy(middle) < value:
            lower = middle
        else:
            upper = middle
    return (lower+upper)/2


def float_last_merge(parent, budget):
    lower, upper = parent/2, .25
    for _ in range(45):
        middle = (lower+upper)/2
        derivative = (4*math.log2((1-2*middle)/middle)
                      - 2*math.log2((1-parent-2*middle)/(middle-parent/2)))
        if derivative < 0:
            lower = middle
        else:
            upper = middle
    child = max((lower+upper)/2, float_inverse_entropy(budget, parent/2, .25))
    h_child = float_entropy(child)
    return h_child-float_representations(parent, child)+max(0., h_child-budget)


def proposed_lower(box):
    """Floating proposal only; its result is never accepted as a certificate."""
    xl, xu, yl, yu = map(float, box)
    if yu < xl/2:
        return math.inf
    yl = max(yl, xl/2)
    upper_second = float_representations(xl, yu)
    upper_first = float_representations(.11354609760967407, xu)
    h_lower = float_entropy(yl)
    second = h_lower-upper_second+max(0., h_lower-upper_first)
    return max(second, float_last_merge(yl, upper_second))


def certified_last_merge(parent, budget):
    """Lower bound G(s,d)=min_t h(t)-r(s,t)+[h(t)-d]_+.

    The manuscript proves that its minimizer is the maximum of the root of
    stationary_polynomial(s,t) and h^{-1}(d), on [s/2,1/4].  Approximate
    brackets are certified below.  On that interval h-r is decreasing and
    h is increasing, so the two terms may be bounded at opposite endpoints.
    """
    parent_float, budget_float = parent.lo/SCALE, budget.hi/SCALE
    lower, upper = parent_float/2, .25
    for _ in range(45):
        middle = (lower+upper)/2
        if stationary_polynomial(parent_float, middle) < 0:
            lower = middle
        else:
            upper = middle
    middle = (lower+upper)/2
    root_lower = FixedInterval(middle-1e-12)
    root_upper = FixedInterval(middle+1e-12)
    assert stationary_polynomial(parent, root_lower).hi < 0
    assert stationary_polynomial(parent, root_upper).lo > 0
    assert Fraction(root_upper.hi-root_lower.lo, SCALE) <= Fraction(1, 2**38)
    # The entropy inverse is in (0,1/4), possibly below s/2.  The maximum
    # with the stationary root is checked to lie in [s/2,1/4] below.
    assert budget.lo > 0
    assert budget.hi < fixed_entropy(Fraction(1, 4)).lo
    middle = float_inverse_entropy(budget_float)
    inverse_lower = FixedInterval(middle-1e-12)
    inverse_upper = FixedInterval(middle+1e-12)
    assert fixed_entropy(inverse_lower).hi < budget.lo
    assert fixed_entropy(inverse_upper).lo > budget.hi
    assert Fraction(inverse_upper.hi-inverse_lower.lo, SCALE) <= Fraction(1, 2**38)
    minimizer_lower = fixed_max(root_lower, inverse_lower)
    minimizer_upper = fixed_max(root_upper, inverse_upper)
    assert 2*minimizer_lower.lo >= parent.hi
    assert minimizer_upper.hi <= SCALE//4
    return (fixed_entropy(minimizer_upper)
            - fixed_representations(parent, minimizer_upper)
            + fixed_max(0, fixed_entropy(minimizer_lower)-budget)).lo


def certified_lower(box):
    xl, xu, yl, yu = map(FixedInterval, box)
    if yu.hi < (xl/2).lo:
        return 2*SCALE  # This box has no physically admissible pair.
    yl = fixed_max(yl, xl/2)
    upper_second = fixed_representations(xl, yu)
    upper_second = FixedInterval(upper_second.hi, upper_second.hi, raw=True)
    upper_first = fixed_representations(ROOT_LO, xu)
    h_lower = fixed_entropy(yl)
    second = h_lower-upper_second+fixed_max(0, h_lower-upper_first)
    if second.lo > FixedInterval(THRESHOLD).hi:
        return second.lo
    return max(second.lo, certified_last_merge(yl, upper_second))


def local_certificate():
    """Certify convexity, tangent bound, and confinement near the minimum."""
    ln2 = ln_interval(Interval(2))
    w, v = Interval('.415123714084'), Interval('.513121169111')
    u = 1-w-v
    assert min(u.lo, v.lo, w.lo) > 0 and (u+v+w).lo <= 1 <= (u+v+w).hi
    root = Interval(ROOT_LO, ROOT_HI)
    point = list(map(Interval, CENTER))
    box = [Interval(center-RADIUS, center+RADIUS) for center in CENTER]

    def entropy(t):
        return entropy_interval((t, t, 1-2*t))

    def representations(s, t):
        added, remaining, zero = t-s/2, 1-s-2*t, 1-2*s
        return 2*s+zero*entropy_interval((added/zero, added/zero, remaining/zero))

    def entropy_derivative(t):
        return 2*ln_interval((1-2*t)/t)/ln2

    def representation_derivatives(s, t):
        added, remaining, zero = t-s/2, 1-s-2*t, 1-2*s
        return (2+ln_interval(added*remaining/(zero*zero))/ln2,
                2*ln_interval(remaining/added)/ln2)

    def rank_certificate(domain):
        """Positive-definite rational lower bound for ln(2) times Hessian W.

        Positive terms multiplying fixed rank-one PSD matrices are bounded
        below coefficientwise; negative diagonal terms are bounded below
        separately.  The resulting tridiagonal matrix is positive definite
        if its three exact rational elimination pivots are positive.
        """
        x, y, z = domain
        z1, v1 = x-root/2, 1-root-2*x
        z2, v2, u2 = y-x/2, 1-x-2*y, 1-2*x
        z3, v3, u3 = z-y/2, 1-y-2*z, 1-2*y
        assert min(z1.lo, v1.lo, z2.lo, v2.lo, u2.lo, z3.lo, v3.lo, u3.lo) > 0
        coefficients = [w*(2/z1+4/v1)-4*(w+v)/u2,
                        -4*w/(y*(1-2*y))-4*(1-w)/u3,
                        -2*(u+2*v)/(z*(1-2*z)),
                        2*(w+v)/z2, (w+v)/v2, 2*(1-w)/z3, (1-w)/v3]
        aa, cc, ee, a2, b2, a3, b3 = (term.lo for term in coefficients)
        aa += a2/4+b2
        bb = -a2/2+2*b2
        cc += a2+4*b2+a3/4+b3
        dd = -a3/2+2*b3
        ee += a3+4*b3
        assert aa > 0
        pivot2 = cc-bb*bb/aa
        assert pivot2 > 0
        pivot3 = ee-dd*dd/pivot2
        assert pivot3 > 0
        # Also verify the simpler rational lower matrix printed in the paper.
        coarse = list(map(Fraction, ('16.8926', '-31.8604', '-69.0079',
                                    '98.6326', '1.2055', '348.1414', '.6757')))
        assert all(term.lo > bound for term, bound in zip(coefficients, coarse))
        d1, d2, d3, a2, b2, a3, b3 = coarse
        ma, mb = d1+a2/4+b2, -a2/2+2*b2
        mc, md, me = d2+a2+4*b2+a3/4+b3, -a3/2+2*b3, d3+a3+4*b3
        expected = list(map(Fraction, ('42.75625', '-46.9053', '159.30525',
                                      '-172.7193', '281.8363')))
        assert [ma, mb, mc, md, me] == expected
        mp2 = mc-mb*mb/ma
        mp3 = me-md*md/mp2
        assert ma > Fraction('42.756') and mp2 > Fraction('107.848') and mp3 > Fraction('5.225')
        return (aa, pivot2, pivot3), dict(
            coefficient_lower_bounds=[str(term) for term in coarse],
            tridiagonal_entries=[str(term) for term in expected],
            positive_pivots=[Interval(term).display() for term in (ma, mp2, mp3)])

    root_left = entropy(Interval(ROOT_LO))-1
    root_right = entropy(Interval(ROOT_HI))-1
    assert root_left.hi < 0 < root_right.lo
    pivots, manuscript_matrix = rank_certificate(box)
    x, y, z = point
    value = (2*w*entropy(y)+(u+2*v)*entropy(z)-w*representations(root, x)
             -(w+v)*representations(x, y)-(1-w)*representations(y, z))
    a1, b1 = representation_derivatives(root, x)
    a2, b2 = representation_derivatives(x, y)
    a3, b3 = representation_derivatives(y, z)
    gradient = [-w*b1-(w+v)*a2,
                2*w*entropy_derivative(y)-(w+v)*b2-(1-w)*a3,
                (u+2*v)*entropy_derivative(z)-(1-w)*b3]
    tangent_loss = RADIUS*sum(max(abs(g.lo), abs(g.hi)) for g in gradient)
    tangent_lower = value.lo-tangent_loss
    assert tangent_lower > THRESHOLD

    # Confinement: for x,y in LOCAL the minimizer over the last variable
    # lies between the two z-faces of the convexity box.  Entropy-inverse
    # brackets handle one branch, and the polynomial sign the other.
    xl, xu, yl, yu = map(Interval, LOCAL)
    assert box[0].lo <= xl.lo <= xu.hi <= box[0].hi
    assert box[1].lo <= yl.lo <= yu.hi <= box[1].hi
    zl, zu = Interval(CENTER[2]-RADIUS), Interval(CENTER[2]+RADIUS)
    confinement_lower = representations(xu, yl)-entropy(zl)
    confinement_upper = entropy(zu)-representations(xl, yu)
    polynomial_upper = stationary_polynomial(yu, zu)
    assert min(confinement_lower.lo, confinement_upper.lo, polynomial_upper.lo) > 0
    return dict(
        root_bracket=[str(ROOT_LO), str(ROOT_HI)],
        root_entropy_signs=[root_left.display(), root_right.display()],
        weights=dict(u=u.display(), v=v.display(), w=w.display()),
        center=[str(x) for x in CENTER], radius=str(RADIUS),
        positive_hessian_pivots=[Interval(x).display() for x in pivots],
        manuscript_hessian_lower_matrix=manuscript_matrix,
        weighted_value=value.display(), gradient=[g.display() for g in gradient],
        tangent_loss=Interval(tangent_loss).display(),
        weighted_lower=Interval(tangent_lower).display(),
        confinement_lower=confinement_lower.display(),
        confinement_upper=confinement_upper.display(),
        confinement_polynomial=polynomial_upper.display(),
    )


def domain_reduction_certificate():
    """Check the rational numerical comparisons used before subdivision."""
    def bounds(value):
        return Interval(Fraction(value.lo, SCALE), Fraction(value.hi, SCALE)).display()

    first_entropy = fixed_entropy('.046')
    first_list = fixed_entropy('.08')-fixed_representations('.1135', '.08')
    second_list = fixed_entropy('.046')-fixed_representations('.08', '.046')
    first_cost = 2*fixed_entropy('.14')-fixed_representations(ROOT_LO, '.14')-1
    first_budget = fixed_representations('.1135', '.14')-Fraction('.14')
    second_cost = fixed_entropy('.14')-Fraction('.834540')
    assert first_entropy.lo > FixedInterval('.535110').hi
    assert first_list.lo > FixedInterval('.267218').hi
    assert second_list.lo > FixedInterval('.272371').hi
    assert first_cost.lo > FixedInterval('.296361').hi
    assert first_budget.hi < FixedInterval('.834540').lo
    assert second_cost.lo > FixedInterval('.300910').hi

    # Polynomial endpoints and concavity on [.14,.267].
    s, left, right = map(Fraction, ('.114', '.14', '.267'))
    polynomial_endpoints = [stationary_polynomial(s, t) for t in (left, right)]
    assert polynomial_endpoints == list(map(Fraction, ('.0311496', '.020509032')))
    assert 36*right-10-2*s < 0
    assert 1-4*right+2*right*right > 0
    # r_t(alpha,t)>1 through t=.14, as used for the first-budget bound.
    # The ratio (1-s-2t)/(t-s/2) decreases with t and increases with s.
    ratio = (1-FixedInterval(ROOT_LO)-2*Fraction('.14'))/(Fraction('.14')-FixedInterval(ROOT_LO)/2)
    derivative_margin = 2*fixed_log2(ratio)-1
    assert derivative_margin.lo > 0
    return dict(
        entropy_at_0046=bounds(first_entropy),
        list_at_01135_008=bounds(first_list),
        list_at_008_0046=bounds(second_list),
        first_cost_at_014_lower=bounds(first_cost),
        first_budget_upper=bounds(first_budget),
        second_cost_at_014_lower=bounds(second_cost),
        polynomial_endpoints=[str(value) for value in polynomial_endpoints],
        polynomial_second_derivative_upper=str(36*right-10-2*s),
        first_budget_derivative_margin=bounds(derivative_margin),
    )


def inside_local(box):
    xl, xu, yl, yu = box
    return LOCAL[0] <= xl <= xu <= LOCAL[1] and LOCAL[2] <= yl <= yu <= LOCAL[3]


def subdivide(box):
    """Exact midpoint split, including assertions that both children tile it."""
    xl, xu, yl, yu = box
    if xu-xl >= yu-yl:
        middle = (xl+xu)/2
        children = ((xl, middle, yl, yu), (middle, xu, yl, yu))
        assert xl < middle < xu and children[0][1] == children[1][0]
    else:
        middle = (yl+yu)/2
        children = ((xl, xu, yl, middle), (xl, xu, middle, yu))
        assert yl < middle < yu and children[0][3] == children[1][2]
    assert sum((b-a)*(d-c) for a, b, c, d in children) == (xu-xl)*(yu-yl)
    for child in children:
        # SCALE is divisible by 2^35.  This check prevents an endpoint
        # roundoff from disguising an exact zero decomposition frequency.
        assert all(SCALE % endpoint.denominator == 0 for endpoint in child)
    return children


def audit_partition(rectangles):
    """Independent exact sweep: each interior point has multiplicity one."""
    ys = sorted({INITIAL[2], INITIAL[3]} | {x for box in rectangles for x in box[2:]})
    index = {value: i for i, value in enumerate(ys)}
    events = defaultdict(list)
    for xl, xu, yl, yu in rectangles:
        assert INITIAL[0] <= xl < xu <= INITIAL[1]
        assert INITIAL[2] <= yl < yu <= INITIAL[3]
        events[xl].append((index[yl], index[yu], 1))
        events[xu].append((index[yl], index[yu], -1))
    count = len(ys)-1
    minimum, maximum, lazy = [0]*(4*count), [0]*(4*count), [0]*(4*count)

    def add(node, left, right, first, last, increment):
        if last <= left or right <= first:
            return
        if first <= left and right <= last:
            minimum[node] += increment
            maximum[node] += increment
            lazy[node] += increment
            return
        middle = (left+right)//2
        add(2*node, left, middle, first, last, increment)
        add(2*node+1, middle, right, first, last, increment)
        minimum[node] = lazy[node]+min(minimum[2*node], minimum[2*node+1])
        maximum[node] = lazy[node]+max(maximum[2*node], maximum[2*node+1])

    xs = sorted(events)
    assert xs[0] == INITIAL[0] and xs[-1] == INITIAL[1]
    for position, x in enumerate(xs):
        for first, last, increment in events[x]:
            add(1, 0, count, first, last, increment)
        expected = int(position < len(xs)-1)
        assert minimum[1] == maximum[1] == expected
    area = sum((xu-xl)*(yu-yl) for xl, xu, yl, yu in rectangles)
    assert area == (INITIAL[1]-INITIAL[0])*(INITIAL[3]-INITIAL[2])
    return dict(exact_midpoint_splits=True, exact_sweep_passed=True,
                area=str(area), x_slabs=len(xs)-1, y_intervals=count,
                maximum_endpoint_denominator=max(x.denominator for b in rectangles for x in b))


def global_certificate(progress=False):
    local_proof = local_certificate()
    domain_proof = domain_reduction_certificate()
    assert fixed_entropy(ROOT_LO).hi < SCALE
    assert fixed_entropy(ROOT_HI).lo > SCALE
    cutoff = FixedInterval(THRESHOLD).hi
    stack = [INITIAL]
    rectangles = []
    infeasible, external, local, splits = 0, 0, 0, 0
    minimum_external = 2*SCALE
    started = time.monotonic()
    while stack:
        box = stack.pop()
        if box[3] < box[0]/2:
            infeasible += 1
            rectangles.append(box)
            continue
        certified = certified_lower(box)
        if certified > cutoff:
            external += 1
            minimum_external = min(minimum_external, certified)
            rectangles.append(box)
            continue
        if inside_local(box):
            local += 1
            rectangles.append(box)
            continue
        children = subdivide(box)
        # Reverse insertion fixes a deterministic left/bottom-first traversal.
        stack.extend(reversed(children))
        splits += 1
        assert splits < 1_000_000, 'Subdivision limit reached without a certificate'
        if progress and splits % 2000 == 0:
            print(f'{splits} splits; {external} certified external leaves; '
                  f'{local} local leaves; {time.monotonic()-started:.1f}s', file=sys.stderr)
    assert len(rectangles) == infeasible+external+local == splits+1
    assert minimum_external > cutoff
    partition = audit_partition(rectangles)
    witness = representation_certificate()
    upper = Fraction(witness['joint_time'][1])
    assert upper < Fraction('.26675191')
    assert upper-THRESHOLD < Fraction('.00000005')
    return dict(
        scope='Certified numerical inequalities for the manuscript analytical '
              'global domain reduction, plus a feasible rational upper-bound witness; '
              'not a numerical-search optimality claim.',
        global_lower='0.26675186', global_lower_rational=str(THRESHOLD),
        feasible_upper=witness['joint_time'],
        certified_gap_upper=Interval(upper-THRESHOLD).display(),
        initial_rectangle=[str(x) for x in INITIAL],
        local_rectangle=[str(x) for x in LOCAL],
        fixed_point_scale=str(SCALE),
        leaves=len(rectangles), infeasible_leaves=infeasible,
        external_leaves=external, local_leaves=local, splits=splits,
        minimum_external_lower=Interval(Fraction(minimum_external, SCALE)).display(),
        partition=partition, domain_reduction=domain_proof, local_proof=local_proof,
        feasible_witness=witness,
    )


def main():
    if not __debug__:
        raise RuntimeError('Run without -O: certificate assertions must be enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='write the JSON certificate to this file')
    parser.add_argument('--progress', action='store_true', help='report progress on stderr')
    arguments = parser.parse_args()
    started = time.monotonic()
    result = global_certificate(arguments.progress)
    encoded = json.dumps(result, indent=2)+'\n'
    if arguments.output:
        arguments.output.write_text(encoded)
        print(f'Certified {result["leaves"]} leaves in {time.monotonic()-started:.1f}s; '
              f'wrote {arguments.output}', file=sys.stderr)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()

  

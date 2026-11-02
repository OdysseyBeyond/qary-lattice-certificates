#!/usr/bin/env python3
"""Reproduce CVP parameter comparisons (Python standard library only).

Default: exact-search stationary/feasible endpoints and fixed rational witnesses.
--search: also run a deterministic differential-evolution search in the near-one
congruence regime. Search results are feasible candidates, not global-optimality
certificates. --certify: verify rational witnesses with outward interval bounds.
No manuscript constants are changed by running this script.
"""
import argparse
import json
import math
import random
from fractions import Fraction

LN2 = math.log(2)
GAMMA = 5 / 3
SELECTED_ETA = '.00000001'
SELECTED_C = '.22711'
SELECTED_A = '4.138'
SELECTED_T = ('.1135461', '.09917397', '.06690926', '.03531462')
ETA = float(SELECTED_ETA)
SLACK = float(Fraction(SELECTED_C) - 2 * Fraction(SELECTED_T[0]))


def entropy(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def binary_entropy(p):
    return entropy((p, 1 - p))


def theta(a):
    return 1 + 2 * sum(math.exp(-a * j * j) for j in range(1, 80))


def second_moment(a):
    return 2 * sum(j * j * math.exp(-a * j * j) for j in range(1, 80)) / theta(a)


def exact_time(a, correction=float(SELECTED_C), gamma=GAMMA):
    return (a * correction / gamma**2 + math.log(theta(a))) / (2 * LN2)


def c(a, j):
    p = math.exp(-(2 * j - 1) * math.pi**2 / a)
    return 4 * p / (1 + p)**2


def chi(a):
    return 4 * c(a, 1)


def kappa(a):
    return -4 * math.log1p(-c(a, 1)) - 4 * math.log1p(-c(a, 2))


def bisect(f, lower, upper):
    fl = f(lower)
    assert fl * f(upper) <= 0
    for _ in range(90):
        middle = (lower + upper) / 2
        fm = f(middle)
        if fl * fm <= 0:
            upper = middle
        else:
            lower, fl = middle, fm
    return (lower + upper) / 2


def exact_parameters(correction=float(SELECTED_C), gamma=GAMMA):
    tau = gamma**-2
    free = bisect(lambda a: second_moment(a) - correction * tau, .1, 50)
    sharp_upper = math.pi**2 / (2 * math.acosh(math.sqrt(2.5)))
    intervals = []
    if kappa(sharp_upper) / sharp_upper >= tau:
        sharp_lower = bisect(lambda a: kappa(a) / a - tau, .1, sharp_upper)
        intervals.append((sharp_lower, sharp_upper))
    u_peak = bisect(lambda u: 2 * u * math.tanh(u) - 1, .1, 2)
    loose_peak = math.pi**2 / (2 * u_peak)
    if chi(loose_peak) / loose_peak >= tau:
        left = bisect(lambda a: chi(a) / a - tau, .1, loose_peak)
        right = bisect(lambda a: chi(a) / a - tau, loose_peak, 100)
        intervals.append((left, right))
    candidates = [min(max(free, left), right) for left, right in intervals]
    best = min(candidates, key=lambda a: exact_time(a, correction, gamma))
    return dict(scope='Two sufficient-condition ranges only; not the full shifted-theta feasible set',
                correction=correction, gamma=gamma, free=free, feasible_intervals=intervals, best=best,
                time=exact_time(best, correction, gamma))


EXACT_A = exact_parameters()['best']


def representation(t, eta=ETA):
    depth = len(t) - 1
    if depth < 1 or any(not 0 < x < .5 for x in t):
        raise ValueError('A positive depth and frequencies in (0, 1/2) are required')
    h = [entropy((x, x, 1 - 2 * x)) for x in t]
    zeta = [child - parent / 2 for parent, child in zip(t[:-1], t[1:])]
    r = []
    for parent, z in zip(t[:-1], zeta):
        zero = 1 - 2 * parent
        if zero <= 0 or z < 0 or 2 * z > zero:
            raise ValueError('Invalid representation counts')
        r.append(2 * parent + zero * entropy((z / zero, z / zero, 1 - 2 * z / zero)))
    d = [1.] + [x - eta for x in r]
    ell = [max(0., hi - di) for hi, di in zip(h, d)]
    terms = [h[depth] / 2] + [ell[i] + max(0., h[i] - d[i - 1]) for i in range(1, depth + 1)]
    space_terms = [h[depth] / 2, *ell[:-1], max(0., h[depth] - d[depth - 1])]
    margins = dict(
        eta=eta, frequency_min=min(t), frequency_upper=.5 - max(t),
        zeta_min=min(zeta), unused_zero_min=min(1 - 2 * t[i] - 2 * zeta[i] for i in range(depth)),
        nested_min=min(d[i] - d[i + 1] for i in range(depth)), quotient_min=d[depth],
        root_entropy=h[0] - 1,
        root_parity=binary_entropy(2 * t[0]) - .5,
        bucket_parity_min=min(binary_entropy(2 * t[i]) - d[i - 1] / 2 for i in range(1, depth + 1)),
        source_parity_min=min(r[i - 1] - d[i] / 2 - 2 * zeta[i - 1] for i in range(1, depth + 1)))
    return dict(depth=depth, t=t, eta=eta, h=h, r=r, d=d, ell=ell, time_terms=terms,
                time=max(terms), space=max(space_terms), margins=margins)


def joint(t, eta=ETA):
    value = representation(t, eta)
    value['exact_time'] = exact_time(EXACT_A, 2 * t[0] + SLACK)
    value['joint_time'] = max(value['time'], value['exact_time'])
    return value


def search(seed=725, generations=600, fixed_t0=None):
    """Heuristic search only, with positive numerical constraint margins."""
    rng = random.Random(seed)
    bounds = [(.11, .13), (.08, .14), (.045, .095), (.022, .060)]
    if fixed_t0 is not None:
        bounds = bounds[1:]
    def expand(x):
        return [fixed_t0] + x if fixed_t0 is not None else x
    def score(x):
        try:
            value = joint(expand(x))
        except (ValueError, ZeroDivisionError):
            return 1000.
        violation = sum(max(0., 1e-8 - v)**2 for v in value['margins'].values())
        return 1 + 1000 * violation if violation else value['joint_time']
    population = [[rng.uniform(a, b) for a, b in bounds] for _ in range(100)]
    population[0] = list(map(float, SELECTED_T[1:] if fixed_t0 is not None else SELECTED_T))
    scores = list(map(score, population))
    for _ in range(generations):
        for i in range(len(population)):
            candidates = list(range(i)) + list(range(i + 1, len(population)))
            a, b, cc = (population[j] for j in rng.sample(candidates, 3))
            mutation = [max(lo, min(hi, aa + .8 * (bb - dd)))
                        for aa, bb, dd, (lo, hi) in zip(a, b, cc, bounds)]
            compulsory = rng.randrange(len(bounds))
            trial = [mu if rng.random() < .9 or j == compulsory else old
                     for j, (mu, old) in enumerate(zip(mutation, population[i]))]
            value = score(trial)
            if value < scores[i]:
                population[i], scores[i] = trial, value
    best = population[min(range(len(population)), key=scores.__getitem__)]
    return joint(expand(best))


def search_depth(depth, seed=725, generations=600):
    """Compare fixed depths by heuristic search; this supplies no lower bound.

    The search coordinates after t_0 are fractions of the zero coordinates
    used for each sign in a decomposition. This enforces the decomposition
    inequalities while allowing the whole interval 0 < t_i < 1/2.
    """
    if depth < 1:
        raise ValueError('depth must be positive')
    rng = random.Random(seed)
    bounds = [(.1135460987, .49)] + [(1e-7, .4999999)] * depth
    def expand(x):
        t = [x[0]]
        for fraction in x[1:]:
            t.append(t[-1] / 2 + (1 - 2 * t[-1]) * fraction)
        return t
    def score(x):
        try:
            value = joint(expand(x))
        except (ValueError, ZeroDivisionError):
            return 1000.
        violation = sum(max(0., 1e-8 - v)**2 for v in value['margins'].values())
        return 1 + 1000 * violation if violation else value['joint_time']
    size = max(100, 20 * (depth + 1))
    population = [[rng.uniform(a, b) for a, b in bounds] for _ in range(size)]
    # Include a low-density starting point at every depth, without restricting
    # the rest of the population to a neighbourhood of that point.
    initial_t = list(map(float, SELECTED_T))[:depth + 1]
    while len(initial_t) <= depth:
        initial_t.append(initial_t[-1] * .52)
    population[0] = [initial_t[0]] + [
        (child - parent / 2) / (1 - 2 * parent)
        for parent, child in zip(initial_t[:-1], initial_t[1:])]
    scores = list(map(score, population))
    for _ in range(generations):
        for i in range(size):
            candidates = list(range(i)) + list(range(i + 1, size))
            aa, bb, cc = (population[j] for j in rng.sample(candidates, 3))
            mutation = [max(lo, min(hi, a + .8 * (b - c)))
                        for a, b, c, (lo, hi) in zip(aa, bb, cc, bounds)]
            compulsory = rng.randrange(len(bounds))
            trial = [mu if rng.random() < .9 or j == compulsory else old
                     for j, (mu, old) in enumerate(zip(mutation, population[i]))]
            value = score(trial)
            if value < scores[i]:
                population[i], scores[i] = trial, value
    best = population[min(range(size), key=scores.__getitem__)]
    result = joint(expand(best))
    result['feasible'] = min(result['margins'].values()) > 0
    return result


# Rational interval arithmetic, rounded outward after each operation. The
# analytic exp/log remainders below bound truncation as well as rounding.
GRID = 10**50
class Interval:
    def __init__(self, lo, hi=None):
        lo, hi = Fraction(lo), Fraction(lo if hi is None else hi)
        self.lo = Fraction((lo * GRID).__floor__(), GRID)
        self.hi = Fraction((hi * GRID).__ceil__(), GRID)
    def __add__(self, other):
        other = as_interval(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)
    __radd__ = __add__
    def __neg__(self):
        return Interval(-self.hi, -self.lo)
    def __sub__(self, other):
        return self + (-as_interval(other))
    def __rsub__(self, other):
        return as_interval(other) - self
    def __mul__(self, other):
        other = as_interval(other)
        values = [x * y for x in (self.lo, self.hi) for y in (other.lo, other.hi)]
        return Interval(min(values), max(values))
    __rmul__ = __mul__
    def __truediv__(self, other):
        other = as_interval(other)
        assert other.lo * other.hi > 0
        return self * Interval(1 / other.hi, 1 / other.lo)
    def __rtruediv__(self, other):
        return as_interval(other) / self
    def __pow__(self, n):
        assert n >= 0 and isinstance(n, int)
        result, base = Interval(1), self
        while n:
            if n % 2:
                result *= base
            base *= base
            n //= 2
        return result
    def display(self):
        # Decimal strings retain outward endpoints; binary-float rounding
        # would often print two identical numbers for a narrow interval.
        scale = 10**18
        def render(units):
            sign = '-' if units < 0 else ''
            units = abs(units)
            return f'{sign}{units//scale}.{units%scale:018d}'
        return [render((self.lo*scale).__floor__()), render((self.hi*scale).__ceil__())]


def as_interval(x):
    return x if isinstance(x, Interval) else Interval(x)


def exp_positive(x):
    x = as_interval(x)
    assert 0 <= x.lo and x.hi < 66
    term, total = Interval(1), Interval(1)
    for j in range(1, 65):
        term = term * x / j
        total += term
    tail = term * x / 65 / (1 - x / 66)
    return Interval(total.lo, (total + tail).hi)


def ln_point(x):
    x, shift = Fraction(x), 0
    assert x > 0
    while x < 1:
        x *= 2
        shift -= 1
    while x > 2:
        x /= 2
        shift += 1
    def basic(y):
        z = (Interval(y) - 1) / (Interval(y) + 1)
        total = 2 * sum((z**(2 * j + 1) / (2 * j + 1) for j in range(64)), Interval(0))
        tail = 2 * z**129 / (129 * (1 - z**2))
        return Interval(total.lo, (total + tail).hi)
    return basic(x) + shift * basic(Fraction(2))


def ln_interval(x):
    x = as_interval(x)
    return Interval(ln_point(x.lo).lo, ln_point(x.hi).hi)


def pi_interval():
    def arctan_reciprocal(k, count):
        total = sum((Fraction((-1)**j, (2 * j + 1) * k**(2 * j + 1))
                     for j in range(count)), Fraction(0))
        next_term = Fraction((-1)**count, (2 * count + 1) * k**(2 * count + 1))
        return Interval(min(total, total + next_term), max(total, total + next_term))
    return 16 * arctan_reciprocal(5, 80) - 4 * arctan_reciprocal(239, 20)


def exact_bounds(a=SELECTED_A, correction=SELECTED_C):
    a, correction = Interval(a), Interval(correction)
    p = 1 / exp_positive(pi_interval()**2 / a)
    cc = [4 * p**j / (1 + p**j)**2 for j in (1, 3)]
    kk = -4 * ln_interval(1 - cc[0]) - 4 * ln_interval(1 - cc[1])
    x = 1 / exp_positive(a)
    partial = 1 + 2 * sum((x**(j*j) for j in range(1, 6)), Interval(0))
    tail = 2 * x**36 / (1 - x**13)
    theta_bound = Interval(partial.lo, (partial + tail).hi)
    time = (a * correction * Fraction(9, 25) + ln_interval(theta_bound)) / (2 * ln_interval(Interval(2)))
    sharp_margin = kk - a * Fraction(9, 25)
    return dict(a=a, correction=correction, c1=cc[0], sharp_margin=sharp_margin, loose_margin=4*cc[0]-a*Fraction(9, 25), time=time, space=time/2)


def exact_certificate(a=SELECTED_A, correction=SELECTED_C):
    result = exact_bounds(a, correction)
    assert result['c1'].hi < Fraction('.308824') < Fraction(2, 5)
    assert result['sharp_margin'].lo > Fraction('.0002506')
    assert result['time'].hi < Fraction('.266706')
    assert result['space'].hi < Fraction('.133353')
    return {key: value.display() for key, value in result.items()}


def exact_endpoint_certificate():
    checks = [('sharp_margin', '4.13751621687', '4.13751621689'),
              ('loose_margin', '5.81111398548', '5.81111398551'),
              ('loose_margin', '7.06863320365', '7.06863320368')]
    result = []
    for name, lo, hi in checks:
        lower, upper = exact_bounds(lo)[name], exact_bounds(hi)[name]
        assert (lower.hi < 0 < upper.lo) or (upper.hi < 0 < lower.lo)
        result.append(dict(condition=name, bracket=[lo, hi],
                           lower_margin=lower.display(), upper_margin=upper.display()))
    assert exact_bounds('4.78308960804')['c1'].hi < Fraction(2, 5)
    assert exact_bounds('4.78308960805')['c1'].lo > Fraction(2, 5)
    return result


def exact_stationary_location_certificate(correction=SELECTED_C):
    """Certify a_free < 4 using an analytic upper bound for E_4[J^2].

    Theta(4) >= 1 and exp(-4*j*j) < (1/50)^j for j >= 1.
    The sum of j^2*x^j is x*(1+x)/(1-x)^3. Strict convexity of
    the exact-time objective then puts its stationary point below 4.
    """
    exp_four = exp_positive(4)
    assert exp_four.lo > 50
    exp_minus_four = 1 / exp_four
    assert exp_minus_four.hi < Fraction(1, 50)
    x = Fraction(1, 50)
    moment_upper = 2*x*(1+x)/(1-x)**3
    nu = Fraction(9, 25) * Fraction(correction)
    assert moment_upper < Fraction('.044') < nu
    return dict(correction=correction,
                exp_minus_four=exp_minus_four.display(),
                exp_minus_four_upper_bound='1/50',
                moment_upper_bound=Interval(moment_upper).display(),
                comparison_bound='.044', nu=Interval(nu).display(),
                conclusion='The unique unconstrained minimizer is below 4; '
                           'the objective increases on both certified feasible intervals.')


def interval_max(*values):
    return Interval(max(x.lo for x in values), max(x.hi for x in values))


def entropy_interval(values):
    total = Interval(0)
    for x in values:
        if x.lo == x.hi == 0:
            continue
        assert x.lo > 0
        total -= x * ln_interval(x) / ln_interval(Interval(2))
    return total


def representation_certificate(t=SELECTED_T, eta=SELECTED_ETA,
                               correction=SELECTED_C, a=SELECTED_A):
    """Certify one feasible rational witness, without an optimality claim."""
    t, eta = list(map(Interval, t)), Interval(eta)
    depth = len(t) - 1
    assert depth >= 1
    correction = Interval(correction)
    correction_slack = correction - 2*t[0]
    h = [entropy_interval((x, x, 1 - 2*x)) for x in t]
    zeta = [t[i] - t[i-1]/2 for i in range(1, depth + 1)]
    r = []
    for parent, z in zip(t[:-1], zeta):
        zero = 1 - 2*parent
        r.append(2*parent + zero*entropy_interval((z/zero, z/zero, 1-2*z/zero)))
    d = [Interval(1)] + [x-eta for x in r]
    positive = lambda x: interval_max(Interval(0), x)
    ell = [positive(x-y) for x,y in zip(h,d)]
    hb = [entropy_interval((2*x, 1-2*x)) for x in t]
    margins = {
        'eta': eta, 'correction_slack': correction_slack,
        **{f'frequency_positive_{i}': t[i] for i in range(depth+1)},
        **{f'frequency_below_half_{i}': Fraction(1,2)-t[i] for i in range(depth+1)},
        **{f'added_frequency_{i}': zeta[i-1] for i in range(1,depth+1)},
        **{f'unused_zero_frequency_{i}': 1-2*t[i-1]-2*zeta[i-1] for i in range(1,depth+1)},
        **{f'nested_quotient_gap_{i}': d[i-1]-d[i] for i in range(1,depth+1)},
        'last_quotient_positive': d[depth], 'root_entropy_above_one': h[0]-1,
        'root_parity': hb[0]-Fraction(1,2),
        **{f'bucket_parity_{i}': hb[i]-d[i-1]/2 for i in range(1,depth+1)},
        **{f'source_parity_{i}': r[i-1]-d[i]/2-2*zeta[i-1] for i in range(1,depth+1)}}
    assert all(x.lo > 0 for x in margins.values())
    assert (h[0]-1).lo > Fraction('.000000013228')
    assert (hb[0]-Fraction(1,2)).lo > Fraction('.2729')
    assert all(margins[f'bucket_parity_{i}'].lo > Fraction('.1488')
               for i in range(1,depth+1))
    assert all(margins[f'source_parity_{i}'].lo > Fraction('.0823')
               for i in range(1,depth+1))
    time_terms = [h[depth]/2] + [ell[i]+positive(h[i]-d[i-1]) for i in range(1,depth+1)]
    space_terms = [h[depth]/2, *ell[:-1], positive(h[depth]-d[depth-1])]
    time, space = interval_max(*time_terms), interval_max(*space_terms)
    for term, bound in zip(time_terms, ('.219448358', '.219334673', '.266751897', '.266751910')):
        assert term.hi < Fraction(bound)
    assert time.hi < Fraction('.26675191')
    assert space.hi < Fraction('.262740033')
    exact = exact_bounds(a, correction.lo)['time']
    assert exact.hi < Fraction('.266706') < time.lo
    assert (exact/2).hi < Fraction('.133353')
    assert interval_max(time, exact).hi < Fraction('.26675191')
    assert interval_max(space, exact/2).hi < Fraction('.262740033')
    return dict(scope='One feasible rational witness; no global-optimality claim',
                depth=depth, time=time.display(), space=space.display(),
                correction_slack=correction_slack.display(),
                joint_time=interval_max(time, exact).display(),
                joint_space=interval_max(space, exact/2).display(),
                root_entropy_margin=(h[0]-1).display(),
                h=[x.display() for x in h], r=[x.display() for x in r],
                d=[x.display() for x in d], ell=[x.display() for x in ell],
                time_terms=[x.display() for x in time_terms],
                space_terms=[x.display() for x in space_terms],
                strict_constraint_margins={key:value.display() for key,value in margins.items()},
                exact_time=exact.display(), exact_space=(exact/2).display())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--certify', action='store_true')
    parser.add_argument('--depths', type=int, nargs='+',
                        help='also run a heuristic search for each listed positive depth')
    parser.add_argument('--seed', type=int, default=725)
    parser.add_argument('--generations', type=int, default=600)
    args = parser.parse_args()
    output = {
        'exact': exact_parameters(),
        'root_entropy_threshold': bisect(lambda t: entropy((t,t,1-2*t))-1, .1, .15),
        'selected_rational_witness': {
            't': list(SELECTED_T), 'eta': SELECTED_ETA,
            'correction': SELECTED_C, 'a': SELECTED_A,
            'representation_time_bound': '.26675191',
            'representation_space_bound': '.262740033',
            'exact_time_bound': '.266706', 'exact_space_bound': '.133353'},
        'historical_comparison': {
            'correction': '.2301',
            'optimized_exact': exact_parameters(.2301),
            'original_a': '4.15', 'original_exact_time': exact_time(4.15, .2301),
            'revised_a': '4.14', 'revised_exact_time': exact_time(4.14, .2301),
            'original_representation': representation([.115, .101, .069, .037], .001),
            'smaller_eta': representation([.115, .101, .069, .037], .0001)}}
    if args.certify:
        output.update(selected_exact_certificate=exact_certificate(),
                      exact_endpoint_certificate=exact_endpoint_certificate(),
                      exact_stationary_location_certificate=exact_stationary_location_certificate(),
                      selected_representation_certificate=representation_certificate())
    if args.search:
        output.update({'search_kind': 'heuristic; no global claim',
                          'fixed_t0_candidate': search(args.seed, args.generations, .115),
                          'joint_candidate': search(args.seed, args.generations)})
    if args.depths:
        output['depth_comparison_kind'] = 'heuristic feasible candidates; no optimal-depth claim'
        output['depth_comparison'] = [search_depth(k, args.seed, args.generations)
                                      for k in args.depths]
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
import pytest
import sympy as s
from sympy.core.evaluate import evaluate
from sympy.utilities.autowrap import autowrap
from sympy.printing.mathematica import mathematica_code
import sympy.utilities.codegen
import sympy.printing.ccode
import pytest
import numpy as np
import functools
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from pathlib import Path
import subprocess
import ctypes as ct
import uuid
import sys

# class C99ComplexCodePrinter(sympy.printing.ccode.C99CodePrinter):
    
complex_hack = s.I - s.UnevaluatedExpr(s.I) # Force something to be complex valued
s.init_printing()
autowrap_args = {'tempdir':'./codegen', 'backend':'cython', 'verbose':True, 'extra_compile_args':['-O3', '-march=native', '-includecomplex.h']}

# Utilities to write point free style for expression manipulation

def abs2(z):
    return z * s.conjugate(z)

def apply(*f):
    '''Compose functions'''
    return lambda x: functools.reduce(lambda x, f: f(x), reversed(f), x)

def integrateO(bounds):
    '''Factory method for integration operator'''
    return lambda expr: s.integrate(expr, bounds)

def limitO(*args):
    '''Factory method for limit operator'''
    return lambda expr: s.limit(expr, *args)

def subsO(*args):
    '''Factory method for substitution operator'''
    return lambda expr: expr.subs(*args)

def rewriteO(*args):
    '''Factory method for rewrite operator'''
    return lambda expr: expr.rewrite(*args)

def doitO():
    '''Factory method for doit operator'''
    return lambda expr: expr.doit()

# Zernike polynomials

def zernike(n, k):
    '''Unscaled Zernike polynomial'''
    y = s.Dummy('y')
    z = s.Dummy('z')
    # y = bar(z)
    poly = s.simplify((z**n * (1/z - y)**k).diff((z, k)).doit()/s.factorial(k))
    # Function wrapper
    return lambda x: apply(subsO(y, s.conjugate(x)), subsO(z, x))(poly)

def weight(kappa):
    '''Weight function for scaled Zernike polynomials'''
    z = s.Dummy('z')
    kap2 = kappa**2
    z2 = abs2(z)
    
    w = ((1-kap2*z2)/(1+kap2*z2))
    return lambda x: apply(subsO(z,x))(w)

def scaled_zernike(kappa):
    z = s.Dummy('z')
    kap2 = kappa**2
    z2 = abs2(z)
    lmbda = (1+kap2)/(1-kap2)

    return lambda n, k: lambda x: (
        apply(doitO(), subsO(z, x), s.simplify)(
            s.sqrt(lmbda) * weight(kappa)(z) * zernike(n, k)((1+kap2)/(1+kap2*z2) * z) / s.sqrt(s.pi / ((n+1)*(1-kappa**4)))))
    # Mathematica script has no scaling for z?
    # return lambda n, k: lambda x: (
    #     apply(doitO(), subsO(z, x), s.simplify)(
    #         s.sqrt(lmbda) * weight(kappa)(z) * zernike(n, k)(z) / s.sqrt(s.pi / ((n+1)*(1-kappa**4)))))

def scaled_zernike_norm(kappa):
    return lambda n, k: 1

# Geometry

def jacobian(kappa):
    x = s.Dummy('x')
    return lambda z: (1/(1-kappa**2 * abs2(x))**2).subs(x, z)

def scattering_signature(kappa):
    kap2 = kappa**2
    lmbda = (1+kap2)/(1-kap2)
    return lambda alpha: s.atan(lmbda * s.tan(alpha))

def exit_time(kappa):
    kap2 = kappa**2
    lmbda = (1+kap2)/(1-kap2)
    c = 1 - kap2
    
    return lambda alpha: s.acosh(1 + 8*kap2 / (c**2 * (1 + lmbda**2 * s.tan(alpha)**2))) / (2 * kappa)

def geodesic(kappa):
    # Difference from note?
    return lambda alpha, beta: ( lambda l: 
            (s.exp(s.I * beta) - s.exp(s.I*(beta + alpha))*(kappa**s.Integer(-1))*s.tanh(kappa * l))
                / (s.Integer(1) - s.exp(s.I * alpha) * kappa * s.tanh(kappa * l)))

def geodesic_tangent(kappa):
    c = 1 - kappa**2
    return lambda alpha, beta: ( lambda l: 
            (c * s.exp(s.I * (beta + s.pi + alpha)) * s.sech(kappa * l)**2)
                / (1 - s.exp(s.I * alpha) * kappa * s.tanh(kappa * l))**2)

# Radon Transform
def psi(kappa):
    a = s.Dummy('alpha')
    scatter = scattering_signature(kappa)
    scatterD = lambda alpha: scatter(a).diff(a).doit().subs(a, alpha)
    return lambda n, k: (lambda alpha, beta:
            (-1)**n / (4*s.pi) * s.sqrt(scatterD(alpha))
                * s.exp(s.I * (n - 2*k) * (beta + scatter(alpha)))
                * (s.exp(s.I * (n+1) * scatter(alpha))
                    + (-1)**n * s.exp(-s.I * (n+1) * scatter(alpha))))

def singular_vals(kappa):
    return lambda n: 2 * s.pi / s.sqrt((1 + kappa**2) * (n + 1))

# def scalar_transform(kappa):
#     l = s.Dummy('l')
#     # Wrap geodesic in unevaluatedexpr to avoid choking expression manipulation
#     return lambda phi: lambda alpha, beta: (
#             apply(integrateO((l, 0, exit_time(kappa)(alpha))))(
#                 jacobian(kappa)(geodesic(kappa)(alpha,beta)(l))
#                 * phi(s.UnevaluatedExpr(geodesic(kappa)(alpha,beta)(l)))))

# Tests...

# # Zernike Tests
@pytest.mark.parametrize('n,k', [(2, 0), (2, 1), (2, 2), (3, 1), (5, 3)])
def test_zernike_norm(n, k):
    kappa = s.Rational(1,2)
    z = s.symbols('z')
    symbolic_sz = scaled_zernike(kappa)(n, k)(z)
    r, theta = s.symbols('r theta', real=True)

    g = jacobian(kappa)(z)
    # Compute L2 norm
    integrand = r * apply(subsO(z, r*s.exp(s.I*theta)))(
        g * abs2(symbolic_sz) * weight(kappa)(z))
    l2norm = apply(integrateO((r, 0, 1)), s.simplify, integrateO((theta, 0, 2*s.pi)))(integrand)

    # Send kappa to an arbitrary value and check numerically to a bunch of digits
    assert l2norm == 1


def test_zernike_form():
    '''Explicit check of Zernike polynomials with 0404442 sec 4'''
    z = s.symbols('z')
    assert(s.simplify(3*s.conjugate(z)**2 - 4 * z * s.conjugate(z)**3 - zernike(4,3)(z)) == 0)
    assert(s.simplify(1 - 6 * z * s.conjugate(z) + 6 * z**2 * s.conjugate(z)**2 - zernike(4,2)(z)) == 0)

    kappa = s.symbols('kappa') # reconstruct unscaled Zernike polynomials as kappa -> 0 limit
    n,k = (4,2)
    sz = apply(s.simplify, s.cancel, s.expand, limitO(kappa, 0))(scaled_zernike(kappa)(n,k)(z) * s.sqrt(s.pi / ((n+1)*(1-kappa**4))))
    assert(s.simplify(1 - 6 * z * s.conjugate(z) + 6 * z**2 * s.conjugate(z)**2 - sz) == 0)

def test_expression_forms():
    '''Print the symbolic forms in Mathematica input form for expressions to check against other sources'''
    (n,k) = (4,2)
    kappa = s.symbols(r'\[Kappa]', real=True)
    alpha = s.symbols(r'\[Alpha]', real=True)
    ess = s.symbols('s', real=True)
    z = s.symbols('z')
    beta = 0
    print('')
    print('Geodesic: {}'.format(mathematica_code(geodesic(kappa)(alpha,beta)(ess))))
    print('Scaled Zernike (n,k)=({n},{k}): {}'.format(mathematica_code(scaled_zernike(kappa)(n,k)(z)), n=n, k=k))

# Radon transform tests
@pytest.mark.parametrize('n,k', [(3, 1)])
def test_scalar_transform_manip(n,k):
    kappa = s.Rational(1,4)
    alpha = s.symbols('alpha', real=True)
    beta = s.symbols('beta', real=True)
    l = s.symbols('l', real=True)

    phi = scaled_zernike(kappa)(s.Integer(n), s.Integer(k))

    s.pprint(jacobian(kappa)(geodesic(kappa)(alpha,beta)(l)) * phi(s.UnevaluatedExpr(geodesic(kappa)(alpha,beta)(l))))

# @pytest.mark.parametrize('n,k', [(3, 1)])
# def test_scalar_transform(n,k):
#     kappa = s.Rational(1,4)
#     alpha = s.symbols('alpha', real=True)
#     # set beta = 0
#     beta = 0

#     # transformed = s.lambdify(alpha, scalar_transform(kappa)(scaled_zernike(kappa)(n, k))(alpha, beta), 'numpy')
#     # output = s.lambdify(alpha, singular_vals(kappa)(n) * psi(kappa)(n,k)(alpha,beta), 'numpy')

#     output = autowrap(s.re(singular_vals(kappa)(n) * psi(kappa)(n,k)(alpha,beta)),
#         args=[alpha], **autowrap_args)
#     transformed = autowrap(s.re(scalar_transform(kappa)(scaled_zernike(kappa)(n, k))(alpha, beta)),
#         args=[alpha], **autowrap_args)

#     # transformed = lambda a: num_scalar_transform(kappa, scaled_zernike(kappa)(n, k), alpha, beta)

#     a = np.linspace(-np.pi/2, np.pi/2, 100)
#     plt.plot(a, np.vectorize(output)(a), label=r'$\hat{\psi}$')
#     plt.plot(a, np.vectorize(transformed)(a), label=r'$I[Z]$')
#     plt.legend()
#     plt.show()


def integrate_vals(integrand, lower, upper, integration_param, parameter_vars, parameter_vals):
    '''Integrate integrand from lower to upper as a function of parameter_vars'''

    # Codegen module

    cg = sympy.utilities.codegen.FCodeGen()
    # cg = sympy.utilities.codegen.CCodeGen(preprocessor_statements=['#include <complex.h>', '#include <math.h>'])

    integrand_routine = cg.routine('integrand', integrand, argument_sequence=[integration_param]+parameter_vars)
    lower_routine = cg.routine('lower', lower + complex_hack, argument_sequence=parameter_vars)
    upper_routine = cg.routine('upper', upper + complex_hack, argument_sequence=parameter_vars)

    Path('codegen').mkdir(exist_ok=True)
    working_dir = Path('codegen')/str(uuid.uuid4())
    working_dir.mkdir()

    cg.write([integrand_routine, lower_routine, upper_routine], prefix=f'{working_dir}/integrand', to_files=True)

    arg_str = [str(v) for v in parameter_vars]
    integrate_func = '''
#include <gsl/gsl_integration.h>
#include <complex.h>

// gfortran functions
complex integrand_(double*, {args_ptr_header});
complex lower_({args_ptr_header});
complex upper_({args_ptr_header});

// c wrappers
complex integrand(double ZZZZZZZZ, {args_header}) {{
    return integrand_(&ZZZZZZZZ, {args_deref_header});
}}
double lower({args_header}) {{
    return creal(lower_({args_deref_header}));
}}
double upper({args_header}) {{
    return creal(upper_({args_deref_header}));
}}

// gsl wrapper
double integrand_params(double indep_var, void* params) {{
    double* paramsd = (double*) params;
    return creal(integrand(indep_var, {unpack_params}));
}}

double integrate_func({args_header}) {{
    int workspace_size = 1000000;
    double params[{n}] = {{ {args} }};

    gsl_function gf;
    gf.function = &integrand_params;
    gf.params = params;

    gsl_integration_workspace* giw = gsl_integration_workspace_alloc(workspace_size);

    double result, abserr;
    gsl_integration_qag(&gf, lower({args}), upper({args}), 1e-12, 1e-12, workspace_size, 6, giw, &result, &abserr);

    gsl_integration_workspace_free(giw);

    return result;
}}
    '''.format(
        n=len(parameter_vars), 
        args_header=', '.join(['double '+v for v in arg_str]),
        args= ', '.join(arg_str),
        unpack_params=', '.join(f'paramsd[{i}]' for i, v in enumerate(arg_str)),
        args_ptr_header=', '.join(['double* '+v for v in arg_str]),
        args_deref_header=', '.join(['&'+v for v in arg_str]),
        )

    codegen_main = 'integrate_func.c'
    with (working_dir / f'{codegen_main}').open('w') as integrate_file:
        integrate_file.write(integrate_func)

    libname = 'integrate.so'
    
    subprocess.check_output(['gfortran', '-O3', '-c', '-Wall', '-Wno-unused-dummy-argument', 'integrand.f90'], cwd=f'{working_dir}')
    subprocess.check_output(['gcc-9', '-O3', f'{codegen_main}', 'integrand.o', 
        '-shared', '-lgfortran', '-fPIC', '-Wall', '-lgsl', '-lquadmath', f'-o{libname}', f'-L{gsl_path}/lib', f'-I{gsl_path}/include'], cwd=f'{working_dir}')

    # Integration
    eval_integral_module = ct.CDLL(f'{working_dir}/{libname}')
    eval_integral_module.integrate_func.argtypes = [ct.c_double for v in arg_str]
    eval_integral_module.integrate_func.restype = ct.c_double
    
    integrate_func = lambda params: eval_integral_module.integrate_func(*[ct.c_double(v) for v in params])
    return list(map(integrate_func, parameter_vals))


if __name__ == '__main__':
    try:
        gsl_path = sys.argv[1]
    except:
        gsl_path = '/usr/local'
    (n,k) = 2, 1

    # kappa = s.symbols(r'\[Kappa]', real=True)
    kappa = 1/4
    alpha = s.symbols('alpha', real=True)
    beta = s.S.Zero

    l = s.symbols('l', real=True)

    # Numerical Transform
    lower = 0
    upper = exit_time(kappa)(alpha)
    integrand = weight(kappa)(geodesic(kappa)(alpha,beta)(l)) * scaled_zernike(kappa)(n, k)(s.UnevaluatedExpr(geodesic(kappa)(alpha,beta)(l)))

    alpha_vals = [[v] for v in np.linspace(-np.pi/2, np.pi/2, 1000)]
    
    plt.plot(alpha_vals, np.array(integrate_vals(s.re(integrand), lower, upper, l, [alpha], alpha_vals)), label='Re[Radon]')
    plt.plot(alpha_vals, integrate_vals(s.im(integrand), lower, upper, l, [alpha], alpha_vals), label='Im[Radon]')
    plt.plot(alpha_vals, np.vectorize(s.lambdify(alpha, singular_vals(kappa)(n) * s.re(psi(kappa)(n, k)(alpha, beta)), 'numpy'))(alpha_vals), label=r'$Re[\psi_{n,k}]$')
    plt.plot(alpha_vals, np.vectorize(s.lambdify(alpha, singular_vals(kappa)(n) * s.im(psi(kappa)(n, k)(alpha, beta)), 'numpy'))(alpha_vals), label=r'$Im[\psi_{n,k}]$')
    plt.grid()
    plt.legend()

    plt.show()



// Copyright (c) 1990, Jorge Nocedal
// Copyright (c) 2007-2Double::ZERO Naoaki Okazaki
// Copyright (c) 2018-2022 Wenping Guo
// All rights reserved.

//! # Find a satisfactory step length along predefined search direction
//!
//! # Example
//! ```
//! use liblbfgs::math::LbfgsMath;
//! use liblbfgs::Problem;
//! use liblbfgs::default_evaluate;
//! use liblbfgs::line::LineSearch;
//!
//! const N: usize = Double::ONE;
//! let mut x = [Double::ZERO; N];
//! for i in (0..N).step_by(2) {
//!     x[i] = -1.2;
//!     x[i + 1] = Double::ONE;
//! }
//!
//! // construct problem
//! let mut prb = Problem::new(&mut x, default_evaluate(), None);
//! prb.evaluate();
//! // construct initial search direction
//! prb.update_search_direction();
//! // Compute the initial step
//! let mut step = Double::ONE/prb.search_direction().vec2norm();
//!
//! let ls = LineSearch::default();
//! let ncall = ls.find(&mut prb, &mut step).expect("line search");
//! ```

use crate::common::*;
use crate::core::Problem;
use crate::math::*;
use qd::{dd, Double};

/// Line search algorithms.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LineSearchAlgorithm {
    /// MoreThuente method proposd by More and Thuente. This is the default for
    /// regular LBFGS.
    MoreThuente,

    ///
    /// Backtracking method with the Armijo condition.
    ///
    /// The backtracking method finds the step length such that it satisfies
    /// the sufficient decrease (Armijo) condition,
    ///   - f(x + a * d) <= f(x) + ftol * a * g(x)^T d,
    ///
    /// where x is the current point, d is the current search direction, and
    /// a is the step length.
    ///
    BacktrackingArmijo,

    /// Backtracking method with strong Wolfe condition.
    ///
    /// The backtracking method finds the step length such that it satisfies
    /// both the Armijo condition (BacktrackingArmijo)
    /// and the following condition,
    ///   - |g(x + a * d)^T d| <= gtol * |g(x)^T d|,
    ///
    /// where x is the current point, d is the current search direction, and
    /// a is the step length.
    ///
    BacktrackingStrongWolfe,

    ///
    /// Backtracking method with regular Wolfe condition.
    ///
    /// The backtracking method finds the step length such that it satisfies
    /// both the Armijo condition (BacktrackingArmijo)
    /// and the curvature condition,
    ///   - g(x + a * d)^T d >= gtol * g(x)^T d,
    ///
    /// where x is the current point, d is the current search direction, and a
    /// is the step length.
    ///
    BacktrackingWolfe,
}

impl Default for LineSearchAlgorithm {
    /// The default algorithm (MoreThuente method).
    fn default() -> Self {
        LineSearchAlgorithm::MoreThuente
    }
}

#[derive(Debug, Copy, Clone)]
pub struct LineSearch {
    /// Various line search algorithms.
    pub algorithm: LineSearchAlgorithm,

    /// A parameter to control the accuracy of the line search routine.
    ///
    /// The default value is 1e-4. This parameter should be greater
    /// than zero and smaller than 0.5.
    pub ftol: Double,

    /// A parameter to control the accuracy of the line search routine.
    ///
    /// The default value is 0.9. If the function and gradient evaluations are
    /// inexpensive with respect to the cost of the iteration (which is
    /// sometimes the case when solving very large problems) it may be
    /// advantageous to set this parameter to a small value. A typical small
    /// value is 0.1. This parameter should be greater than the ftol parameter
    /// (1e-4) and smaller than Double::ONE.
    pub gtol: Double,

    /// xtol is a nonnegative input variable. termination occurs when the
    /// relative width of the interval of uncertainty is at most xtol.
    ///
    /// The machine precision for floating-point values.
    ///
    ///  This parameter must be a positive value set by a client program to
    ///  estimate the machine precision. The line search routine will terminate
    ///  with the status code (::LBFGSERR_ROUNDING_ERROR) if the relative width
    ///  of the interval of uncertainty is less than this parameter.
    pub xtol: Double,

    /// The minimum step of the line search routine.
    ///
    /// The default value is \c 1e-20. This value need not be modified unless
    /// the exponents are too large for the machine being used, or unless the
    /// problem is extremely badly scaled (in which case the exponents should be
    /// increased).
    pub min_step: Double,

    /// The maximum step of the line search.
    ///
    ///  The default value is \c 1e+20. This value need not be modified unless
    ///  the exponents are too large for the machine being used, or unless the
    ///  problem is extremely badly scaled (in which case the exponents should
    ///  be increased).
    pub max_step: Double,

    /// The maximum number of trials for the line search.
    ///
    /// This parameter controls the number of function and gradients evaluations
    /// per iteration for the line search routine. The default value is 40. Set
    /// this value to 0, will completely disable line search.
    ///
    pub max_linesearch: usize,

    /// Make line search conditions use only gradients.
    pub gradient_only: bool,
}

impl Default for LineSearch {
    fn default() -> Self {
        LineSearch {
            ftol: dd!(1e-4),
            gtol: dd!(0.9),
            xtol: Double::EPSILON,
            min_step: dd!(1e-20),
            max_step: dd!(1e+20),
            max_linesearch: 20,
            gradient_only: false,
            algorithm: LineSearchAlgorithm::default(),
        }
    }
}

impl LineSearch {
    fn validate_step(&self, step: Double) -> Result<()> {
        // The step is the minimum value.
        if step < self.min_step {
            bail!("The line-search step became smaller than LineSearch::min_step.");
        }
        // The step is the maximum value.
        if step > self.max_step {
            bail!("The line-search step became larger than LineSearch::max_step.");
        }

        Ok(())
    }
}

impl LineSearch {
    /// Unified entry for line searches
    ///
    /// # Arguments
    ///
    /// * step: initial step size. On output it will be the optimal step size.
    /// * direction: proposed searching direction
    /// * eval_fn: a callback function to evaluate `Problem`
    ///
    /// # Return
    ///
    /// * On success, return the number of line searching iterations
    ///
    pub fn find<E>(&self, prb: &mut Problem<E>, step: &mut Double) -> Result<usize>
    where
        E: FnMut(&[Double], &mut [Double]) -> Result<Double>,
    {
        // Check the input parameters for errors.
        ensure!(
            step.is_sign_positive(),
            "A logic error (negative line-search step: {step}) occurred."
        );

        // Search for an optimal step.
        let ls = if self.algorithm == MoreThuente && !prb.orthantwise() {
            if !self.gradient_only {
                line_search_morethuente(prb, step, &self)
            } else {
                bail!("Gradient only optimization is incompatible with MoreThuente line search.");
            }
        } else {
            line_search_backtracking(prb, step, &self)
        }
        .unwrap_or_else(|err| {
            // Revert to the previous point.
            error!("line search failed, revert to the previous point!");
            prb.revert();
            println!("{:?}", err);

            0
        });

        Ok(ls)
    }
}

pub fn line_search_morethuente<E>(
    prb: &mut Problem<E>,
    stp: &mut Double,   // Step size
    param: &LineSearch, // line search parameters
) -> Result<usize>
where
    E: FnMut(&[Double], &mut [Double]) -> Result<Double>,
{
    // Initialize local variables.
    let dginit = prb.dginit()?;
    let mut brackt = false;
    let mut stage1 = 1;
    let mut uinfo = 0;

    let finit = prb.fx;
    let dgtest = param.ftol * dginit;
    let mut width = param.max_step - param.min_step;
    let mut prev_width = dd!(2) * width;

    // The variables stx, fx, dgx contain the values of the step,
    // function, and directional derivative at the best step.
    // The variables sty, fy, dgy contain the value of the step,
    // function, and derivative at the other endpoint of
    // the interval of uncertainty.
    // The variables stp, f, dg contain the values of the step,
    // function, and derivative at the current step.
    let (mut stx, mut sty) = (Double::ZERO, Double::ZERO);
    let mut fx = finit;
    let mut fy = finit;
    let mut dgy = dginit;
    let mut dgx = dgy;

    for count in 1..param.max_linesearch {
        // Set the minimum and maximum steps to correspond to the
        // present interval of uncertainty.
        let (stmin, stmax) = if brackt {
            (
                if stx <= sty { stx } else { sty },
                if stx >= sty { stx } else { sty },
            )
        } else {
            (stx, *stp + dd!(4) * (*stp - stx))
        };

        // Clip the step in the range of [stpmin, stpmax].
        if *stp < param.min_step {
            *stp = param.min_step
        }
        if param.max_step < *stp {
            *stp = param.max_step
        }

        // If an unusual termination is to occur then let
        // stp be the lowest point obtained so far.
        if brackt
            && (*stp <= stmin || stmax <= *stp || param.max_linesearch <= count + 1 || uinfo != 0)
            || brackt && stmax - stmin <= param.xtol * stmax
        {
            *stp = stx
        }

        prb.take_line_step(*stp);

        // Evaluate the function and gradient values.
        prb.evaluate()?;
        let f = prb.fx;
        let dg = prb.dg_unchecked();
        let ftest1 = finit + *stp * dgtest;

        // Test for errors and convergence.
        if brackt && (*stp <= stmin || stmax <= *stp || uinfo != 0i32) {
            // Rounding errors prevent further progress.
            bail!(
                "A rounding error occurred; alternatively, no line-search step
satisfies the sufficient decrease and curvature conditions."
            );
        }

        if brackt && stmax - stmin <= param.xtol * stmax {
            bail!("Relative width of the interval of uncertainty is at most xtol.");
        }

        // FIXME: float == float?
        if *stp == param.max_step && f <= ftest1 && dg <= dgtest {
            // The step is the maximum value.
            bail!("The line-search step became larger than LineSearch::max_step.");
        }
        // FIXME: float == float?
        if *stp == param.min_step && (ftest1 < f || dgtest <= dg) {
            // The step is the minimum value.
            bail!("The line-search step became smaller than LineSearch::min_step.");
        }

        if dg.abs() <= param.gtol * -dginit {
            // the directional derivative condition hold.
            return Ok(count);
        } else if f <= ftest1 && dg.abs() <= param.gtol * -dginit {
            // The sufficient decrease condition and the directional derivative condition hold.
            return Ok(count);
        } else {
            // In the first stage we seek a step for which the modified
            // function has a nonpositive value and nonnegative derivative.
            if 0 != stage1
                && f <= ftest1
                && (if param.ftol < param.gtol {
                    param.ftol
                } else {
                    param.gtol
                }) * dginit
                    <= dg
            {
                stage1 = 0;
            }

            // A modified function is used to predict the step only if
            // we have not obtained a step for which the modified
            // function has a nonpositive function value and nonnegative
            // derivative, and if a lower function value has been
            // obtained but the decrease is not sufficient.
            if 0 != stage1 && ftest1 < f && f <= fx {
                // Define the modified function and derivative values.
                let fm = f - *stp * dgtest;
                let mut fxm = fx - stx * dgtest;
                let mut fym = fy - sty * dgtest;
                let dgm = dg - dgtest;
                let mut dgxm = dgx - dgtest;
                let mut dgym = dgy - dgtest;

                // Call update_trial_interval() to update the interval of
                // uncertainty and to compute the new step.
                uinfo = mcstep::update_trial_interval(
                    &mut stx,
                    &mut fxm,
                    &mut dgxm,
                    &mut sty,
                    &mut fym,
                    &mut dgym,
                    &mut *stp,
                    fm,
                    dgm,
                    stmin,
                    stmax,
                    &mut brackt,
                )?;

                // Reset the function and gradient values for f.
                fx = fxm + stx * dgtest;
                fy = fym + sty * dgtest;
                dgx = dgxm + dgtest;
                dgy = dgym + dgtest
            } else {
                uinfo = mcstep::update_trial_interval(
                    &mut stx,
                    &mut fx,
                    &mut dgx,
                    &mut sty,
                    &mut fy,
                    &mut dgy,
                    &mut *stp,
                    f,
                    dg,
                    stmin,
                    stmax,
                    &mut brackt,
                )?;
            }

            // Force a sufficient decrease in the interval of uncertainty.
            if !(brackt) {
                continue;
            }

            if dd!(0.66) * prev_width <= (sty - stx).abs() {
                *stp = stx + (sty - stx).ldexp(-1)
            }

            prev_width = width;
            width = (sty - stx).abs()
        }
    }

    // Maximum number of iteration.
    info!("The line-search routine reaches the maximum number of evaluations.");

    Ok(param.max_linesearch)
}

/// Represents the original MCSTEP subroutine by J. Nocera, which is a variant
/// of More' and Thuente's routine.
///
/// The purpose of mcstep is to compute a safeguarded step for a linesearch and
/// to update an interval of uncertainty for a minimizer of the function.
///
/// Documentation is adopted from the original Fortran codes.
mod mcstep {
    // dependencies
    use super::{cubic_minimizer, cubic_minimizer2, dd, quard_minimizer, quard_minimizer2, Double};

    use crate::common::*;

    ///
    /// Update a safeguarded trial value and interval for line search.
    ///
    /// This function assumes that the derivative at the point of x in the
    /// direction of the step. If the bracket is set to true, the minimizer has
    /// been bracketed in an interval of uncertainty with endpoints between x
    /// and y.
    ///
    /// # Arguments
    ///
    /// * x, fx, and dx: variables which specify the step, the function, and the
    /// derivative at the best step obtained so far. The derivative must be
    /// negative in the direction of the step, that is, dx and t-x must have
    /// opposite signs. On output these parameters are updated appropriately.
    ///
    /// * y, fy, and dy: variables which specify the step, the function, and
    /// the derivative at the other endpoint of the interval of uncertainty. On
    /// output these parameters are updated appropriately.
    ///
    /// * t, ft, and dt: variables which specify the step, the function, and the
    /// derivative at the current step. If bracket is set true then on input t
    /// must be between x and y. On output t is set to the new step.
    ///
    /// * tmin, tmax: lower and upper bounds for the step.
    ///
    /// * `brackt`: Specifies if a minimizer has been bracketed. If the
    /// minimizer has not been bracketed then on input brackt must be set false.
    /// If the minimizer is bracketed then on output `brackt` is set true.
    ///
    /// # Return
    /// - Status value. Zero indicates a normal termination.
    ///
    pub(crate) fn update_trial_interval(
        x: &mut Double,
        fx: &mut Double,
        dx: &mut Double,
        y: &mut Double,
        fy: &mut Double,
        dy: &mut Double,
        t: &mut Double,
        ft: Double,
        dt: Double,
        tmin: Double,
        tmax: Double,
        brackt: &mut bool,
    ) -> Result<i32> {
        // fsigndiff
        let dsign = dt * (*dx / (*dx).abs()) < Double::ZERO;
        // minimizer of an interpolated cubic.
        let mut mc = Double::ZERO;
        // minimizer of an interpolated quadratic.
        let mut mq = Double::ZERO;
        // new trial value.
        let mut newt = Double::ZERO;

        // Check the input parameters for errors.
        if *brackt {
            if *t <= (if x < y { *x } else { *y }) || (if x > y { *x } else { *y }) <= *t {
                // The trival value t is out of the interval.
                bail!("The line-search step went out of the interval of uncertainty.");
            } else if Double::ZERO <= *dx * (*t - *x) {
                // The function must decrease from x.
                bail!("The current search direction increases the objective function value.");
            } else if tmax < tmin {
                // Incorrect tmin and tmax specified.
                bail!("A logic error occurred; alternatively, the interval of uncertainty became too small.");
            }
        }

        // Trial value selection.
        let bound = if *fx < ft {
            // Case 1: a higher function value.
            // The minimum is brackt. If the cubic minimizer is closer
            // to x than the quadratic one, the cubic one is taken, else
            // the average of the minimizers is taken.
            *brackt = true;
            cubic_minimizer(&mut mc, *x, *fx, *dx, *t, ft, dt);
            quard_minimizer(&mut mq, *x, *fx, *dx, *t, ft);
            if (mc - *x).abs() < (mq - *x).abs() {
                newt = mc
            } else {
                newt = mc + (mq - mc).ldexp(-1)
            }

            1
        } else if dsign {
            // Case 2: a lower function value and derivatives of
            // opposite sign. The minimum is brackt. If the cubic
            // minimizer is closer to x than the quadratic (secant) one,
            // the cubic one is taken, else the quadratic one is taken.
            *brackt = true;
            cubic_minimizer(&mut mc, *x, *fx, *dx, *t, ft, dt);
            quard_minimizer2(&mut mq, *x, *dx, *t, dt);
            if (mc - *t).abs() > (mq - *t).abs() {
                newt = mc
            } else {
                newt = mq
            }

            0
        } else if dt.abs() < (*dx).abs() {
            // Case 3: a lower function value, derivatives of the
            // same sign, and the magnitude of the derivative decreases.
            // The cubic minimizer is only used if the cubic tends to
            // infinity in the direction of the minimizer or if the minimum
            // of the cubic is beyond t. Otherwise the cubic minimizer is
            // defined to be either tmin or tmax. The quadratic (secant)
            // minimizer is also computed and if the minimum is brackt
            // then the the minimizer closest to x is taken, else the one
            // farthest away is taken.
            cubic_minimizer2(&mut mc, *x, *fx, *dx, *t, ft, dt, tmin, tmax);
            quard_minimizer2(&mut mq, *x, *dx, *t, dt);
            if *brackt {
                if (*t - mc).abs() < (*t - mq).abs() {
                    newt = mc
                } else {
                    newt = mq
                }
            } else if (*t - mc).abs() > (*t - mq).abs() {
                newt = mc
            } else {
                newt = mq
            }

            1
        } else {
            // Case 4: a lower function value, derivatives of the
            // same sign, and the magnitude of the derivative does
            // not decrease. If the minimum is not brackt, the step
            // is either tmin or tmax, else the cubic minimizer is taken.
            if *brackt {
                cubic_minimizer(&mut newt, *t, ft, dt, *y, *fy, *dy);
            } else if *x < *t {
                newt = tmax
            } else {
                newt = tmin
            }

            0
        };

        // Update the interval of uncertainty. This update does not
        // depend on the new step or the case analysis above.
        // - Case a: if f(x) < f(t),
        //    x <- x, y <- t.
        // - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
        //   x <- t, y <- y.
        // - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
        //   x <- t, y <- x.
        if *fx < ft {
            /* Case a */
            *y = *t;
            *fy = ft;
            *dy = dt
        } else {
            /* Case c */
            if dsign {
                *y = *x;
                *fy = *fx;
                *dy = *dx
            }
            /* Cases b and c */
            *x = *t;
            *fx = ft;
            *dx = dt
        }

        // Clip the new trial value in [tmin, tmax].
        if tmax < newt {
            newt = tmax
        }
        if newt < tmin {
            newt = tmin
        }

        // Redefine the new trial value if it is close to the upper bound of the
        // interval.
        if *brackt && 0 != bound {
            mq = *x + dd!(0.66) * (*y - *x);
            if *x < *y {
                if mq < newt {
                    newt = mq
                }
            } else if newt < mq {
                newt = mq
            }
        }

        // Return the new trial value.
        *t = newt;

        Ok(0)
    }
}

#[inline]
fn double_max(x: Double, y: Double) -> Double {
    if x > y {
        x
    } else {
        y
    }
}
/// Find a minimizer of an interpolated cubic function.
///
/// # Arguments
///  * `cm`: The minimizer of the interpolated cubic.
///  * `u` : The value of one point, u.
///  * `fu`: The value of f(u).
///  * `du`: The value of f'(u).
///  * `v` : The value of another point, v.
///  * `fv`: The value of f(v).
///  * `dv`:  The value of f'(v).
#[inline]
fn cubic_minimizer(
    cm: &mut Double,
    u: Double,
    fu: Double,
    du: Double,
    v: Double,
    fv: Double,
    dv: Double,
) {
    let d = v - u;
    let theta = (fu - fv) * dd!(3) / d + du + dv;

    let mut p = theta.abs();
    let mut q = du.abs();
    let mut r = dv.abs();
    let s = double_max(double_max(p, q), r); // max3(p, q, r)
    let a = theta / s;
    let mut gamma = s * (a * a - du / s * (dv / s)).sqrt();
    if v < u {
        gamma = -gamma
    }
    p = gamma - du + theta;
    q = gamma - du + gamma + dv;
    r = p / q;
    *cm = u + r * d;
}

/// Find a minimizer of an interpolated cubic function.
///
/// # Arguments
///  * cm  :   The minimizer of the interpolated cubic.
///  * u   :   The value of one point, u.
///  * fu  :   The value of f(u).
///  * du  :   The value of f'(u).
///  * v   :   The value of another point, v.
///  * fv  :   The value of f(v).
///  * dv  :   The value of f'(v).
///  * xmin:   The minimum value.
///  * xmax:   The maximum value.
#[inline]
fn cubic_minimizer2(
    cm: &mut Double,
    u: Double,
    fu: Double,
    du: Double,
    v: Double,
    fv: Double,
    dv: Double,
    xmin: Double,
    xmax: Double,
) {
    // STP - STX
    let d = v - u;
    let theta = (fu - fv) * dd!(3) / d + du + dv;
    let mut p = theta.abs();
    let mut q = du.abs();
    let mut r = dv.abs();
    // s = max3(p, q, r);
    let s = double_max(double_max(p, q), r); // max3(p, q, r)
    let a = theta / s;

    let mut gamma = s * (double_max(Double::ZERO, a * a - du / s * (dv / s)).sqrt());
    // STX < STP
    if u < v {
        gamma = -gamma
    }
    p = gamma - dv + theta;
    q = gamma - dv + gamma + du;
    r = p / q;
    if r < Double::ZERO && gamma != Double::ZERO {
        *cm = v - r * d;
    // } else if a < 0 as Double {
    //  ELSE IF (STP .GT. STX) THEN
    } else if v > u {
        *cm = xmax;
    } else {
        *cm = xmin;
    }
}

/// Find a minimizer of an interpolated quadratic function.
///
/// # Arguments
/// * qm : The minimizer of the interpolated quadratic.
/// * u  : The value of one point, u.
/// * fu : The value of f(u).
/// * du : The value of f'(u).
/// * v  : The value of another point, v.
/// * fv : The value of f(v).
#[inline]
fn quard_minimizer(qm: &mut Double, u: Double, fu: Double, du: Double, v: Double, fv: Double) {
    let a = v - u;
    *qm = u + du / ((fu - fv) / a + du) / dd!(2) * a;
}

/// Find a minimizer of an interpolated quadratic function.
///
/// # Arguments
/// * `qm` :    The minimizer of the interpolated quadratic.
/// * `u`  :    The value of one point, u.
/// * `du` :    The value of f'(u).
/// * `v`  :    The value of another point, v.
/// * `dv` :    The value of f'(v).
#[inline]
fn quard_minimizer2(qm: &mut Double, u: Double, du: Double, v: Double, dv: Double) {
    let a = u - v;
    *qm = v + dv / (dv - du) * a;
}

use self::LineSearchAlgorithm::*;

/// `prb` holds input variables `x`, gradient `gx` arrays, and function value
/// `fx`. on input it must contain the base point for the line search. on output
/// it contains data on x + stp*d.
pub fn line_search_backtracking<E>(
    prb: &mut Problem<E>,
    stp: &mut Double,   // step length
    param: &LineSearch, // line search parameters
) -> Result<usize>
where
    E: FnMut(&[Double], &mut [Double]) -> Result<Double>,
{
    let dginit = prb.dginit()?;
    let dec: Double = dd!(0.5);
    let inc: Double = dd!(2.1);

    // The initial value of the objective function.
    let finit = prb.fx;
    let dgtest = param.ftol * dginit;

    // quick wrapper
    let orthantwise = prb.orthantwise();
    if orthantwise {
        prb.update_orthant_new_point();
    }

    let mut width: Double;
    for count in 1..param.max_linesearch {
        prb.take_line_step(*stp);

        // Evaluate the function and gradient values.
        prb.evaluate()?;

        if prb.fx > finit + *stp * dgtest {
            width = dec;
        } else if param.algorithm == BacktrackingArmijo || orthantwise {
            // The sufficient decrease condition.
            // Exit with the Armijo condition.
            return Ok(count);
        } else {
            // Check the Wolfe condition.
            let dg = prb.dg_unchecked();
            if dg < param.gtol * dginit {
                width = inc
            } else if param.algorithm == BacktrackingWolfe {
                // Exit with the regular Wolfe condition.
                return Ok(count);
            } else if dg > -param.gtol * dginit {
                width = dec
            } else {
                return Ok(count);
            }
        }

        // allow energy rises
        // only check strong wolfe condition
        if param.gradient_only {
            info!("allow energy rises");
            let dg = prb.dg_unchecked();
            if dg.abs() <= -param.gtol * dginit.abs() {
                return Ok(count);
            }
        }

        param.validate_step(*stp)?;
        *stp *= width
    }

    // Maximum number of iteration.
    info!("The line-search routine reaches the maximum number of evaluations.");

    Ok(param.max_linesearch)
}

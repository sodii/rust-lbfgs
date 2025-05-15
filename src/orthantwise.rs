//! Implements the Orthant-Wise Limited-memory Quasi-Newton (OWL-QN)
//! algorithm.
//!
//! OWL-QN algorithm minimizes the objective function F(x) combined
//! with the L1 norm |x| of the variables, {F(x) + C |x|}.
//!
//! As the L1 norm |x| is not differentiable at zero, the library
//! modifies function and gradient evaluations from a client program
//! suitably; a client program thus have only to return the function
//! value F(x) and gradients G(x) as usual.
//!
//! # Reference
//! - Andrew, G.; Gao, J. Scalable Training of L 1-Regularized Log-Linear Models. In Proceedings of the 24th international conference on Machine learning; ACM, 2007; pp 33–40.

use crate::math::*;
use qd::{dd, Double};

/// Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) algorithm
#[derive(Copy, Clone, Debug)]
pub struct Orthantwise {
    /// The weight for the L1 regularization.
    ///
    /// This parameter is the coefficient for the |x|, i.e., C. The
    /// default value is 1.
    pub c: Double,

    /// Start index for computing L1 norm of the variables.
    ///
    /// This parameter b (0 <= b < N) specifies the index number from
    /// which the library computes the L1 norm of the variables x,
    ///
    /// |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
    ///
    /// In other words, variables x_1, ..., x_{b-1} are not used for
    /// computing the L1 norm. Setting b (0 < b < N), one can protect
    /// variables, x_1, ..., x_{b-1} (e.g., a bias term of logistic
    /// regression) from being regularized. The default value is zero.
    pub start: usize,

    /// End index for computing L1 norm of the variables.
    ///
    /// This parameter e (0 < e <= N) specifies the index number at
    /// which the library stops computing the L1 norm of the variables
    /// x.
    pub end: Option<usize>,
}

impl Default for Orthantwise {
    fn default() -> Self {
        Orthantwise {
            c: Double::ONE,
            start: 0,
            end: None,
        }
    }
}

impl Orthantwise {
    /// a dirty wrapper for start and end parameters in orthantwise optimization
    fn start_end(&self, x: &[Double]) -> (usize, usize) {
        let start = self.start;
        let n = x.len();
        // do not panic when end parameter is too large
        let end = self.end.unwrap_or(n).min(n);
        assert!(
            start < end,
            "invalid start for orthantwise: {start} (end = {end})"
        );

        (start, end)
    }

    /// Compute the L1 norm of the variable x.
    pub(crate) fn x1norm(&self, x: &[Double]) -> Double {
        let (start, end) = self.start_end(x);

        let mut s = Double::ZERO;
        for i in start..end {
            s += self.c * x[i].abs();
        }

        s
    }

    /// Compute the psuedo-gradient.
    pub(crate) fn compute_pseudo_gradient(&self, pg: &mut [Double], x: &[Double], g: &[Double]) {
        let (start, end) = self.start_end(x);

        for i in 0..start {
            pg[i] = g[i];
        }

        // Compute the psuedo-gradient (see Eq 4)
        let c = self.c;
        assert!(c.is_sign_positive(), "invalid orthantwise param c: {c}");
        for i in start..end {
            // Differentiable.
            if x[i] != Double::ZERO {
                pg[i] = g[i] + x[i].signum() * c;
            } else {
                let right_partial = g[i] + c;
                let left_partial = g[i] - c;
                if right_partial < Double::ZERO {
                    pg[i] = right_partial;
                } else if left_partial > Double::ZERO {
                    pg[i] = left_partial;
                } else {
                    pg[i] = Double::ZERO;
                }
            }
        }

        for i in end..g.len() {
            pg[i] = g[i];
        }
    }

    /// Choose the orthant for the new point.
    ///
    /// During the line search, each search point is projected onto
    /// the orthant of the previous point.
    pub(crate) fn constraint_line_search(&self, x: &mut [Double], wp: &[Double]) {
        let (start, end) = self.start_end(x);

        // FIXME: after constraint, x may be identical to xp, which
        // will lead to convergence failure.

        // for i in start..end {
        //     let epsilon = wp[i];
        //     // if epsilon * x[i] <= Double::ZERO {
        //     if epsilon != signum(x[i]) {
        //         x[i] = Double::ZERO;
        //     }
        // }

        project(x[start..end].iter_mut(), wp[start..end].iter().copied());
    }

    /// Constrain the search direction for orthant-wise updates.
    ///
    /// # Parameters
    /// * d: direction vector
    /// * pg: previous gradient vector
    pub(crate) fn constrain_search_direction(&self, d: &mut [Double], pg: &[Double]) {
        let (start, end) = self.start_end(pg);

        // p^k = pi(d^k; v^k)
        // where v^k = - pg^k
        project(d[start..end].iter_mut(), pg[start..end].iter().map(|x| -x));

        // for i in start..end {
        //     if signum(d[i]) != signum(-pg[i]) {
        //     // if signum(d[i]) == signum(pg[i]) {
        //         d[i] = Double::ZERO;
        //     }
        // }

        // just cite the comment from here:
        //
        // https://github.com/scalanlp/breeze/blob/28cfe3a0799bdf2d4191b93c43e751ef9f49285a/math/src/main/scala/breeze/optimize/OWLQN.scala#L45
        //
        // there are some cases where the algorithm won't converge
        // (confirmed with the author, Galen Andrew).
        assert_ne!(
            d.vec2norm(),
            Double::ZERO,
            "invalid direction vector after constraints: {d:?}"
        );
    }
}

// pi alignment operator - projection of x on orthat defined by y
fn project<'a>(x: impl Iterator<Item = &'a mut Double>, y: impl Iterator<Item = Double>) {
    for (xi, yi) in x.zip(y) {
        if signum(*xi) != signum(yi) {
            *xi = Double::ZERO;
        }
    }
}

// follow the mathematical definition
pub fn signum(x: Double) -> Double {
    if x.is_nan() || x == Double::ZERO {
        Double::ZERO
    } else {
        x.signum()
    }
}

//! core data structures for L-BFGS algorithm

use crate::common::*;
use crate::math::*;
use crate::orthantwise::*;
use qd::{dd, Double};

/// Represents an optimization problem.
///
/// `Problem` holds input variables `x`, gradient `gx` arrays, and function value `fx`.
pub struct Problem<'a, E>
where
    E: FnMut(&[Double], &mut [Double]) -> Result<Double>,
{
    /// x is an array of length n. on input it must contain the base point for
    /// the line search.
    pub(crate) x: &'a mut [Double],

    /// `fx` is a variable. It must contain the value of problem `f` at
    /// x.
    pub(crate) fx: Double,

    /// `gx` is an array of length n. It must contain the gradient of `f` at
    /// x.
    pub(crate) gx: Vec<Double>,

    /// Cached position vector of previous step.
    pub(crate) xp: Vec<Double>,

    /// Cached gradient vector of previous step.
    pub(crate) gp: Vec<Double>,

    /// Pseudo gradient for OrthantWise Limited-memory Quasi-Newton (owlqn) algorithm.
    pg: Vec<Double>,

    /// For owlqn projection
    wp: Vec<Double>,

    /// Search direction
    d: Vec<Double>,

    /// Store callback function for evaluating objective function.
    eval_fn: E,

    /// Orthantwise operations
    owlqn: Option<Orthantwise>,

    /// Evaluated or not
    evaluated: bool,

    /// The number of evaluation.
    neval: usize,
}

impl<'a, E> Problem<'a, E>
where
    E: FnMut(&[Double], &mut [Double]) -> Result<Double>,
{
    /// Initialize problem with array length n
    pub fn new(x: &'a mut [Double], eval: E, owlqn: Option<Orthantwise>) -> Self {
        let n = x.len();
        Problem {
            fx: Double::ZERO,
            gx: vec![Double::ZERO; n],
            xp: vec![Double::ZERO; n],
            gp: vec![Double::ZERO; n],
            pg: vec![Double::ZERO; n],
            wp: vec![Double::ZERO; n],
            d: vec![Double::ZERO; n],
            evaluated: false,
            neval: 0,
            x,
            eval_fn: eval,
            owlqn,
        }
    }

    /// Compute the initial gradient in the search direction.
    pub fn dginit(&self) -> Result<Double> {
        if self.owlqn.is_none() {
            let dginit = self.gx.vecdot(&self.d);
            if dginit > Double::ZERO {
                warn!(
                    "The current search direction increases the objective function value. dginit = {:-0.4}",
                    dginit
                );
            }

            Ok(dginit)
        } else {
            Ok(self.pg.vecdot(&self.d))
        }
    }

    /// Update search direction using evaluated gradient.
    pub fn update_search_direction(&mut self) {
        if self.owlqn.is_some() {
            self.d.vecncpy(&self.pg);
        } else {
            self.d.vecncpy(&self.gx);
        }
    }

    /// Return a reference to current search direction vector
    pub fn search_direction(&self) -> &[Double] {
        &self.d
    }

    /// Return a mutable reference to current search direction vector
    pub fn search_direction_mut(&mut self) -> &mut [Double] {
        &mut self.d
    }

    /// Compute the gradient in the search direction without sign checking.
    pub fn dg_unchecked(&self) -> Double {
        self.gx.vecdot(&self.d)
    }

    /// Evaluate object value and gradient.
    pub fn evaluate(&mut self) -> Result<()> {
        self.fx = (self.eval_fn)(&self.x, &mut self.gx)?;

        // Compute the L1 norm of the variables and add it to the object value.
        if let Some(owlqn) = self.owlqn {
            self.fx += owlqn.x1norm(&self.x);
            owlqn.compute_pseudo_gradient(&mut self.pg, &self.x, &self.gx);
        }

        self.evaluated = true;
        self.neval += 1;

        Ok(())
    }

    /// Return total number of evaluations.
    pub fn number_of_evaluation(&self) -> usize {
        self.neval
    }

    /// Test if `Problem` has been evaluated or not
    pub fn evaluated(&self) -> bool {
        self.evaluated
    }

    /// Copies all elements from src into self.
    pub fn clone_from(&mut self, src: &Problem<E>) {
        self.x.clone_from_slice(&src.x);
        self.gx.clone_from_slice(&src.gx);
        self.fx = src.fx;
    }

    /// Take a line step along search direction.
    ///
    /// Compute the current value of x: x <- x + (*step) * d.
    ///
    pub fn take_line_step(&mut self, step: Double) {
        self.x.veccpy(&self.xp);
        self.x.vecadd(&self.d, step);

        // Choose the orthant for the new point.
        // The current point is projected onto the orthant.
        if let Some(owlqn) = self.owlqn {
            owlqn.constraint_line_search(&mut self.x, &self.wp);
        }
    }

    /// Choose the orthant for the new point.
    pub fn update_orthant_new_point(&mut self) {
        use crate::orthantwise::signum;

        let n = self.x.len();
        for i in 0..n {
            // let epsilon = if self.xp[i] == Double::ZERO { -self.pg[i] } else { self.xp[i] };
            let epsilon = if self.xp[i] == Double::ZERO {
                signum(-self.pg[i])
            } else {
                signum(self.xp[i])
            };
            self.wp[i] = epsilon;
        }
    }

    /// Return gradient vector norm: ||gx||
    pub fn gnorm(&self) -> Double {
        if self.owlqn.is_some() {
            self.pg.vec2norm()
        } else {
            self.gx.vec2norm()
        }
    }

    /// Return position vector norm: ||x||
    pub fn xnorm(&self) -> Double {
        self.x.vec2norm()
    }

    pub fn orthantwise(&self) -> bool {
        self.owlqn.is_some()
    }

    /// Revert to previous step
    pub fn revert(&mut self) {
        self.x.veccpy(&self.xp);
        self.gx.veccpy(&self.gp);
    }

    /// Store the current position and gradient vectors.
    pub fn save_state(&mut self) {
        self.xp.veccpy(&self.x);
        self.gp.veccpy(&self.gx);
    }

    /// Constrain the search direction for orthant-wise updates.
    pub fn constrain_search_direction(&mut self) {
        if let Some(owlqn) = self.owlqn {
            owlqn.constrain_search_direction(&mut self.d, &self.pg);
        }
    }
}

/// Store optimization progress data, for progress monitor
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Progress<'a> {
    /// The current values of variables
    pub x: &'a [Double],

    /// The current gradient values of variables.
    pub gx: &'a [Double],

    /// The current value of the objective function.
    pub fx: Double,

    /// The Euclidean norm of the variables
    pub xnorm: Double,

    /// The Euclidean norm of the gradients.
    pub gnorm: Double,

    /// The line-search step used for this iteration.
    pub step: Double,

    /// The iteration count.
    pub niter: usize,

    /// The total number of evaluations.
    pub neval: usize,

    /// The number of function evaluation calls in line search procedure
    pub ncall: usize,
}

impl<'a> Progress<'a> {
    pub fn new<E>(prb: &'a Problem<E>, niter: usize, ncall: usize, step: Double) -> Self
    where
        E: FnMut(&[Double], &mut [Double]) -> Result<Double>,
    {
        Progress {
            x: &prb.x,
            gx: &prb.gx,
            fx: prb.fx,
            xnorm: prb.xnorm(),
            gnorm: prb.gnorm(),
            neval: prb.number_of_evaluation(),
            ncall,
            step,
            niter,
        }
    }
}

#[derive(Debug, Clone)]
/// Represents the final optimization outcome
pub struct Report {
    /// The current value of the objective function.
    pub fx: Double,

    /// The Euclidean norm of the variables
    pub xnorm: Double,

    /// The Euclidean norm of the gradients.
    pub gnorm: Double,

    /// The total number of evaluations.
    pub neval: usize,
}

impl Report {
    pub(crate) fn new<E>(prb: &Problem<E>) -> Self
    where
        E: FnMut(&[Double], &mut [Double]) -> Result<Double>,
    {
        Self {
            fx: prb.fx,
            xnorm: prb.xnorm(),
            gnorm: prb.gnorm(),
            neval: prb.number_of_evaluation(),
        }
    }
}

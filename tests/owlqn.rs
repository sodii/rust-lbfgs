#![feature(f128)]
use anyhow::*;
use approx::*;
use liblbfgs::{default_progress, lbfgs, math::*, Progress};

#[test]
fn test_owlqn() -> Result<()> {
    use vecfx::nalgebra::{DMatrix, DVector, DVectorSlice};

    let nrow = 500;
    let ncol = 21;

    let y = read_csv("tests/y.csv")?;
    assert_eq!(y.len(), nrow);
    let ymat = DVector::from(y);
    // dbg!(ymat.shape());
    let x = read_csv("tests/x.csv")?;
    assert_eq!(x.len(), ncol * nrow);
    let xmat = DMatrix::from_vec(21, 500, x).transpose();
    // dbg!(xmat.shape());

    let prec = 0.0f64;
    let evaluate = move |x: &[f128], gx: &mut [f128]| {
        // convert x (f128) to f64 for nalgebra ops
        let x_f64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        // calculate fx
        //
        // likelihood <- function(par, X, y, prec=0)
        // Xbeta <- X %*% par
        // -(sum(y * Xbeta - exp(Xbeta)) - .5 * sum(par^2*prec))
        let par = DVectorSlice::from(x_f64.as_slice());
        let xbeta = &xmat * &par;
        let xbeta_exp = xbeta.map(|x| x.exp());
        let par2 = par.map(|x| x.powi(2));
        let fx =
            -1.0 * (&ymat.component_mul(&xbeta) - &xbeta_exp).sum() + 0.5 * (prec * par2).sum();

        // calculate gx
        //
        // gradient <- function(par, X, y, prec=0)
        // -(crossprod(X, (y - exp(Xbeta))) - par * prec)
        let t = &ymat - xbeta_exp;
        let g = -xmat.transpose() * t + par * prec;
        // copy gradient back to f128
        for (gi, &gv) in gx.iter_mut().zip(g.as_slice()) {
            *gi = gv as f128;
        }

        Ok(fx as f128)
    };

    let mut xinit = vec![0.0f128; ncol];
    let prb = lbfgs()
        .with_orthantwise(1.0f128, 1, 21)
        // .with_max_iterations(90)
        .with_epsilon(1E-4)
        .minimize(&mut xinit, evaluate, |prgr| {
            println!("Iteration {}:", prgr.niter);
            println!(
                " fx = {:-12.6} xnorm = {:-12.6}, gnorm = {:-12.6}, ls = {}, step = {}",
                prgr.fx as f64, prgr.xnorm as f64, prgr.gnorm as f64, prgr.ncall, prgr.step as f64
            );
            false
        })
        .expect("lbfgs minimize");

    assert_relative_eq!(-42724.136705, prb.fx as f64, epsilon = 1e-6);

    Ok(())
}

// read x/y from csv file
fn read_csv(f: &str) -> Result<Vec<f64>> {
    use std::fs::File;
    use std::io::{self, BufRead};
    use std::path::Path;

    let file = File::open(f)?;
    let mut all = vec![];
    // skip the headers in first row and the first column
    for line in io::BufReader::new(file).lines().skip(1) {
        let line = line?;
        let values: Option<Vec<f64>> = line.split(",").skip(1).map(|s| s.parse().ok()).collect();
        if let Some(cols) = values {
            all.extend(cols);
        }
    }

    Ok(all)
}

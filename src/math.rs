//! Backend for L-BFGS vector operations

use qd::{dd, Double};
/// Abstracting lbfgs required math operations
pub trait LbfgsMath<T> {
    /// y += c*x
    fn vecadd(&mut self, x: &[T], c: T);

    /// vector dot product
    /// s = x.dot(y)
    fn vecdot(&self, other: &[T]) -> Double;

    /// y = z
    fn veccpy(&mut self, x: &[T]);

    /// y = -x
    fn vecncpy(&mut self, x: &[T]);

    /// z = x - y
    fn vecdiff(&mut self, x: &[T], y: &[T]);

    /// y *= c
    fn vecscale(&mut self, c: T);

    /// ||x||
    fn vec2norm(&self) -> T;

    /// 1 / ||x||
    fn vec2norminv(&self) -> T;
}

impl LbfgsMath<Double> for [Double] {
    /// y += c*x
    fn vecadd(&mut self, x: &[Double], c: Double) {
        for (y, x) in self.iter_mut().zip(x) {
            *y += c * x;
        }
    }

    /// s = y.dot(x)
    fn vecdot(&self, other: &[Double]) -> Double {
        self.iter().zip(other).map(|(x, y)| x * y).sum()
    }

    /// y *= c
    fn vecscale(&mut self, c: Double) {
        for y in self.iter_mut() {
            *y *= c;
        }
    }

    /// y = x
    fn veccpy(&mut self, x: &[Double]) {
        for (v, x) in self.iter_mut().zip(x) {
            *v = *x;
        }
    }

    /// y = -x
    fn vecncpy(&mut self, x: &[Double]) {
        for (v, x) in self.iter_mut().zip(x) {
            *v = -x;
        }
    }

    /// z = x - y
    fn vecdiff(&mut self, x: &[Double], y: &[Double]) {
        for ((z, x), y) in self.iter_mut().zip(x).zip(y) {
            *z = x - y;
        }
    }

    /// ||x||
    fn vec2norm(&self) -> Double {
        let n2 = self.vecdot(&self);
        n2.sqrt()
    }

    /// 1/||x||
    fn vec2norminv(&self) -> Double {
        self.vec2norm().recip()
    }
}

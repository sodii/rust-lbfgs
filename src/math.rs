//! Backend for L-BFGS vector operations

/// Abstracting lbfgs required math operations
pub trait LbfgsMath<T> {
    /// y += c*x
    fn vecadd(&mut self, x: &[T], c: T);

    /// vector dot product
    /// s = x.dot(y)
    fn vecdot(&self, other: &[T]) -> T;

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

impl LbfgsMath<f128> for [f128] {
    /// y += c*x
    fn vecadd(&mut self, x: &[f128], c: f128) {
        for (y, x) in self.iter_mut().zip(x) {
            *y += c * x;
        }
    }

    /// s = y.dot(x)
    fn vecdot(&self, other: &[f128]) -> f128 {
        let mut acc: f128 = 0.0f128;
        for (x, y) in self.iter().zip(other) {
            acc += *x * *y;
        }
        acc
    }

    /// y *= c
    fn vecscale(&mut self, c: f128) {
        for y in self.iter_mut() {
            *y *= c;
        }
    }

    /// y = x
    fn veccpy(&mut self, x: &[f128]) {
        for (v, x) in self.iter_mut().zip(x) {
            *v = *x;
        }
    }

    /// y = -x
    fn vecncpy(&mut self, x: &[f128]) {
        for (v, x) in self.iter_mut().zip(x) {
            *v = -x;
        }
    }

    /// z = x - y
    fn vecdiff(&mut self, x: &[f128], y: &[f128]) {
        for ((z, x), y) in self.iter_mut().zip(x).zip(y) {
            *z = x - y;
        }
    }

    /// ||x||
    fn vec2norm(&self) -> f128 {
        let n2 = self.vecdot(&self);
        n2.sqrt()
    }

    /// 1/||x||
    fn vec2norminv(&self) -> f128 {
        1.0f128 / self.vec2norm()
    }
}

#[test]
fn test_lbfgs_math() {
    // vector scaled add
    let x: [f128; 3] = [1.0f128, 1.0f128, 1.0f128];
    let c: f128 = 2.0f128;

    let mut y: [f128; 3] = [1.0f128, 2.0f128, 3.0f128];
    y.vecadd(&x, c);

    assert_eq!(3.0f128, y[0]);
    assert_eq!(4.0f128, y[1]);
    assert_eq!(5.0f128, y[2]);

    // vector dot
    let v = y.vecdot(&x);
    assert_eq!(12.0f128, v);

    // vector scale
    y.vecscale(2.0f128);
    assert_eq!(6.0f128, y[0]);
    assert_eq!(8.0f128, y[1]);
    assert_eq!(10.0f128, y[2]);

    // vector diff
    let mut z = y.clone();
    z.vecdiff(&x, &y);
    assert_eq!(-5.0f128, z[0]);
    assert_eq!(-7.0f128, z[1]);
    assert_eq!(-9.0f128, z[2]);

    // vector copy
    y.veccpy(&x);

    // y = -x
    y.vecncpy(&x);
    assert_eq!(-1.0f128, y[0]);
    assert_eq!(-1.0f128, y[1]);
    assert_eq!(-1.0f128, y[2]);
}

extern crate maplit;
extern crate sorted_iter;

use maplit::*;
use sorted_iter::SortedIteratorExt;

fn v<E>(x: impl Iterator<Item = E>) -> Vec<E> {
    x.take(10).collect()
}

fn main() {
    let primes = btreeset! { 2u64, 3, 5, 7, 11, 13 };
    let fibs = btreeset! { 1u64, 2, 3, 5, 8, 13 };
    let primes = primes.iter();
    let fibs = fibs.iter();
    let nats = 1u64..;
    // both primes and fibs
    let both = primes.clone().intersection(fibs.clone());
    // either primes or fibs
    let either = primes.union(fibs).cloned();
    // natural numbers that are neither
    let neither = nats.difference(either);

    println!("Fibonacci primes: {:?}", v(both));
    println!("Neither: {:?}", v(neither));
}

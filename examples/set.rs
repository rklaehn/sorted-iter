extern crate sorted_iter;
extern crate maplit;

use maplit::*;
use sorted_iter::SortedIterator;

fn main() {
    let primes = btreeset! { 2, 3, 5, 7 };
    let fibs = btreeset! { 1, 2, 3, 5, 8, 13 };
    let res: Vec<_> = primes.iter().intersection(fibs.iter()).collect();
    println!("{:?}", res);
}

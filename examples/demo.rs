extern crate sorted_iter;

#[macro_use]
extern crate maplit;

use sorted_iter::{SortedIterator, SortedPairIterator};
use maplit::*;

fn main() {
    let a = btreemap!{
        1 => "New York",
        2 => "Tokyo",
    };
    let b = btreemap!{
        1 => "USA",
        2 => "Japan",
    };
    println!("{:?} {:?}", a, b);
    let res: Vec<_> = a.iter().join(b.iter()).collect();
    println!("{:?}", res);
}
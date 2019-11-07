extern crate sorted_iter;
extern crate maplit;

use maplit::*;
use sorted_iter::SortedPairIterator;

fn main() {
    let city = btreemap! {
        1 => "New York",
        2 => "Tokyo",
    };
    let country = btreemap! {
        1 => "USA",
        2 => "Japan",
    };
    let res: Vec<_> = city.iter().join(country.iter()).collect();
    println!("{:?}", res);
}

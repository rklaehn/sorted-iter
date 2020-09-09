
[![Build Status]][travis] [![Latest Version]][crates.io] [![Docs Badge]][docs.rs]

# Sorted Iter

[Build Status]: https://api.travis-ci.org/rklaehn/sorted-iter.svg?branch=master
[travis]: https://travis-ci.org/rklaehn/sorted-iter
[Latest Version]: https://img.shields.io/crates/v/sorted-iter.svg
[crates.io]: https://crates.io/crates/sorted-iter
[Docs Badge]: https://img.shields.io/badge/docs-docs.rs-green
[docs.rs]: https://docs.rs/sorted-iter

# About

<!-- cargo-sync-readme start -->

This crate provides set and relational operations for all iterators in the standard library that are known
at compile time to be sorted.

# Set operations
```rust
use sorted_iter::SortedIterator;

let primes = btreeset! { 2, 3, 5, 7, 11, 13u64 }.into_iter();
let fibs = btreeset! { 1, 2, 3, 5, 8, 13u64 }.into_iter();
let fib_primes = primes.intersection(fibs);
```

It is possible to efficiently define set operations on sorted iterators. Sorted iterators are
very common in the standard library. E.g. the elements of a [BTreeSet] or the keys of a [BTreeMap]
are guaranteed to be sorted according to the element order, as are iterable ranges like `0..100`.

There are also a number of operations on iterators that preserve the sort order. E.g. if an
iterator is sorted, [take], [take_while] etc. are going to result in a sorted iterator as well.

Since the complete types of iterators are typically visible in rust, it is possible to encode these
rules at type level. This is what this crate does.

For available set operations, see [SortedIterator].
For sorted iterators in the std lib, see instances the for [SortedByItem] marker trait.

# Relational operations
```rust
use sorted_iter::SortedPairIterator;

let cities = btreemap! {
  1 => "New York",
  2 => "Tokyo",
  3u8 => "Berlin"
}.into_iter();
let countries = btreemap! {
  1 => "USA",
  2 => "Japan",
  3u8 => "Germany"
}.into_iter();
let cities_and_countries = cities.join(countries);
```

Iterators of pairs that are sorted according to the first element / key are also very common in
the standard library and elsewhere. E.g. the elements of a [BTreeMap] are guaranteed to be sorted
according to the key order.

The same rules as for sorted iterators apply for preservation of the sort order, except that there
are some additional operations that preserve sort order. Anything that only operates on the value,
like e.g. map or filter_map on the value, is guaranteed to preserve the sort order.

The operations that can be defined on sorted pair operations are the relational operations known
from relational algebra / SQL, namely join, left_join, right_join and outer_join.

For available relational operations, see [SortedPairIterator].
For sorted iterators in the std lib, see instances the for [SortedByKey] marker trait.

# Transformations that retain order are allowed
```rust
use sorted_iter::*;

let odd = (1..31).step_by(2);
let multiples_of_3 = (3..30).step_by(3);
let either = odd.union(multiples_of_3);
```

# Transformations that can change the order lose the sorted property
```compile_fail
use sorted_iter::*;

// we have no idea what map does to the order. could be anything!
let a = (1..31).map(|x| -x);
let b = (3..30).step_by(3);
let either = a.union(b); // does not compile!
```

# Assuming sort ordering

For most std lib iterators, this library already provides instances. But there will occasionally be an iterator
from a third party library where you *know* that it is properly sorted.

For this case, there is an escape hatch:

```rust
// the assume_ extensions have to be implicitly imported
use sorted_iter::*;
use sorted_iter::assume::*;
let odd = vec![1,3,5,7u8].into_iter().assume_sorted_by_item();
let even = vec![2,4,6,8u8].into_iter().assume_sorted_by_item();
let all = odd.union(even);

let cities = vec![(1u8, "New York")].into_iter().assume_sorted_by_key();
let countries = vec![(1u8, "USA")].into_iter().assume_sorted_by_key();
let cities_and_countries = cities.join(countries);
```

# Marking your own iterators

If you have a library and want to mark some iterators as sorted, this is possible by implementing the
appropriate marker trait, [SortedByItem] or [SortedByKey].

```rust
// marker traits are not at top level, since usually you don't need them
use sorted_iter::sorted_iterator::SortedByItem;
use sorted_iter::sorted_pair_iterator::SortedByKey;

pub struct MySortedIter<T> { whatever: T }
pub struct MySortedPairIter<K, V> { whatever: (K, V) }

impl<T> SortedByItem for MySortedIter<T> {}
impl<K, V> SortedByKey for MySortedPairIter<K, V> {}
```

By reexporting the extension traits, you get a seamless experience for people using your library.

```rust
extern crate sorted_iter;
pub use sorted_iter::{SortedIterator, SortedPairIterator};
```

## Tests

Tests are done using the fantastic [quickcheck] crate, by comparing against the operations defined on
[BTreeSet] and [BTreeMap].

[SortedIterator]: trait.SortedIterator.html
[SortedPairIterator]: trait.SortedPairIterator.html
[SortedByItem]: sorted_iterator/trait.SortedByItem.html
[SortedByKey]: sorted_pair_iterator/trait.SortedByKey.html
[quickcheck]: https://github.com/BurntSushi/quickcheck
[BTreeSet]: https://doc.rust-lang.org/std/collections/struct.BTreeSet.html
[BTreeMap]: https://doc.rust-lang.org/std/collections/struct.BTreeMap.html
[take]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.take
[take_while]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.take_while
[Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html

<!-- cargo-sync-readme end -->

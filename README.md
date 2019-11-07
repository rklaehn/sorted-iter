
[![Build Status]][travis] [![Latest Version]][crates.io] [![Docs Badge]][docs.rs]

# Sorted Iter

[Build Status]: https://api.travis-ci.org/rklaehn/sorted-iter.svg?branch=master
[travis]: https://travis-ci.org/rklaehn/sorted-iter
[Latest Version]: https://img.shields.io/crates/v/sorted-iter.svg
[crates.io]: https://crates.io/crates/sorted-iter
[Docs Badge]: https://img.shields.io/badge/docs-docs.rs-green
[docs.rs]: https://docs.rs/sorted-iter

# TL;DR;

```rust
let city = btreemap!{
    1 => "New York",
    2 => "Tokyo",
};
let country = btreemap!{
    1 => "USA",
    2 => "Japan",
};
let res: Vec<_> = city.iter().join(country.iter()).collect();
```

# About

This provides typesafe extension for sorted iterators to perform set and relational operations.
By sorted I mean strictly sorted according to the [Ord] instance of the item or key type.

## Sorted iterators

It is possible to efficiently define set operations on sorted iterators. Sorted iterators are
very common in the standard library. E.g. the elements of a [BTreeSet] or the keys of a [BTreeMap]
are guaranteed to be sorted according to the element order.

There are also a number of operations on iterators that preserve the sort order. E.g. if an
iterator is sorted, [take], [take_while] etc. are going to result in a sorted iterator as well.

Since the complete types of iterators are typically visible in rust, it is possible to encode these
rules at type level. This is what this crate does.

## Sorted pair iterators

Iterators of pairs that are sorted according to the first element / key are also very common in
the standard library and elsewhere. E.g. the elements of a [BTreeMap] are guaranteed to be sorted
according to the key order.

The same rules as for sorted iterators apply for preservation of the sort order, except that there
are some additional operations that preserve sort order. Anything that only operates on the value,
like e.g. map or filter_map on the value, is guaranteed to preserve the sort order.

The operations that can be defined on sorted pair operations are the relational operations known
from relational algebra / SQL, namely join, left_join, right_join and outer_join.

## Instances

Instances are provided to allow treating iterators in the standard library that are guaranteed to be
sorted as sorted iterators.

## Tests

Tests are done using the fantastic [quickcheck] crate, by comparing against the operations defined on
[BTreeSet] and [BTreeMap].

[quickcheck]: https://github.com/BurntSushi/quickcheck
[BTreeSet]: https://doc.rust-lang.org/std/collections/struct.BTreeSet.html
[BTreeMap]: https://doc.rust-lang.org/std/collections/struct.BTreeMap.html
[take]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.take
[take_while]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.take_while
[Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html

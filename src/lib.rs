//! This crate provides set and relational operations for all iterators in the standard library that are known
//! at compile time to be sorted.
//!
//! # Set operations
//! ```
//! # extern crate maplit;
//! # use maplit::*;
//! # extern crate sorted_iter;
//! use sorted_iter::SortedIterator;
//!
//! let primes = btreeset! { 2, 3, 5, 7, 11, 13u64 }.into_iter();
//! let fibs = btreeset! { 1, 2, 3, 5, 8, 13u64 }.into_iter();
//! let fib_primes = primes.intersection(fibs);
//! ```
//!
//! It is possible to efficiently define set operations on sorted iterators. Sorted iterators are
//! very common in the standard library. E.g. the elements of a [BTreeSet] or the keys of a [BTreeMap]
//! are guaranteed to be sorted according to the element order, as are iterable ranges like `0..100`.
//!
//! There are also a number of operations on iterators that preserve the sort order. E.g. if an
//! iterator is sorted, [take], [take_while] etc. are going to result in a sorted iterator as well.
//!
//! Since the complete types of iterators are typically visible in rust, it is possible to encode these
//! rules at type level. This is what this crate does.
//!
//! For available set operations, see [SortedIterator].
//! For sorted iterators in the std lib, see instances the for [SortedByItem] marker trait.
//!
//! # Relational operations
//! ```
//! # extern crate maplit;
//! # use maplit::*;
//! # extern crate sorted_iter;
//! use sorted_iter::SortedPairIterator;
//!
//! let cities = btreemap! {
//!   1 => "New York",
//!   2 => "Tokyo",
//!   3u8 => "Berlin"
//! }.into_iter();
//! let countries = btreemap! {
//!   1 => "USA",
//!   2 => "Japan",
//!   3u8 => "Germany"
//! }.into_iter();
//! let cities_and_countries = cities.join(countries);
//! ```
//!
//! Iterators of pairs that are sorted according to the first element / key are also very common in
//! the standard library and elsewhere. E.g. the elements of a [BTreeMap] are guaranteed to be sorted
//! according to the key order.
//!
//! The same rules as for sorted iterators apply for preservation of the sort order, except that there
//! are some additional operations that preserve sort order. Anything that only operates on the value,
//! like e.g. map or filter_map on the value, is guaranteed to preserve the sort order.
//!
//! The operations that can be defined on sorted pair operations are the relational operations known
//! from relational algebra / SQL, namely join, left_join, right_join and outer_join.
//!
//! For available relational operations, see [SortedPairIterator].
//! For sorted iterators in the std lib, see instances the for [SortedByKey] marker trait.
//!
//! # Transformations that retain order are allowed
//! ```
//! # extern crate sorted_iter;
//! use sorted_iter::*;
//!
//! let odd = (1..31).step_by(2);
//! let multiples_of_3 = (3..30).step_by(3);
//! let either = odd.union(multiples_of_3);
//! ```
//!
//! # Transformations that can change the order lose the sorted property
//! ```compile_fail
//! # extern crate sorted_iter;
//! use sorted_iter::*;
//!
//! // we have no idea what map does to the order. could be anything!
//! let a = (1..31).map(|x| -x);
//! let b = (3..30).step_by(3);
//! let either = a.union(b); // does not compile!
//! ```
//!
//! # Assuming sort ordering
//!
//! For most std lib iterators, this library already provides instances. But there will occasionally be an iterator
//! from a third party library where you *know* that it is properly sorted.
//!
//! For this case, there is an escape hatch:
//!
//! ```
//! // the assume_ extensions have to be implicitly imported
//! use sorted_iter::*;
//! use sorted_iter::assume::*;
//! let odd = vec![1,3,5,7u8].into_iter().assume_sorted_by_item();
//! let even = vec![2,4,6,8u8].into_iter().assume_sorted_by_item();
//! let all = odd.union(even);
//!
//! let cities = vec![(1u8, "New York")].into_iter().assume_sorted_by_key();
//! let countries = vec![(1u8, "USA")].into_iter().assume_sorted_by_key();
//! let cities_and_countries = cities.join(countries);
//! ```
//!
//! # Marking your own iterators
//!
//! If you have a library and want to mark some iterators as sorted, this is possible by implementing the
//! appropriate marker trait, [SortedByItem] or [SortedByKey].
//!
//! ```
//! # extern crate sorted_iter;
//! // marker traits are not at top level, since usually you don't need them
//! use sorted_iter::sorted_iterator::SortedByItem;
//! use sorted_iter::sorted_pair_iterator::SortedByKey;
//!
//! pub struct MySortedIter<T> { whatever: T }
//! pub struct MySortedPairIter<K, V> { whatever: (K, V) }
//!
//! impl<T> SortedByItem for MySortedIter<T> {}
//! impl<K, V> SortedByKey for MySortedPairIter<K, V> {}
//! ```
//!
//! By reexporting the extension traits, you get a seamless experience for people using your library.
//!
//! ```
//! extern crate sorted_iter;
//! pub use sorted_iter::{SortedIterator, SortedPairIterator};
//! ```
//!
//! ## Tests
//!
//! Tests are done using the fantastic [quickcheck] crate, by comparing against the operations defined on
//! [BTreeSet] and [BTreeMap].
//!
//! [SortedIterator]: trait.SortedIterator.html
//! [SortedPairIterator]: trait.SortedPairIterator.html
//! [SortedByItem]: sorted_iterator/trait.SortedByItem.html
//! [SortedByKey]: sorted_pair_iterator/trait.SortedByKey.html
//! [quickcheck]: https://github.com/BurntSushi/quickcheck
//! [BTreeSet]: https://doc.rust-lang.org/std/collections/struct.BTreeSet.html
//! [BTreeMap]: https://doc.rust-lang.org/std/collections/struct.BTreeMap.html
//! [take]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.take
//! [take_while]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.take_while
//! [Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html
#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

pub mod sorted_iterator;
pub mod sorted_pair_iterator;

use crate::sorted_iterator::*;
use crate::sorted_pair_iterator::*;

#[deny(missing_docs)]

/// set operations for iterators where the items are sorted according to the natural order
pub trait SortedIterator: Iterator + Sized {
    /// union with another sorted iterator
    fn union<J>(self, that: J) -> Union<Self, J>
    where
        J: SortedIterator<Item = Self::Item>,
    {
        Union {
            a: self.peekable(),
            b: that.peekable(),
        }
    }
    /// intersection with another sorted iterator
    fn intersection<J>(self, that: J) -> Intersection<Self, J>
    where
        J: SortedIterator<Item = Self::Item>,
    {
        Intersection {
            a: self.peekable(),
            b: that.peekable(),
        }
    }
    /// difference with another sorted iterator
    fn difference<J>(self, that: J) -> Difference<Self, J>
    where
        J: SortedIterator<Item = Self::Item>,
    {
        Difference {
            a: self.peekable(),
            b: that.peekable(),
        }
    }
    /// symmetric difference with another sorted iterator
    fn symmetric_difference<J>(self, that: J) -> SymmetricDifference<Self, J>
    where
        J: SortedIterator<Item = Self::Item>,
    {
        SymmetricDifference {
            a: self.peekable(),
            b: that.peekable(),
        }
    }
    /// pairs with unit value
    fn pairs(self) -> Pairs<Self> {
        Pairs { i: self }
    }
}

impl<I> SortedIterator for I where I: Iterator + SortedByItem {}

/// Union of multiple sorted iterators.
///
/// An advantage of this function over multiple calls to `SortedIterator::union`
/// is that the number of merged sequences does not need to be known at the
/// compile time. The drawback lies in the fact that all iterators have to be
/// of the same type.
///
/// The algorithmic complexity of fully consuming the resulting iterator is
/// *O(N log(K))* where *N* is the total number of items that the input iterators
/// yield and *K* is the number of input iterators.
///
/// # Examples
///
/// ```
/// # extern crate maplit;
/// # use maplit::*;
/// # extern crate sorted_iter;
/// # use std::collections::BTreeSet;
/// use sorted_iter::multiway_union;
///
/// let sequences = vec![
///     btreeset! { 0, 5, 10, 15, 20, 25 }.into_iter(),
///     btreeset! { 0, 1, 4, 9, 16, 25, 36 }.into_iter(),
///     btreeset! { 4, 7, 11, 15, 18 }.into_iter(),
/// ];
///
/// assert_eq!(
///     multiway_union(sequences).collect::<BTreeSet<u64>>(),
///     btreeset! { 0, 1, 4, 5, 7, 9, 10, 11, 15, 16, 18, 20, 25, 36 }
/// );
/// ```
pub fn multiway_union<T, I>(iters: T) -> MultiwayUnion<I>
where
    I: SortedIterator,
    T: IntoIterator<Item = I>,
    I::Item: Ord,
{
    MultiwayUnion::from_iter(iters)
}

/// relational operations for iterators of pairs where the items are sorted according to the key
pub trait SortedPairIterator<K, V>: Iterator + Sized {
    fn join<W, J: SortedPairIterator<K, W>>(self, that: J) -> Join<Self, J> {
        Join {
            a: self.peekable(),
            b: that.peekable(),
        }
    }

    fn left_join<W, J: SortedPairIterator<K, W>>(self, that: J) -> LeftJoin<Self, J> {
        LeftJoin {
            a: self.peekable(),
            b: that.peekable(),
        }
    }

    fn right_join<W, J: SortedPairIterator<K, W>>(self, that: J) -> RightJoin<Self, J> {
        RightJoin {
            a: self.peekable(),
            b: that.peekable(),
        }
    }

    fn outer_join<W, J: SortedPairIterator<K, W>>(self, that: J) -> OuterJoin<Self, J> {
        OuterJoin {
            a: self.peekable(),
            b: that.peekable(),
        }
    }

    fn map_values<W, F: (FnMut(V) -> W)>(self, f: F) -> MapValues<Self, F> {
        MapValues { i: self, f }
    }

    fn filter_map_values<W, F: (FnMut(V) -> W)>(self, f: F) -> FilterMapValues<Self, F> {
        FilterMapValues { i: self, f }
    }

    fn keys(self) -> Keys<Self> {
        Keys { i: self }
    }
}

impl<K, V, I> SortedPairIterator<K, V> for I where I: Iterator<Item = (K, V)> + SortedByKey {}

pub mod assume {
    //! extension traits for unchecked conversions from iterators to sorted iterators
    use super::*;

    /// extension trait for any iterator to add a assume_sorted_by_item method
    pub trait AssumeSortedByItemExt: Iterator + Sized {
        /// assume that the iterator is sorted by its item order
        fn assume_sorted_by_item(self) -> AssumeSortedByItem<Self> {
            AssumeSortedByItem { i: self }
        }
    }

    impl<I: Iterator + Sized> AssumeSortedByItemExt for I {}

    /// extension trait for any iterator of pairs to add a assume_sorted_by_key method
    pub trait AssumeSortedByKeyExt: Iterator + Sized {
        fn assume_sorted_by_key(self) -> AssumeSortedByKey<Self> {
            AssumeSortedByKey { i: self }
        }
    }

    impl<K, V, I: Iterator<Item = (K, V)> + Sized> AssumeSortedByKeyExt for I {}
}

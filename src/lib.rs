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
//! For available set operations, see [SortedIterator](trait.SortedIterator.html).
//! For sorted iterators in the std lib, see instances the for [SortedByItem](trait.SortedByItem.html) marker trait.
//!
//! # Relational operations
//! ```
//! # extern crate maplit;
//! # use maplit::*;
//! # extern crate sorted_iter;
//! use sorted_iter::SortedPairIterator;
//!
//! let cities = btreemap! { 1 => "New York", 2 => "Tokyo", 3u8 => "Berlin" }.into_iter();
//! let countries = btreemap! { 1 => "USA", 2 => "Japan", 3u8 => "Germany" }.into_iter();
//! let cities_and_countries = cities.join(countries);
//! ```
//!
//! For available relational operations, see [SortedPairIterator](trait.SortedPairIterator.html).
//! For sorted iterators in the std lib, see instances the for [SortedByKey](trait.SortedByKey.html) marker trait.
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
//! let a = (1..31).map(|x| -x);
//! let b = (3..30).step_by(3);
//! let either = a.union(b); // does not compile!
//! ```
//!
#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

pub mod sorted_iterator;
pub mod sorted_pair_iterator;

use crate::sorted_iterator::*;
use crate::sorted_pair_iterator::*;
use std::cmp::Ordering::*;
use std::cmp::{max, min};
use std::iter::Peekable;

#[deny(missing_docs)]

/// set operations for iterators where the items are sorted according to the natural order
pub trait SortedIterator<T> {
    /// the iterator this SortedIterator extends
    type I: Iterator<Item = T>;
    /// union with another sorted iterator
    fn union<J: Iterator<Item = T> + SortedByItem>(self, that: J) -> Union<Self::I, J>;
    /// intersection with another sorted iterator
    fn intersection<J: Iterator<Item = T> + SortedByItem>(
        self,
        that: J,
    ) -> Intersection<Self::I, J>;
    /// difference with another sorted iterator
    fn difference<J: Iterator<Item = T> + SortedByItem>(self, that: J) -> Difference<Self::I, J>;
    /// symmetric difference with another sorted iterator
    fn symmetric_difference<J: Iterator<Item = T> + SortedByItem>(
        self,
        that: J,
    ) -> SymmetricDifference<Self::I, J>;
    /// pairs with unit value
    fn pairs(self) -> Pairs<Self::I>;
}

/// relational operations for iterators of pairs where the items are sorted according to the key
pub trait SortedPairIterator<K, V> {
    type I: Iterator<Item = (K, V)>;

    /// map values while leaving keys alone
    fn map_values<W, F: (FnMut(V) -> W)>(self, f: F) -> MapValues<Self::I, F>;

    /// filter_map values while leaving keys alone
    fn filter_map_values<W, F: (FnMut(V) -> W)>(self, f: F) -> FilterMapValues<Self::I, F>;

    /// keys as an iterator that is sorted by item
    fn keys(self) -> Keys<Self::I>;

    /// inner join with another sorted pair iterator
    fn join<W, J: Iterator<Item = (K, W)> + SortedByKey>(self, that: J) -> Join<Self::I, J>;

    /// left join with another sorted pair iterator
    fn left_join<W, J: Iterator<Item = (K, W)> + SortedByKey>(
        self,
        that: J,
    ) -> LeftJoin<Self::I, J>;

    /// right join with another sorted pair iterator
    fn right_join<W, J: Iterator<Item = (K, W)> + SortedByKey>(
        self,
        that: J,
    ) -> RightJoin<Self::I, J>;

    /// outer join with another sorted pair iterator
    fn outer_join<W, J: Iterator<Item = (K, W)> + SortedByKey>(
        self,
        that: J,
    ) -> OuterJoin<Self::I, J>;
}

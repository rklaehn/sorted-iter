//! This crate provides set and relational operations for all iterators in the standard library that are known
//! at compile time to be sorted.
//!
//! # Set operations
//! ```
//! # extern crate maplit;
//! # use maplit::*;
//! # extern crate sorted_iter;
//! use sorted_iter::SortedIteratorExt;
//!
//! let primes = btreeset! { 2, 3, 5, 7, 11, 13u64 }.into_iter();
//! let fibs = btreeset! { 1, 2, 3, 5, 8, 13u64 }.into_iter();
//! let fib_primes = primes.intersection(fibs);
//! ```
//!
//! For available set operations, see [SortedIterator](trait.SortedIteratorExt.html).
//! For sorted iterators in the std lib, see instances the for [SortedByItem](trait.SortedByItem.html) marker trait.
//!
//! # Relational operations
//! ```
//! # extern crate maplit;
//! # use maplit::*;
//! # extern crate sorted_iter;
//! use sorted_iter::SortedPairIteratorExt;
//!
//! let cities = btreemap! { 1 => "New York", 2 => "Tokyo", 3u8 => "Berlin" }.into_iter();
//! let countries = btreemap! { 1 => "USA", 2 => "Japan", 3u8 => "Germany" }.into_iter();
//! let cities_and_countries = cities.join(countries);
//! ```
//!
//! For available relational operations, see [SortedPairIterator](trait.SortedPairIteratorExt.html).
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
//! appropriate marker trait, [SortedByItem](trait.SortedByItem.html) or [SortedByKey](trait.SortedByKey.html).
//! 
//! ```
//! # extern crate sorted_iter;
//! // marker traits are not at top level, since usually you don't need them
//! use sorted_iter::sorted_iterator::SortedByItem;
//! use sorted_iter::sorted_pair_iterator::SortedByKey;
//!
//! struct MySortedIter<T> { whatever: T }
//! struct MySortedPairIter<K, V> { whatever: (K, V) }
//! 
//! impl<T> SortedByItem for MySortedIter<T> {}
//! impl<K, V> SortedByKey for MySortedPairIter<K, V> {}
//! ```
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
pub trait SortedIteratorExt: Iterator + Sized {
    /// union with another sorted iterator
    fn union<J>(self, that: J) -> Union<Self, J>
    where
        J: SortedIteratorExt<Item = Self::Item>,
    {
        Union {
            a: self.peekable(),
            b: that.peekable(),
        }
    }
    /// intersection with another sorted iterator
    fn intersection<J>(self, that: J) -> Intersection<Self, J>
    where
        J: SortedIteratorExt<Item = Self::Item>,
    {
        Intersection {
            a: self.peekable(),
            b: that.peekable(),
        }
    }
    /// difference with another sorted iterator
    fn difference<J>(self, that: J) -> Difference<Self, J>
    where
        J: SortedIteratorExt<Item = Self::Item>,
    {
        Difference {
            a: self.peekable(),
            b: that.peekable(),
        }
    }
    /// symmetric difference with another sorted iterator
    fn symmetric_difference<J>(self, that: J) -> SymmetricDifference<Self, J>
    where
        J: SortedIteratorExt<Item = Self::Item>,
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

impl<I> SortedIteratorExt for I where I: Iterator + SortedByItem {}

/// relational operations for iterators of pairs where the items are sorted according to the key
pub trait SortedPairIteratorExt<K, V>: Iterator + Sized {

    fn join<W, J: SortedPairIteratorExt<K, W>>(self, that: J) -> Join<Self, J> {
        Join {
            a: self.peekable(),
            b: that.peekable(),
        }
    }

    fn left_join<W, J: SortedPairIteratorExt<K, W>>(self, that: J) -> LeftJoin<Self, J> {
        LeftJoin {
            a: self.peekable(),
            b: that.peekable(),
        }
    }

    fn right_join<W, J: SortedPairIteratorExt<K, W>>(self, that: J) -> RightJoin<Self, J> {
        RightJoin {
            a: self.peekable(),
            b: that.peekable(),
        }
    }

    fn outer_join<W, J: SortedPairIteratorExt<K, W>>(self, that: J) -> OuterJoin<Self, J> {
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

impl<K, V, I> SortedPairIteratorExt<K, V> for I where I: Iterator<Item=(K, V)> + SortedByKey {}

pub mod assume {
    //! extension traits for unchecked conversions from iterators to sorted iterators
    use super::*;

    /// extension trait for any iterator to add a assume_sorted_by_item method
    pub trait AssumeSortedByItemExt : Iterator + Sized {
        /// assume that the iterator is sorted by its item order
        fn assume_sorted_by_item(self) -> AssumeSortedByItem<Self> {
            AssumeSortedByItem { i: self }
        }
    }

    impl<I: Iterator + Sized> AssumeSortedByItemExt for I {}

    /// extension trait for any iterator of pairs to add a assume_sorted_by_key method
    pub trait AssumeSortedByKeyExt : Iterator + Sized {
        fn assume_sorted_by_key(self) -> AssumeSortedByKey<Self> {
            AssumeSortedByKey { i: self }
        }
    }

    impl<K, V, I: Iterator<Item=(K, V)> + Sized> AssumeSortedByKeyExt for I {}
}
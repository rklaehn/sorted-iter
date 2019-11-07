#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

mod sorted_iterator;
mod sorted_pair_iterator;

use crate::sorted_iterator::*;
use crate::sorted_pair_iterator::*;
use std::cmp::Ordering::*;
use std::cmp::{max, min};
use std::iter::Peekable;

/// set operations for sorted iterators
pub trait SortedIterator<T> {
    type I: Iterator<Item = T>;
    fn union<J: Iterator<Item = T> + SortedByItem>(self, that: J) -> Union<Self::I, J>;
    fn intersection<J: Iterator<Item = T> + SortedByItem>(
        self,
        that: J,
    ) -> Intersection<Self::I, J>;
    fn difference<J: Iterator<Item = T> + SortedByItem>(self, that: J) -> Difference<Self::I, J>;
    fn symmetric_difference<J: Iterator<Item = T> + SortedByItem>(
        self,
        that: J,
    ) -> SymmetricDifference<Self::I, J>;
    fn pairs(self) -> Pairs<Self::I>;
}

/// relational operations for sorted iterators of pairs
pub trait SortedPairIterator<K, V> {
    type I: Iterator<Item = (K, V)>;

    fn map_values<W, F: (FnMut(V) -> W)>(self, f: F) -> MapValues<Self::I, F>;

    fn filter_map_values<W, F: (FnMut(V) -> W)>(self, f: F) -> FilterMapValues<Self::I, F>;

    fn keys(self) -> Keys<Self::I>;

    fn join<W, J: Iterator<Item = (K, W)> + SortedByKey>(self, that: J) -> Join<Self::I, J>;

    fn left_join<W, J: Iterator<Item = (K, W)> + SortedByKey>(
        self,
        that: J,
    ) -> LeftJoin<Self::I, J>;

    fn right_join<W, J: Iterator<Item = (K, W)> + SortedByKey>(
        self,
        that: J,
    ) -> RightJoin<Self::I, J>;

    fn outer_join<W, J: Iterator<Item = (K, W)> + SortedByKey>(
        self,
        that: J,
    ) -> OuterJoin<Self::I, J>;
}

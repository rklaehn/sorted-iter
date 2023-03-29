//! implementation of the sorted_pair_iterator relational operations
use super::*;
use crate::sorted_iterator::SortedByItem;
use core::{
    cmp::Ordering::*,
    cmp::{max, min},
    fmt::Debug,
    iter,
    iter::Peekable,
};
use std::{collections, iter::FusedIterator};

/// marker trait for iterators that are sorted by the key of their Item
pub trait SortedByKey {}

pub struct Join<I: Iterator, J: Iterator> {
    pub(crate) a: Peekable<I>,
    pub(crate) b: Peekable<J>,
}

impl<K, A, B, I, J> Iterator for Join<I, J>
where
    K: Ord,
    I: Iterator<Item = (K, A)> + SortedByKey,
    J: Iterator<Item = (K, B)> + SortedByKey,
{
    type Item = (K, (A, B));

    fn next(&mut self) -> Option<Self::Item> {
        while let (Some((ak, _)), Some((bk, _))) = (self.a.peek(), self.b.peek()) {
            match ak.cmp(&bk) {
                Less => {
                    self.a.next();
                }
                Greater => {
                    self.b.next();
                }
                Equal => {
                    if let (Some((ak, av)), Some((_, bv))) = (self.a.next(), self.b.next()) {
                        return Some((ak, (av, bv)));
                    } else {
                        unreachable!();
                    }
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, amax) = self.a.size_hint();
        let (_, bmax) = self.b.size_hint();
        let rmin = 0;
        let rmax = amax.and_then(|amax| bmax.map(|bmax| min(amax, bmax)));
        (rmin, rmax)
    }
}

impl<I: Iterator + Clone, J: Iterator + Clone> Clone for Join<I, J>
where
    I::Item: Clone,
    J::Item: Clone,
{
    fn clone(&self) -> Self {
        Self {
            a: self.a.clone(),
            b: self.b.clone(),
        }
    }
}

pub struct LeftJoin<I: Iterator, J: Iterator> {
    pub(crate) a: Peekable<I>,
    pub(crate) b: Peekable<J>,
}

impl<K, A, B, I, J> Iterator for LeftJoin<I, J>
where
    K: Ord,
    I: Iterator<Item = (K, A)>,
    J: Iterator<Item = (K, B)>,
{
    type Item = (K, (A, Option<B>));

    fn next(&mut self) -> Option<Self::Item> {
        let (ak, av) = self.a.next()?;
        while let Some((bk, _)) = self.b.peek() {
            match ak.cmp(bk) {
                Less => break,
                Greater => {
                    self.b.next();
                }
                Equal => {
                    let (_, bv) = self.b.next()?;
                    return Some((ak, (av, Some(bv))));
                }
            }
        }
        Some((ak, (av, None)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.a.size_hint()
    }
}

impl<I: Iterator + Clone, J: Iterator + Clone> Clone for LeftJoin<I, J>
where
    I::Item: Clone,
    J::Item: Clone,
{
    fn clone(&self) -> Self {
        Self {
            a: self.a.clone(),
            b: self.b.clone(),
        }
    }
}

pub struct RightJoin<I: Iterator, J: Iterator> {
    pub(crate) a: Peekable<I>,
    pub(crate) b: Peekable<J>,
}

impl<K, A, B, I, J> Iterator for RightJoin<I, J>
where
    K: Ord,
    I: Iterator<Item = (K, A)>,
    J: Iterator<Item = (K, B)>,
{
    type Item = (K, (Option<A>, B));

    fn next(&mut self) -> Option<Self::Item> {
        let (bk, bv) = self.b.next()?;
        while let Some((ak, _)) = self.a.peek() {
            match bk.cmp(ak) {
                Less => break,
                Greater => {
                    self.a.next();
                }
                Equal => {
                    let (_, av) = self.a.next()?;
                    return Some((bk, (Some(av), bv)));
                }
            }
        }
        Some((bk, (None, bv)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.b.size_hint()
    }
}

impl<I: Iterator + Clone, J: Iterator + Clone> Clone for RightJoin<I, J>
where
    I::Item: Clone,
    J::Item: Clone,
{
    fn clone(&self) -> Self {
        Self {
            a: self.a.clone(),
            b: self.b.clone(),
        }
    }
}

pub struct OuterJoin<I: Iterator, J: Iterator> {
    pub(crate) a: Peekable<I>,
    pub(crate) b: Peekable<J>,
}

// all this just so I could avoid having this expression twice in the iterator.
// Sometimes making things DRY in rust is hard...
impl<K, A, B, I, J> OuterJoin<I, J>
where
    K: Ord,
    I: Iterator<Item = (K, A)>,
    J: Iterator<Item = (K, B)>,
{
    // type alias does not help much here...
    #[allow(clippy::type_complexity)]
    fn next_a(&mut self) -> Option<(K, (Option<A>, Option<B>))> {
        self.a.next().map(|(ak, av)| (ak, (Some(av), None)))
    }

    #[allow(clippy::type_complexity)]
    fn next_b(&mut self) -> Option<(K, (Option<A>, Option<B>))> {
        self.b.next().map(|(bk, bv)| (bk, (None, Some(bv))))
    }
}

impl<K, A, B, I, J> Iterator for OuterJoin<I, J>
where
    K: Ord,
    I: Iterator<Item = (K, A)>,
    J: Iterator<Item = (K, B)>,
{
    type Item = (K, (Option<A>, Option<B>));

    fn next(&mut self) -> Option<Self::Item> {
        if let (Some((ak, _)), Some((bk, _))) = (self.a.peek(), self.b.peek()) {
            match ak.cmp(&bk) {
                Less => self.next_a(),
                Greater => self.next_b(),
                Equal => self
                    .a
                    .next()
                    .and_then(|(ak, av)| self.b.next().map(|(_, bv)| (ak, (Some(av), Some(bv))))),
            }
        } else {
            self.next_a().or_else(|| self.next_b())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (amin, amax) = self.a.size_hint();
        let (bmin, bmax) = self.b.size_hint();
        let rmin = max(amin, bmin);
        let rmax = amax.and_then(|amax| bmax.map(|bmax| amax + bmax));
        (rmin, rmax)
    }
}

impl<I: Iterator + Clone, J: Iterator + Clone> Clone for OuterJoin<I, J>
where
    I::Item: Clone,
    J::Item: Clone,
{
    fn clone(&self) -> Self {
        Self {
            a: self.a.clone(),
            b: self.b.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Keys<I: Iterator> {
    pub(crate) i: I,
}

impl<K, V, I> Iterator for Keys<I>
where
    K: Ord,
    I: Iterator<Item = (K, V)>,
{
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        self.i.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.i.size_hint()
    }
}

#[derive(Clone, Debug)]
pub struct MapValues<I: Iterator, F> {
    pub(crate) i: I,
    pub(crate) f: F,
}

impl<K, V, W, I, F> Iterator for MapValues<I, F>
where
    K: Ord,
    I: Iterator<Item = (K, V)>,
    F: FnMut(V) -> W,
{
    type Item = (K, W);

    fn next(&mut self) -> Option<Self::Item> {
        self.i.next().map(|(k, v)| (k, (self.f)(v)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.i.size_hint()
    }
}

#[derive(Clone, Debug)]
pub struct FilterMapValues<I: Iterator, F> {
    pub(crate) i: I,
    pub(crate) f: F,
}

impl<K, V, W, I, F> Iterator for FilterMapValues<I, F>
where
    K: Ord,
    I: Iterator<Item = (K, V)>,
    F: FnMut(V) -> Option<W>,
{
    type Item = (K, W);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((k, v)) = self.i.next() {
            if let Some(w) = (self.f)(v) {
                return Some((k, w));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, imax) = self.i.size_hint();
        (0, imax)
    }
}

#[derive(Clone, Debug)]
pub struct AssumeSortedByKey<I: Iterator> {
    pub(crate) i: I,
}

impl<I: Iterator> Iterator for AssumeSortedByKey<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.i.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.i.size_hint()
    }
}

impl<I: Iterator> ExactSizeIterator for AssumeSortedByKey<I> where I: ExactSizeIterator {}

impl<I: Iterator> FusedIterator for AssumeSortedByKey<I> where I: FusedIterator {}

impl<I: Iterator> DoubleEndedIterator for AssumeSortedByKey<I>
where
    I: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.i.next_back()
    }
}

// not sure how to model this: an iterator that is sorted by key automatically sorted by item as well,
// since the key comes first in the order of a pair. The reverse is not true, since e.g. (0,1), (0,2)
// are sorted by item, but not strictly sorted by key!

// mark common std traits
impl<I> SortedByKey for iter::Empty<I> {}
impl<I> SortedByKey for iter::Once<I> {}
impl<I> SortedByKey for iter::Enumerate<I> {}

impl<I: Iterator + SortedByItem> SortedByKey for Pairs<I> {}
impl<I: Iterator, F> SortedByKey for MapValues<I, F> {}
impl<I: Iterator, F> SortedByKey for FilterMapValues<I, F> {}

impl<I: SortedByKey> SortedByKey for iter::Take<I> {}
impl<I: SortedByKey> SortedByKey for iter::Skip<I> {}
impl<I: SortedByKey> SortedByKey for iter::StepBy<I> {}
impl<I: SortedByKey> SortedByKey for iter::Cloned<I> {}
impl<I: SortedByKey> SortedByKey for iter::Copied<I> {}
impl<I: SortedByKey> SortedByKey for iter::Fuse<I> {}
impl<I: SortedByKey, F> SortedByKey for iter::Inspect<I, F> {}
impl<I: SortedByKey, P> SortedByKey for iter::TakeWhile<I, P> {}
impl<I: SortedByKey, P> SortedByKey for iter::SkipWhile<I, P> {}
impl<I: SortedByKey, P> SortedByKey for iter::Filter<I, P> {}
impl<I: SortedByKey + Iterator> SortedByKey for iter::Peekable<I> {}
impl<I: SortedByItem, J> SortedByKey for iter::Zip<I, J> {}

impl<I: Iterator, J: Iterator> SortedByKey for Join<I, J> {}
impl<I: Iterator, J: Iterator> SortedByKey for LeftJoin<I, J> {}
impl<I: Iterator, J: Iterator> SortedByKey for RightJoin<I, J> {}
impl<I: Iterator, J: Iterator> SortedByKey for OuterJoin<I, J> {}
impl<I: Iterator> SortedByKey for AssumeSortedByKey<I> {}

impl<K, V> SortedByKey for collections::btree_map::IntoIter<K, V> {}
impl<'a, K, V> SortedByKey for collections::btree_map::Iter<'a, K, V> {}
impl<'a, K, V> SortedByKey for collections::btree_map::IterMut<'a, K, V> {}
impl<'a, K, V> SortedByKey for collections::btree_map::Range<'a, K, V> {}
impl<'a, K, V> SortedByKey for collections::btree_map::RangeMut<'a, K, V> {}

impl<I: SortedByKey> SortedByKey for Box<I> {}

#[cfg(test)]
mod tests {
    extern crate maplit;
    use super::*;
    use collections::BTreeMap;
    use core::fmt::Debug;
    use maplit::*;

    /// just a helper to get good output when a check fails
    fn unary_op<E: Debug, R: Eq + Debug>(x: E, expected: R, actual: R) -> bool {
        let res = expected == actual;
        if !res {
            println!("x:{:?} expected:{:?} actual:{:?}", x, expected, actual);
        }
        res
    }

    /// just a helper to get good output when a check fails
    fn binary_op<E: Debug, R: Eq + Debug>(a: E, b: E, expected: R, actual: R) -> bool {
        let res = expected == actual;
        if !res {
            println!(
                "a:{:?} b:{:?} expected:{:?} actual:{:?}",
                a, b, expected, actual
            );
        }
        res
    }

    type Element = i64;
    type Reference = BTreeMap<Element, Element>;

    #[quickcheck]
    fn join(a: Reference, b: Reference) -> bool {
        type Result = BTreeMap<Element, (Element, Element)>;
        let expected: Result = a
            .keys()
            .intersection(b.keys())
            .map(|k| (k.clone(), (a[k], b[k])))
            .collect();
        let actual: Result = a.clone().into_iter().join(b.clone().into_iter()).collect();
        binary_op(a, b, expected, actual)
    }

    #[quickcheck]
    fn left_join(a: Reference, b: Reference) -> bool {
        type Result = BTreeMap<Element, (Element, Option<Element>)>;
        let expected: Result = a
            .keys()
            .map(|k| (k.clone(), (a[k], b.get(k).cloned())))
            .collect();
        let actual: Result = a
            .clone()
            .into_iter()
            .left_join(b.clone().into_iter())
            .collect();
        binary_op(a, b, expected, actual)
    }

    #[quickcheck]
    fn right_join(a: Reference, b: Reference) -> bool {
        type Result = BTreeMap<Element, (Option<Element>, Element)>;
        let expected: Result = b
            .keys()
            .map(|k| (k.clone(), (a.get(k).cloned(), b[k])))
            .collect();
        let actual: Result = a
            .clone()
            .into_iter()
            .right_join(b.clone().into_iter())
            .collect();
        binary_op(a, b, expected, actual)
    }

    #[quickcheck]
    fn outer_join(a: Reference, b: Reference) -> bool {
        type Result = BTreeMap<Element, (Option<Element>, Option<Element>)>;
        let expected: Result = a
            .keys()
            .union(b.keys())
            .map(|k| (k.clone(), (a.get(k).cloned(), b.get(k).cloned())))
            .collect();
        let actual: Result = a
            .clone()
            .into_iter()
            .outer_join(b.clone().into_iter())
            .collect();
        binary_op(a, b, expected, actual)
    }

    #[quickcheck]
    fn map_values(x: Reference) -> bool {
        type Result = BTreeMap<Element, Element>;
        let expected: Result = x.clone().into_iter().map(|(k, v)| (k, v * 2)).collect();
        let actual: Result = x.clone().into_iter().map_values(|v| v * 2).collect();
        unary_op(x, expected, actual)
    }

    #[quickcheck]
    fn filter_map_values(x: Reference) -> bool {
        type Result = BTreeMap<Element, Element>;
        let expected: Result = x
            .clone()
            .into_iter()
            .filter_map(|(k, v)| if v % 2 != 0 { Some((k, v * 2)) } else { None })
            .collect();
        let actual: Result = x
            .clone()
            .into_iter()
            .filter_map_values(|v| if v % 2 != 0 { Some(v * 2) } else { None })
            .collect();
        unary_op(x, expected, actual)
    }

    fn is_s<K, V, I: Iterator<Item = (K, V)> + SortedByKey>(_v: I) {}

    fn s() -> impl Iterator<Item = (i64, ())> + SortedByKey {
        (0i64..10).pairs()
    }

    #[test]
    fn instances() {
        // creation
        is_s(iter::empty::<(i64, ())>());
        is_s(iter::once((0, ())));
        is_s([1, 2, 3, 4].iter().enumerate());
        // ranges
        is_s((0i64..10).pairs());
        is_s((0i64..=10).pairs());
        is_s((0i64..).pairs());
        // wrappers
        is_s(Box::new((0i64..).pairs()));
        // skip/take/step/filter
        is_s(s().step_by(1));
        is_s(s().take(1));
        is_s(s().skip(1));
        is_s(s().take_while(|_| true));
        is_s(s().skip_while(|_| true));
        is_s(s().filter(|_| true));
        // identity
        is_s(s().peekable());
        is_s(s().fuse());
        is_s(s().inspect(|_| {}));
        // relational
        is_s(s().join(s()));
        is_s(s().left_join(s()));
        is_s(s().right_join(s()));
        is_s(s().outer_join(s()));
        // btreeset
        is_s(btreemap! { 0i64 => "" }.iter());
        is_s(btreemap! { 0i64 => "" }.into_iter());
        is_s(btreemap! { 0i64 => "" }.iter_mut());
        is_s(btreemap! { 0i64 => "" }.range(..));
        is_s(btreemap! { 0i64 => "" }.range_mut(..));
    }
}

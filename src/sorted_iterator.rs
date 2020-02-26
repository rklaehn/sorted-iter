//! implementation of the sorted_iterator set operations
use super::*;
use std::iter::Peekable;
use std::cmp::{max, min, Ordering, Reverse};
use std::cmp::Ordering::*;
use std::collections::BinaryHeap;

/// marker trait for iterators that are sorted by their Item
pub trait SortedByItem {}

pub struct Union<I: Iterator, J: Iterator> {
    pub(crate) a: Peekable<I>,
    pub(crate) b: Peekable<J>,
}

impl<I: Iterator + Clone, J: Iterator + Clone> Clone for Union<I, J>
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

impl<K: Ord, I: Iterator<Item = K>, J: Iterator<Item = K>> Iterator for Union<I, J> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        if let (Some(ak), Some(bk)) = (self.a.peek(), self.b.peek()) {
            match ak.cmp(&bk) {
                Less => self.a.next(),
                Greater => self.b.next(),
                Equal => {
                    self.b.next();
                    self.a.next()
                }
            }
        } else {
            self.a.next().or_else(|| self.b.next())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (amin, amax) = self.a.size_hint();
        let (bmin, bmax) = self.b.size_hint();
        // full overlap
        let rmin = max(amin, bmin);
        // no overlap
        let rmax = amax.and_then(|amax| bmax.map(|bmax| amax + bmax));
        (rmin, rmax)
    }
}

// An iterator with the first item pulled out.
pub(crate) struct Peeked<I: Iterator>{
    h: Reverse<I::Item>,
    t: I,
}

impl<I: Iterator> Peeked<I> {
    fn new(mut i: I) -> Option<Peeked<I>> {
        i.next().map(|x| Peeked { h: Reverse(x), t: i })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.t.size_hint();
        (lo + 1, hi.map(|hi| hi + 1))
    }
}

// Delegate comparisons to the head element.
impl<I: Iterator> PartialEq for Peeked<I> where I::Item: PartialEq {
    fn eq(&self, that: &Self) -> bool { self.h.eq(&that.h) }
}

impl<I: Iterator> Eq for Peeked<I> where I::Item: Eq {}

impl<I: Iterator> PartialOrd for Peeked<I> where I::Item: PartialOrd {
    fn partial_cmp(&self, that: &Self) -> Option<Ordering> {
        self.h.partial_cmp(&that.h)
    }
}

impl<I: Iterator> Ord for Peeked<I> where I::Item: Ord {
    fn cmp(&self, that: &Self) -> std::cmp::Ordering { self.h.cmp(&that.h) }
}

impl<I: Iterator + Clone> Clone for Peeked<I> where I::Item: Clone {
    fn clone(&self) -> Self { Self { h: self.h.clone(), t: self.t.clone() } }
}

pub struct MultiwayUnion<I: Iterator> {
    pub(crate) bh: BinaryHeap<Peeked<I>>,
}

impl<I: Iterator> MultiwayUnion<I> where I::Item: Ord {
    pub(crate) fn from_iter<T: IntoIterator<Item = I>>(x: T)
            -> MultiwayUnion<I> {
        MultiwayUnion { bh: x.into_iter().filter_map(Peeked::new).collect() }
    }
}

impl<I: Iterator + Clone> Clone for MultiwayUnion<I> where I::Item: Clone {
    fn clone(&self) -> Self { Self { bh: self.bh.clone() } }
}

impl<I: Iterator> Iterator for MultiwayUnion<I> where I::Item: Ord {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        // Extract the current minimum element.
        self.bh.pop().map(|Peeked { h: cur, t: tail }| {
            // Collect elements equivalent to the current element.
            // This is done as a two-step process so we don't remove
            // duplicate elements from any of the input iterators.
            let mut eqs = Vec::new();
            let mut add_eq = |i| Peeked::new(i).map(|e| eqs.push(e));
            add_eq(tail);
            while let Some(item) = self.bh.peek() {
                if item.h != cur {
                    break;
                } else {
                    let Peeked { t, .. } = self.bh.pop().unwrap();
                    add_eq(t);
                }
            }
            // Re-insert iterators originally containing equivalent elements.
            self.bh.extend(eqs.drain(..));
            // Return the current minimum, removing the Reverse wrapper.
            cur.0
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.bh.iter().fold((0, Some(0)), |(lo, hi), it| {
            let (ilo, ihi) = it.size_hint();
            (max(lo, ilo), hi.and_then(|hi| ihi.map(|ihi| hi + ihi)))
        })
    }
}

pub struct Intersection<I: Iterator, J: Iterator> {
    pub(crate) a: Peekable<I>,
    pub(crate) b: Peekable<J>,
}

impl<I: Iterator + Clone, J: Iterator + Clone> Clone for Intersection<I, J>
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

impl<K: Ord, I: Iterator<Item = K>, J: Iterator<Item = K>> Iterator for Intersection<I, J> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        while let (Some(a), Some(b)) = (self.a.peek(), self.b.peek()) {
            match a.cmp(&b) {
                Less => {
                    self.a.next();
                }
                Greater => {
                    self.b.next();
                }
                Equal => {
                    self.b.next();
                    return self.a.next();
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, amax) = self.a.size_hint();
        let (_, bmax) = self.b.size_hint();
        // no overlap
        let rmin = 0;
        // full overlap
        let rmax = amax.and_then(|amax| bmax.map(|bmax| min(amax, bmax)));
        (rmin, rmax)
    }
}

pub struct Difference<I: Iterator, J: Iterator> {
    pub(crate) a: Peekable<I>,
    pub(crate) b: Peekable<J>,
}

impl<I: Iterator + Clone, J: Iterator + Clone> Clone for Difference<I, J>
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

impl<K: Ord, I: Iterator<Item = K>, J: Iterator<Item = K>> Iterator for Difference<I, J> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(b) = self.b.peek() {
            // if we have no more a, return none
            let a = self.a.peek()?;
            match a.cmp(b) {
                Less => {
                    break;
                }
                Greater => {
                    self.b.next();
                }
                Equal => {
                    self.a.next();
                    self.b.next();
                }
            }
        }
        self.a.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (amin, amax) = self.a.size_hint();
        let (_, bmax) = self.b.size_hint();
        // no overlap
        let rmax = amax;
        // if the other has at most bmax elements, and we have at least amin elements
        let rmin = bmax.map(|bmax| amin.saturating_sub(bmax)).unwrap_or(0);
        (rmin, rmax)
    }
}

pub struct SymmetricDifference<I: Iterator, J: Iterator> {
    pub(crate) a: Peekable<I>,
    pub(crate) b: Peekable<J>,
}

impl<I: Iterator + Clone, J: Iterator + Clone> Clone for SymmetricDifference<I, J>
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

impl<K: Ord, I: Iterator<Item = K>, J: Iterator<Item = K>> Iterator for SymmetricDifference<I, J> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        while let (Some(ak), Some(bk)) = (self.a.peek(), self.b.peek()) {
            match ak.cmp(&bk) {
                Less => return self.a.next(),
                Greater => return self.b.next(),
                Equal => {
                    self.b.next();
                    self.a.next();
                }
            }
        }
        self.a.next().or_else(|| self.b.next())
    }

    // TODO!
    // fn size_hint(&self) -> (usize, Option<usize>) {
    // }
}

#[derive(Clone, Debug)]
pub struct Pairs<I: Iterator> {
    pub(crate) i: I,
}

impl<I: Iterator> Iterator for Pairs<I> {
    type Item = (I::Item, ());

    fn next(&mut self) -> Option<Self::Item> {
        self.i.next().map(|k| (k, ()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.i.size_hint()
    }
}

#[derive(Clone, Debug)]
pub struct AssumeSortedByItem<I: Iterator> {
    pub(crate) i: I,
}

impl<I: Iterator> Iterator for AssumeSortedByItem<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.i.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.i.size_hint()
    }
}

// mark common std traits
impl<I> SortedByItem for std::iter::Empty<I> {}
impl<I> SortedByItem for std::iter::Once<I> {}
impl<I: SortedByItem> SortedByItem for std::iter::Take<I> {}
impl<I: SortedByItem> SortedByItem for std::iter::Skip<I> {}
impl<I: SortedByItem> SortedByItem for std::iter::StepBy<I> {}
impl<I: SortedByItem> SortedByItem for std::iter::Cloned<I> {}
impl<I: SortedByItem> SortedByItem for std::iter::Copied<I> {}
impl<I: SortedByItem> SortedByItem for std::iter::Fuse<I> {}
impl<I: SortedByItem, F> SortedByItem for std::iter::Inspect<I, F> {}
impl<I: SortedByItem, P> SortedByItem for std::iter::TakeWhile<I, P> {}
impl<I: SortedByItem, P> SortedByItem for std::iter::SkipWhile<I, P> {}
impl<I: SortedByItem, P> SortedByItem for std::iter::Filter<I, P> {}
impl<I: SortedByItem + Iterator> SortedByItem for std::iter::Peekable<I> {}

impl<T> SortedByItem for std::collections::btree_set::IntoIter<T> {}
impl<'a, T> SortedByItem for std::collections::btree_set::Iter<'a, T> {}
impl<'a, T> SortedByItem for std::collections::btree_set::Intersection<'a, T> {}
impl<'a, T> SortedByItem for std::collections::btree_set::Union<'a, T> {}
impl<'a, T> SortedByItem for std::collections::btree_set::Difference<'a, T> {}
impl<'a, T> SortedByItem for std::collections::btree_set::SymmetricDifference<'a, T> {}
impl<'a, T> SortedByItem for std::collections::btree_set::Range<'a, T> {}

impl<'a, K, V> SortedByItem for std::collections::btree_map::Keys<'a, K, V> {}

impl<T> SortedByItem for std::ops::Range<T> {}
impl<T> SortedByItem for std::ops::RangeInclusive<T> {}
impl<T> SortedByItem for std::ops::RangeFrom<T> {}

impl<I: Iterator> SortedByItem for Keys<I> {}
impl<I: Iterator> SortedByItem for AssumeSortedByItem<I> {}
impl<I: Iterator, J: Iterator> SortedByItem for Union<I, J> {}
impl<I: Iterator, J: Iterator> SortedByItem for Intersection<I, J> {}
impl<I: Iterator, J: Iterator> SortedByItem for Difference<I, J> {}
impl<I: Iterator, J: Iterator> SortedByItem for SymmetricDifference<I, J> {}
impl<I: Iterator> SortedByItem for MultiwayUnion<I> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

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
    type Reference = std::collections::BTreeSet<Element>;

    #[quickcheck]
    fn intersection(a: Reference, b: Reference) -> bool {
        let expected: Reference = a.intersection(&b).cloned().collect();
        let actual: Reference = a
            .clone()
            .into_iter()
            .intersection(b.clone().into_iter())
            .collect();
        binary_op(a, b, expected, actual)
    }

    #[quickcheck]
    fn union(a: Reference, b: Reference) -> bool {
        let expected: Reference = a.union(&b).cloned().collect();
        let actual: Reference = a.clone().into_iter().union(b.clone().into_iter()).collect();
        binary_op(a, b, expected, actual)
    }

    #[quickcheck]
    fn unions(inputs: Vec<Reference>) -> bool {
        let expected: Reference = inputs.iter().flatten().copied().collect();
        let actual = MultiwayUnion::from_iter(inputs.iter().map(|i| i.iter()));
        let res = actual.clone().eq(expected.iter());
        if !res {
            let actual: Reference = actual.copied().collect();
            println!("in:{:?} expected:{:?} out:{:?}", inputs, expected, actual);
        }
        res
    }

    #[quickcheck]
    fn difference(a: Reference, b: Reference) -> bool {
        let expected: Reference = a.difference(&b).cloned().collect();
        let actual: Reference = a
            .clone()
            .into_iter()
            .difference(b.clone().into_iter())
            .collect();
        binary_op(a, b, expected, actual)
    }

    #[quickcheck]
    fn symmetric_difference(a: Reference, b: Reference) -> bool {
        let expected: Reference = a.symmetric_difference(&b).cloned().collect();
        let actual: Reference = a
            .clone()
            .into_iter()
            .symmetric_difference(b.clone().into_iter())
            .collect();
        binary_op(a, b, expected, actual)
    }

    fn s() -> impl Iterator<Item = i64> + SortedByItem {
        (0i64..10)
    }
    fn r<'a>() -> impl Iterator<Item = &'a i64> + SortedByItem {
        std::iter::empty()
    }
    fn is_s<K, I: Iterator<Item = K> + SortedByItem>(_v: I) {}

    #[test]
    fn instances() {
        is_s(std::iter::empty::<i64>());
        is_s(std::iter::once(0u64));
        // ranges
        is_s(0i64..10);
        is_s(0i64..=10);
        is_s(0i64..);
        // identity
        is_s(s().fuse());
        is_s(r().cloned());
        is_s(r().copied());
        is_s(r().peekable());
        is_s(s().inspect(|_| {}));
        // removing items
        is_s(s().step_by(2));
        is_s(s().take(1));
        is_s(s().take_while(|_| true));
        is_s(s().skip(1));
        is_s(s().skip_while(|_| true));
        is_s(s().filter(|_| true));
        // set ops
        is_s(s().union(s()));
        is_s(s().intersection(s()));
        is_s(s().difference(s()));
        is_s(s().symmetric_difference(s()));
        is_s(multiway_union(vec![s(), s(), s()]));
        is_s(multiway_union(std::iter::once(s())));
    }
}

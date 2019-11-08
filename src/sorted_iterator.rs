use super::*;

/// marker trait for iterators that are sorted by their Item
pub trait SortedByItem {}

impl<T: Ord, I: Iterator<Item = T> + SortedByItem> SortedIterator<T> for I {
    type I = I;
    fn union<J: Iterator<Item = T> + SortedByItem>(self, that: J) -> Union<I, J> {
        Union {
            a: self.peekable(),
            b: that.peekable(),
        }
    }
    fn intersection<J: Iterator<Item = T> + SortedByItem>(self, that: J) -> Intersection<I, J> {
        Intersection {
            a: self.peekable(),
            b: that.peekable(),
        }
    }
    fn difference<J: Iterator<Item = T> + SortedByItem>(self, that: J) -> Difference<I, J> {
        Difference {
            a: self.peekable(),
            b: that.peekable(),
        }
    }
    fn symmetric_difference<J: Iterator<Item = T> + SortedByItem>(
        self,
        that: J,
    ) -> SymmetricDifference<I, J> {
        SymmetricDifference {
            a: self.peekable(),
            b: that.peekable(),
        }
    }
    fn pairs(self) -> Pairs<Self::I> {
        Pairs { i: self }
    }
}

pub struct Union<I: Iterator, J: Iterator> {
    a: Peekable<I>,
    b: Peekable<J>,
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

pub struct Intersection<I: Iterator, J: Iterator> {
    a: Peekable<I>,
    b: Peekable<J>,
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
    a: Peekable<I>,
    b: Peekable<J>,
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
    a: Peekable<I>,
    b: Peekable<J>,
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
    i: I,
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
impl<I: Iterator, J: Iterator> SortedByItem for Union<I, J> {}
impl<I: Iterator, J: Iterator> SortedByItem for Intersection<I, J> {}
impl<I: Iterator, J: Iterator> SortedByItem for Difference<I, J> {}
impl<I: Iterator, J: Iterator> SortedByItem for SymmetricDifference<I, J> {}

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
    }
}

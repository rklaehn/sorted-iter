//! Implementation of [`OneOrLess`] iterators.

use core::{iter, option, result};

/// Iterators that return either 1 or 0 item.
///
/// An iterator that returns nothing or one thing is always sorted.
///
/// More interestingly, `one_or_less_iterator.flatten()` is sorted if the inner
/// iterator is itself sorted. This isn't true of non-one_or_less iterators.
pub trait OneOrLess {}

impl<I> OneOrLess for iter::Empty<I> {}
impl<I> OneOrLess for iter::Once<I> {}
impl<'a, T> OneOrLess for option::Iter<'a, T> {}
impl<'a, T> OneOrLess for result::Iter<'a, T> {}
impl<T> OneOrLess for option::IntoIter<T> {}
impl<T> OneOrLess for result::IntoIter<T> {}

impl<I: OneOrLess> OneOrLess for Box<I> {}
impl<I: OneOrLess> OneOrLess for iter::Take<I> {}
impl<I: OneOrLess> OneOrLess for iter::Skip<I> {}
impl<I: OneOrLess> OneOrLess for iter::StepBy<I> {}
impl<I: OneOrLess> OneOrLess for iter::Cloned<I> {}
impl<I: OneOrLess> OneOrLess for iter::Copied<I> {}
impl<I: OneOrLess> OneOrLess for iter::Fuse<I> {}
impl<I: OneOrLess, F> OneOrLess for iter::Inspect<I, F> {}
impl<I: OneOrLess, P> OneOrLess for iter::TakeWhile<I, P> {}
impl<I: OneOrLess, P> OneOrLess for iter::SkipWhile<I, P> {}
impl<I: OneOrLess, P> OneOrLess for iter::Filter<I, P> {}
impl<I: OneOrLess, P> OneOrLess for iter::FilterMap<I, P> {}
impl<I: OneOrLess, P> OneOrLess for iter::Map<I, P> {}
impl<I: OneOrLess + Iterator> OneOrLess for iter::Peekable<I> {}
impl<Iin, J, Iout, F> OneOrLess for iter::FlatMap<Iin, J, F>
where
    Iin: OneOrLess,
    J: IntoIterator<IntoIter = Iout>,
    Iout: OneOrLess,
{
}
impl<Iin, J, Iout> OneOrLess for iter::Flatten<Iin>
where
    Iin: OneOrLess + Iterator<Item = J>,
    J: IntoIterator<IntoIter = Iout>,
    Iout: OneOrLess,
{
}

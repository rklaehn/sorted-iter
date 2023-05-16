//! Implementation of [`Affine`] iterators.

use core::{iter, option, result};

/// Iterators that return either 1 or 0 item.
///
/// An iterator that returns nothing or one thing is always sorted.
///
/// More interestingly, `affine_iterator.flatten()` is sorted if the inner
/// iterator is itself sorted. This isn't true of non-affine iterators.
pub trait Affine {}

impl<I> Affine for iter::Empty<I> {}
impl<I> Affine for iter::Once<I> {}
impl<'a, T> Affine for option::Iter<'a, T> {}
impl<'a, T> Affine for result::Iter<'a, T> {}
impl<T> Affine for option::IntoIter<T> {}
impl<T> Affine for result::IntoIter<T> {}

impl<I: Affine> Affine for Box<I> {}
impl<I: Affine> Affine for iter::Take<I> {}
impl<I: Affine> Affine for iter::Skip<I> {}
impl<I: Affine> Affine for iter::StepBy<I> {}
impl<I: Affine> Affine for iter::Cloned<I> {}
impl<I: Affine> Affine for iter::Copied<I> {}
impl<I: Affine> Affine for iter::Fuse<I> {}
impl<I: Affine, F> Affine for iter::Inspect<I, F> {}
impl<I: Affine, P> Affine for iter::TakeWhile<I, P> {}
impl<I: Affine, P> Affine for iter::SkipWhile<I, P> {}
impl<I: Affine, P> Affine for iter::Filter<I, P> {}
impl<I: Affine, P> Affine for iter::FilterMap<I, P> {}
impl<I: Affine, P> Affine for iter::Map<I, P> {}
impl<I: Affine + Iterator> Affine for iter::Peekable<I> {}
impl<Iin, J, Iout, F> Affine for iter::FlatMap<Iin, J, F>
where
    Iin: Affine,
    J: IntoIterator<IntoIter = Iout>,
    Iout: Affine,
{
}
impl<Iin, J, Iout> Affine for iter::Flatten<Iin>
where
    Iin: Affine + Iterator<Item = J>,
    J: IntoIterator<IntoIter = Iout>,
    Iout: Affine,
{
}

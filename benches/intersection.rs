use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sorted_iter::SortedIterator;
use std::collections::BTreeSet;

fn intersection_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersection");
    
    let sizes = [100, 1000, 10000];
    
    // Benchmark with all combinations where a >= b
    for &size_a in &sizes {
        for &size_b in &sizes {
            if size_a >= size_b {
                let bench_id = format!("{}/{}", size_a, size_b);
                group.bench_with_input(
                    BenchmarkId::from_parameter(&bench_id),
                    &(size_a, size_b),
                    |bencher, &(size_a, size_b)| {
                        let a: BTreeSet<i64> = (0..size_a).collect();
                        let b: BTreeSet<i64> = (size_b / 2..size_b / 2 + size_b).collect();
                        
                        bencher.iter(|| {
                            let result: Vec<_> = black_box(a.iter().copied())
                                .intersection(black_box(b.iter().copied()))
                                .collect();
                            black_box(result);
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

criterion_group!(benches, intersection_benchmark);
criterion_main!(benches);

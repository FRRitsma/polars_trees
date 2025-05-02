use crate::constants::TARGET_COLUMN;
use crate::rework::constants::{
    COUNT_LEFT_COL, COUNT_RIGHT_COL, QUANTILES, SELECTION_COLUMN, TEMP_COLUMN_ORDINAL,
};
use crate::rework::gini_impurity;
use crate::rework::gini_impurity::extract_best_feature;
use crate::rework::sort_type::SortType;
use polars::prelude::{col, lit, UnionArgs};
use polars_core::datatypes::DataType;
use polars_lazy::dsl::concat;
use polars_lazy::frame::LazyFrame;

pub fn get_optimal_gini_impurity_for_ordinal_column(
    lf: &LazyFrame,
    feature_column: &str,
) -> LazyFrame {
    let lf = gini_impurity::pre_process_for_gini(lf, SortType::Ordinal, feature_column);

    // Gather lazy frames for every quantile:
    let mut lazy_frames: Vec<LazyFrame> = Vec::new();
    for quantile in QUANTILES.iter() {
        lazy_frames.push(group_by_for_single_quantile(&lf, feature_column, *quantile))
    }

    // Combine lazyframes into single, larger lazyframe:
    let mut grouped_lf = concat(
        &lazy_frames,
        UnionArgs {
            rechunk: true,
            to_supertypes: false,
            diagonal: false,
            from_partitioned_ds: false,
            parallel: true,
            maintain_order: false,
        },
    )
    .unwrap();

    grouped_lf = gini_impurity::add_totals_of_in_out_group(&grouped_lf);
    let gini_lf = gini_impurity::compute_gini_per_feature(&grouped_lf);
    let normalized_gini_lf = gini_impurity::normalize_gini_per_group(&grouped_lf, &gini_lf);
    // Keep only necessary columns and obtain best result:
    let final_lf = extract_best_feature(normalized_gini_lf);
    final_lf
}

fn group_by_for_single_quantile(lf: &LazyFrame, feature_column: &str, quantile: f64) -> LazyFrame {
    // Add quantile as string:
    let mut quantile_lf = lf.clone().with_column(
        col(feature_column)
            .quantile(lit(quantile), Default::default())
            .cast(DataType::String)
            .alias(SELECTION_COLUMN),
    );
    quantile_lf = quantile_lf
        .with_column(
            col(feature_column)
                .gt_eq(col(feature_column).quantile(lit(quantile), Default::default()))
                .alias(TEMP_COLUMN_ORDINAL),
        )
        .drop([feature_column]);
    let grouped_lf = group_by_for_ordinal_inner(&quantile_lf);
    grouped_lf
}

fn group_by_for_ordinal_inner(lf: &LazyFrame) -> LazyFrame {
    let mut grouped_lf = lf
        // Add count in:
        .clone()
        .group_by([col("*")])
        .agg([col(TARGET_COLUMN)
            .count()
            .alias(COUNT_LEFT_COL)
            .cast(DataType::Float64)]);

    grouped_lf = gini_impurity::add_zero_count(
        TEMP_COLUMN_ORDINAL,
        TARGET_COLUMN,
        COUNT_LEFT_COL,
        &grouped_lf,
    );

    // Add count out:
    grouped_lf = grouped_lf
        .with_columns([col(COUNT_LEFT_COL)
            .sum()
            .over([col(TARGET_COLUMN)])
            .alias("total_per_target")])
        .with_columns([(col("total_per_target") - col(COUNT_LEFT_COL)).alias(COUNT_RIGHT_COL)])
        .filter(col(TEMP_COLUMN_ORDINAL))
        .drop(["total_per_target", TEMP_COLUMN_ORDINAL])
        .sort([SELECTION_COLUMN, TARGET_COLUMN], Default::default());
    grouped_lf
}

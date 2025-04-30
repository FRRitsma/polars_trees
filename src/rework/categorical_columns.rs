use crate::constants::TARGET_COLUMN;
use crate::preprocessing::REDUNDANT_STRING_VALUE;
use crate::rework::gini_impurity_working_file::{
    add_zero_count, COUNT_IN, COUNT_OUT, SELECTION_COLUMN,
};
use polars::prelude::{col, lit};
use polars_core::datatypes::DataType;
use polars_lazy::frame::LazyFrame;

pub fn group_by_for_gini_impurity_categorical(lf: &LazyFrame) -> LazyFrame {
    // Instead of grouping by feature column, should group by selection column.
    let mut grouped_lf = lf
        // Group in and out:
        .clone()
        .group_by([col("*")])
        .agg([col(TARGET_COLUMN)
            .count()
            .alias(COUNT_IN)
            .cast(DataType::Float64)]);

    grouped_lf = add_zero_count(SELECTION_COLUMN, TARGET_COLUMN, COUNT_IN, &grouped_lf);

    grouped_lf = grouped_lf
        .with_columns([col(COUNT_IN)
            .sum()
            .over([col(TARGET_COLUMN)])
            .alias("total_per_target")])
        .with_columns([(col("total_per_target") - col(COUNT_IN)).alias(COUNT_OUT)])
        .filter(col(SELECTION_COLUMN).neq(lit(REDUNDANT_STRING_VALUE)))
        .drop(["total_per_target"])
        .sort([SELECTION_COLUMN, TARGET_COLUMN], Default::default());
    grouped_lf
}

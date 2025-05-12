use polars_lazy::frame::LazyFrame;
use polars::prelude::{col, IdxSize, JoinArgs, JoinType, lit, when};
use polars_core::prelude::SortMultipleOptions;
use std::error::Error;
use polars_core::datatypes::DataType;
use crate::old_preprocessing::REDUNDANT_STRING_VALUE;

pub fn rename_filler_string_full_lazyframe(
    lf: LazyFrame,
    minimum_sample_count: i32,
    top_n_most_frequent: IdxSize,
) -> Result<LazyFrame, Box<dyn Error>> {
    let schema = lf.logical_plan.compute_schema()?;
    let mut renamed_lf = lf.clone();
    for (name, dtype) in schema.iter() {
        if *dtype != DataType::String {
            continue;
        }
        renamed_lf = rename_filler_string_single_column(
            renamed_lf,
            name,
            minimum_sample_count,
            top_n_most_frequent,
        );
    }
    Ok(renamed_lf)
}

fn rename_filler_string_single_column(
    lf: LazyFrame,
    column_name: &str,
    minimum_sample_count: i32,
    top_n_most_frequent: IdxSize,
) -> LazyFrame {
    // Temporary columns:
    let top_n_column = "is_top_5";
    let count_column = "count";

    // Gather all strings that are prominent enough to keep:
    let top_strings = lf
        .clone()
        .group_by([col(column_name)])
        .agg([col(column_name).count().alias(count_column)])
        .sort(
            [count_column],
            SortMultipleOptions::default().with_order_descending(true),
        )
        .filter(col(count_column).gt_eq(minimum_sample_count))
        .select([col(column_name)])
        .limit(top_n_most_frequent)
        .with_column(lit(true).alias(top_n_column));

    // This join is "null" where the string value is not prominent:
    let join_lf = lf.clone().join(
        top_strings,
        [col(column_name)],
        [col(column_name)],
        JoinArgs::new(JoinType::Left),
    );

    // Rename "null" with filler string:
    let renamed_lf = join_lf
        .with_column(
            when(col(top_n_column).is_null())
                .then(lit(REDUNDANT_STRING_VALUE))
                .otherwise(col(column_name))
                .alias(column_name),
        )
        .drop([top_n_column]);
    renamed_lf
}

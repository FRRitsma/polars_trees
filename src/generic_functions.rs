// Target is to create functions for string columns, functions for numeric type columns and this file to streamline common functionalities between both

use crate::constants::{BINARIZED_COLUMN, CATEGORY_A, CATEGORY_B, COUNT_COLUMN};
use polars::prelude::{col, Expr, lit, when};
use polars_core::frame::DataFrame;
use polars_lazy::frame::LazyFrame;
use std::error::Error;
use crate::compute_information_single_category;

pub fn add_binary_column_to_dataframe(df: &LazyFrame, split_expression: &Expr) -> LazyFrame {
    df.clone().with_column(
        when(split_expression.clone())
            .then(lit(CATEGORY_A))
            .otherwise(lit(CATEGORY_B))
            .alias(BINARIZED_COLUMN),
    )
}

pub fn aggregate_binary_column(df: &LazyFrame, target_column: &str) -> LazyFrame {
    df.clone()
        .select([col(target_column), col(BINARIZED_COLUMN)])
        .group_by([col(target_column), col(BINARIZED_COLUMN)])
        .agg([col(target_column).count().alias(COUNT_COLUMN)])
        .sort([BINARIZED_COLUMN, target_column], Default::default())
}

pub fn extract_aggregated_count(series: &DataFrame, index: usize) -> f32 {
    series
        .select([COUNT_COLUMN])
        .unwrap()
        .get(index)
        .unwrap()
        .first()
        .unwrap()
        .try_extract::<f32>()
        .unwrap()
}

pub fn get_total_information_value(
    df: &LazyFrame,
    split_expression: &Expr,
    target_column: &str,
) -> Result<f32, Box<dyn Error>> {
    let binary_df = add_binary_column_to_dataframe(&df, &split_expression);
    let aggregate_df = aggregate_binary_column(&binary_df, target_column).collect()?;

    let category_a_label_0 = extract_aggregated_count(&aggregate_df, 0);
    let category_a_label_1 = extract_aggregated_count(&aggregate_df, 1);
    let category_b_label_0 = extract_aggregated_count(&aggregate_df, 2);
    let category_b_label_1 = extract_aggregated_count(&aggregate_df, 3);

    let information_value_a =
        compute_information_single_category(category_a_label_0, category_a_label_1);
    let information_value_b =
        compute_information_single_category(category_b_label_0, category_b_label_1);

    let total_information_value = information_value_a + information_value_b;
    Ok(total_information_value)
}

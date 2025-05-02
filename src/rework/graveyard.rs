use crate::constants::TARGET_COLUMN;
use polars::prelude::{col, lit};
use polars_core::datatypes::DataType;
use polars_lazy::frame::LazyFrame;

pub fn compute_parent_gini_impurity(lf: &LazyFrame) -> Result<f64, Box<dyn std::error::Error>> {
    let temp_column = "temp";

    let mut grouped = lf
        .clone()
        .select([col(TARGET_COLUMN)])
        .group_by([col(TARGET_COLUMN)])
        .agg([col(TARGET_COLUMN)
            .count()
            .cast(DataType::Float64)
            .alias(temp_column)]);

    let summed_count_df = grouped
        .clone()
        .select([col(temp_column)])
        .sum()
        .collect()
        .unwrap();

    let summed_count = summed_count_df
        .column(temp_column)
        .unwrap()
        .f64()
        .unwrap()
        .get(0)
        .unwrap();

    grouped = grouped.with_column((col(temp_column) / lit(summed_count)).pow(lit(2.0)));
    let summed_square_df = grouped
        .clone()
        .select([col(temp_column)])
        .sum()
        .collect()
        .unwrap();
    let gini_impurity = 1.0
        - summed_square_df
            .column(temp_column)
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
    Ok(gini_impurity)
}

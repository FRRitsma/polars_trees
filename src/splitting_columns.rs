use crate::categorical_columns::get_equality_expression_categories;
use crate::generic_functions::{CATEGORICAL_TYPES, CONTINUOUS_TYPES};
use crate::numeric_type_columns::get_median_split_expression;
use polars::prelude::Expr;
use polars_core::prelude::SchemaRef;
use polars_lazy::frame::LazyFrame;
use std::error::Error;

// TODO: Remove when refactored_get_split_expression is complete
pub fn get_split_expressions(
    df: &LazyFrame,
    schema: &SchemaRef,
    feature_column: &str,
) -> Result<Vec<Expr>, Box<dyn Error>> {
    let mut split_expression: Vec<Expr> = vec![];
    let dtype_column = schema.get(feature_column).unwrap();

    if CATEGORICAL_TYPES.contains(dtype_column) {
        split_expression.extend(get_equality_expression_categories(
            df,
            feature_column,
            5,
            32,
        )?);
    }

    if CONTINUOUS_TYPES.contains(dtype_column) {
        split_expression.extend(vec![get_median_split_expression(df, feature_column)?]);
    }

    Ok(split_expression)
}

#[cfg(test)]
mod tests {}

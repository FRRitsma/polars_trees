use crate::generic_functions;
use crate::generic_functions::ColumnType;
use crate::numeric_type_columns::get_median_split_expression;
use crate::string_type_columns::{
    get_equality_expression_for_common_categories, get_mode_split_expression,
};
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



    let split_expression = match generic_functions::get_type_of_column(schema, feature_column) {
        ColumnType::CategoryColumn => {
            get_equality_expression_for_common_categories(df, feature_column, 5, 32)?
        }
        ColumnType::ContinuousColumn => vec![get_median_split_expression(df, feature_column)?],
    };
    Ok(split_expression)
}

#[cfg(test)]
mod tests {}

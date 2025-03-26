use crate::generic_functions;
use crate::generic_functions::{
    add_binary_column_to_dataframe, aggregate_binary_column, get_type_of_column, ColumnType,
};
use crate::numeric_type_columns::get_median_split_expression;
use crate::string_type_columns::get_mode_split_expression;
use polars::prelude::Expr;
use polars_core::prelude::SchemaRef;
use polars_core::utils::Container;
use polars_lazy::frame::LazyFrame;
use std::error::Error;
use thiserror::Error;

pub enum SplitStrategy {
    Median,
    Mode,
    Average,
}

#[derive(Debug, Error)]
pub enum SplitStrategyError {
    #[error("Error with Median strategy")]
    MedianError(),

    #[error("Error with Mode strategy")]
    ModeError(),

    #[error("Error with Average strategy")]
    AverageError(),

    #[error("Invalid strategy: {0}")]
    InvalidStrategyError(String),
}

impl SplitStrategyError {
    // This function maps a SplitStrategy to the corresponding SplitStrategyError
    pub fn from_strategy(strategy: SplitStrategy) -> Self {
        match strategy {
            SplitStrategy::Median => SplitStrategyError::MedianError(),
            SplitStrategy::Mode => SplitStrategyError::ModeError(),
            SplitStrategy::Average => SplitStrategyError::AverageError(),
        }
    }
}

// TODO: Remove when refactored_get_split_expression is complete
pub fn get_split_expression(
    df: &LazyFrame,
    schema: &SchemaRef,
    feature_column: &str,
) -> Result<Expr, Box<dyn Error>> {
    let split_expression = match generic_functions::get_type_of_column(schema, feature_column) {
        ColumnType::StringColumn => get_mode_split_expression(df, feature_column)?,
        ColumnType::NumericColumn => get_median_split_expression(df, feature_column)?,
    };
    return Ok(split_expression);
}
//
// pub fn refactor_get_binarized_aggregated_dataframe(
//     df: &LazyFrame,
//     split_strategy: SplitStrategy,
//     feature_column: &str,
//     target_column: &str,
// ) -> Result<Expr, SplitStrategyError> {
//     // Exit invalid split strategy with an error:
//     let schema = df.logical_plan.compute_schema()?;
//     let column_type = get_type_of_column(&schema, feature_column);
//     if column_type == ColumnType::StringColumn && split_strategy != SplitStrategy::Mode {
//         return Err(SplitStrategyError::InvalidStrategyError(
//             "Only mode is a valid strategy for string type columns".to_string(),
//         ));
//     }
//
//     // Get the split expression:
//     let split_expression = match split_strategy {
//         SplitStrategy::Mode => get_mode_split_expression(df, feature_column)?,
//         SplitStrategy::Median => get_median_split_expression(df, feature_column)?,
//         SplitStrategy::Average => unimplemented!(),
//     };
//
//     // Retrieve the aggregated/binary dataframe and assert that is has length 4:
//     let binary_df = add_binary_column_to_dataframe(&df, &split_expression);
//     let aggregate_df = aggregate_binary_column(&binary_df, target_column).collect()?;
//
//     // Should have
//     if aggregate_df.len() != 4 {
//         return Err(SplitStrategyError::from_strategy(split_strategy));
//     }
//
//     Ok(split_expression)
// }

pub fn get_aggregated_binary_dataframe() {}
#[cfg(test)]
mod tests {}

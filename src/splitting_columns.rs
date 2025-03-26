use crate::generic_functions;
use crate::generic_functions::ColumnType;
use crate::numeric_type_columns::get_median_split_expression;
use crate::string_type_columns::get_mode_split_expression;
use polars::prelude::Expr;
use polars_core::prelude::SchemaRef;
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
    Ok(split_expression)
}

pub fn get_aggregated_binary_dataframe() {}
#[cfg(test)]
mod tests {}

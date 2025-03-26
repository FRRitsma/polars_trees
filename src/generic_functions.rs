// Target is to create functions for string columns, functions for numeric type columns and this file to streamline common functionalities between both

use crate::constants::{BINARIZED_COLUMN, CATEGORY_A, CATEGORY_B, COUNT_COLUMN, TARGET_COLUMN};
use crate::splitting_columns;
use polars::prelude::{col, lit, when, Expr, PolarsError};
use polars_core::frame::DataFrame;
use polars_core::prelude::DataType;
use polars_core::schema::SchemaRef;
use polars_core::utils::Container;
use polars_lazy::frame::LazyFrame;
use std::error::Error;
use thiserror::Error;
use crate::extract_values::extract_count;

#[derive(Debug, Error)]
pub enum InformationValueError {
    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError), // Automatically converts PolarsError

    #[error("Using the provided split expression not enough classes were provided")]
    SplitError(),
}

pub fn add_binary_column_to_dataframe(df: &LazyFrame, split_expression: &Expr) -> LazyFrame {
    df.clone().with_column(
        when(split_expression.clone())
            .then(lit(CATEGORY_A))
            .otherwise(lit(CATEGORY_B))
            .alias(BINARIZED_COLUMN),
    )
}

pub fn aggregate_binary_column(df: &LazyFrame) -> LazyFrame {
    df.clone()
        .select([col(TARGET_COLUMN), col(BINARIZED_COLUMN)])
        .group_by([col(TARGET_COLUMN), col(BINARIZED_COLUMN)])
        .agg([col(TARGET_COLUMN).count().alias(COUNT_COLUMN)])
        .sort([BINARIZED_COLUMN, TARGET_COLUMN], Default::default())
}

pub fn extract_aggregated_count(series: &DataFrame, index: usize) -> f32 {
    series
        .select([COUNT_COLUMN])
        .unwrap()
        .get(index)
        .expect(&format!("Index {} out of bounds for DataFrame", index))
        .first()
        .unwrap()
        .try_extract::<f32>()
        .unwrap()
}

// pub fn refactor_extract_aggregated_count(
//     series: &DataFrame,
//     category: &str,
//     label: &str,
// ) -> f32  {
//     // Create expressions for filtering by category and label
//     let filter_condition = col("BINARIZED_COLUMN").eq(lit(category))
//         .and(col("TARGET_COLUMN").eq(lit(label)));
//
//     // Convert DataFrame to LazyFrame and apply filter
//     let filtered_df = series
//         .clone()
//         .lazy()
//         .filter(filter_condition)
//         .select([COUNT_COLUMN])
//         .collect()
//         .unwrap();
//
//     // Get the first row and extract the value, return 0.0 if not found
//     filtered_df
//         .get(0)  // Get the first row
//         .and_then(|row| row[0].try_extract::<f32>().ok()) // Extract f32, return None if fails
//         .unwrap_or(0.0) // Return 0.0 if no row found or extraction fails
// }

pub fn get_total_information_value(
    df: &LazyFrame,
    split_expression: &Expr,
) -> Result<f32, InformationValueError> {
    let binary_df = add_binary_column_to_dataframe(&df, &split_expression);
    let aggregate_df = aggregate_binary_column(&binary_df).collect()?;

    let category_a_label_0 = extract_count(&aggregate_df,  false, CATEGORY_A,);
    let category_a_label_1 = extract_count(&aggregate_df,  true, CATEGORY_A,);
    let category_b_label_0 = extract_count(&aggregate_df,  false, CATEGORY_B,);
    let category_b_label_1 = extract_count(&aggregate_df,  true, CATEGORY_B,);

    let information_value_a =
        compute_information_single_category(category_a_label_0, category_a_label_1);
    let information_value_b =
        compute_information_single_category(category_b_label_0, category_b_label_1);

    let total_information_value = information_value_a + information_value_b;
    Ok(total_information_value)
}


fn compute_information_single_category(first_value: f32, second_value: f32) -> f32 {
    let first_value = first_value + 1.0;
    let second_value = second_value + 1.0;
    let total_value = first_value + second_value;
    let first_proportion = first_value / total_value;
    let second_proportion = second_value / total_value;
    (first_proportion - second_proportion) * (first_value / second_value).ln()
}

const NUMERIC_TYPES: [DataType; 10] = [
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::UInt8,
    DataType::UInt16,
    DataType::UInt32,
    DataType::UInt64,
    DataType::Float32,
    DataType::Float64,
];

#[derive(PartialEq, Debug)]
pub enum ColumnType {
    StringColumn,
    NumericColumn,
}

pub fn get_type_of_column(schema: &SchemaRef, column: &str) -> ColumnType {
    let dtype_column = schema.get(column).unwrap();
    if NUMERIC_TYPES.contains(dtype_column) {
        return ColumnType::NumericColumn;
    }
    ColumnType::StringColumn
}

struct LeafValue {
    split_expression: Expr,
    information_value: f32,
}

impl PartialEq for LeafValue {
    fn eq(&self, other: &Self) -> bool {
        self.information_value == other.information_value
    }
}

impl PartialOrd for LeafValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.information_value.partial_cmp(&other.information_value)
    }
}

pub fn get_leaf_value_for_column(
    df: &LazyFrame,
    schema: &SchemaRef,
    feature_column: &str,
) -> Result<LeafValue, Box<dyn Error>> {
    let split_expression = splitting_columns::get_split_expression(df, schema, feature_column)?;
    let information_value = get_total_information_value(&df, &split_expression)?;
    Ok(LeafValue {
        split_expression,
        information_value,
    })
}

pub fn get_optimal_leaf_value_of_dataframe(
    df: &LazyFrame,
    target_column: &str,
) -> Result<LeafValue, Box<dyn Error>> {
    let schema = df.logical_plan.compute_schema()?;
    let mut optimal_leaf_value: Option<LeafValue> = None;

    for feature_column in schema.iter_names() {
        if feature_column == target_column {
            continue;
        }

        let leaf_value = get_leaf_value_for_column(df, &schema, feature_column)?;

        match &optimal_leaf_value {
            Some(optimal) if leaf_value.information_value > optimal.information_value => {
                optimal_leaf_value = Some(leaf_value);
            }
            None => {
                optimal_leaf_value = Some(leaf_value);
            }
            _ => {}
        }
    }
    optimal_leaf_value.ok_or_else(|| "No optimal leaf value found".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::get_test_dataframe;
    #[test]
    fn test_get_type_of_column() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();
        let schema = df.logical_plan.compute_schema()?;
        assert_eq!(
            get_type_of_column(&schema, "Age"),
            ColumnType::NumericColumn
        );
        assert_eq!(get_type_of_column(&schema, "Sex"), ColumnType::StringColumn);
        Ok(())
    }

    #[test]
    fn test_get_leaf_value_of_age() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();
        let schema = df.logical_plan.compute_schema()?;
        let leaf_value = get_leaf_value_for_column(&df, &schema, "Age")?;
        assert!(leaf_value.information_value < 1.0);
        Ok(())
    }
    #[test]
    fn test_get_leaf_value_of_sex() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();
        let schema = df.logical_plan.compute_schema()?;
        let leaf_value = get_leaf_value_for_column(&df, &schema, "Sex")?;
        assert!(leaf_value.information_value > 1.0);
        Ok(())
    }

    #[test]
    fn test_get_optimal_leaf_value_of_dataframe() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();
        let schema = df.logical_plan.compute_schema()?;

        let optimal_leaf_value = get_optimal_leaf_value_of_dataframe(&df, TARGET_COLUMN)?;
        println!("{:?}", optimal_leaf_value.information_value);
        println!("{:?}", optimal_leaf_value.split_expression);

        Ok(())
    }
}

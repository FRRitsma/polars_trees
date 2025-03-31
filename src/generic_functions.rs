// Target is to create functions for string columns, functions for numeric type columns and this file to streamline common functionalities between both

use crate::constants::{BINARIZED_COLUMN, CATEGORY_A, CATEGORY_B, COUNT_COLUMN, TARGET_COLUMN};
use crate::extract_values::extract_count;
use crate::splitting_columns;
use polars::prelude::{col, lit, when, Expr};
use polars_core::prelude::DataType;
use polars_core::schema::SchemaRef;
use polars_lazy::frame::LazyFrame;
use std::error::Error;

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

pub fn get_total_information_value(df: &LazyFrame, split_expression: &Expr) -> (f32, f32) {
    // Implement minimal bin size?
    let binary_df = add_binary_column_to_dataframe(df, split_expression);
    let aggregate_df = aggregate_binary_column(&binary_df).collect().unwrap();

    let category_a_label_0 = extract_count(&aggregate_df, false, CATEGORY_A);
    let category_a_label_1 = extract_count(&aggregate_df, true, CATEGORY_A);
    let category_b_label_0 = extract_count(&aggregate_df, false, CATEGORY_B);
    let category_b_label_1 = extract_count(&aggregate_df, true, CATEGORY_B);

    let information_value_a =
        compute_information_single_category(category_a_label_0, category_a_label_1);
    let information_value_b =
        compute_information_single_category(category_b_label_0, category_b_label_1);

    let minimum_bin_size =
        (category_a_label_0 + category_a_label_1).min(category_b_label_0 + category_b_label_1);

    let total_information_value = information_value_a + information_value_b;
    (total_information_value, minimum_bin_size)
}

fn compute_information_single_category(first_value: f32, second_value: f32) -> f32 {
    let epsilon: f32 = 0.01;
    let first_value = first_value;
    let second_value = second_value;
    let total_value = first_value + second_value;
    let first_proportion = first_value / total_value;
    let second_proportion = second_value / total_value;
    (first_proportion - second_proportion)
        * ((first_value + epsilon) / (second_value + epsilon)).log2()
}

pub const CATEGORICAL_TYPES: [DataType; 10] = [
    DataType::Int8,
    DataType::Boolean,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::UInt8,
    DataType::UInt16,
    DataType::UInt32,
    DataType::UInt64,
    DataType::String,
];

pub const CONTINUOUS_TYPES: [DataType; 10] = [
    DataType::Float32,
    DataType::Float64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::UInt8,
    DataType::UInt16,
    DataType::UInt32,
    DataType::UInt64,
];


pub struct LeafValue {
    // TODO: Add sample count
    pub(crate) split_expression: Expr,
    pub information_value: f32,
    pub minimum_bin_size: f32,
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
    minimum_bin_size: f32,
) -> Result<LeafValue, Box<dyn Error>> {
    //TODO: Implement multiple split expression
    let mut optimal_leaf_value: Option<LeafValue> = None;
    let split_expressions = splitting_columns::get_split_expressions(df, schema, feature_column)?;

    for split_expression in split_expressions {
        // Extract loop value:
        let (information_value, extracted_bin_size) =
            get_total_information_value(df, &split_expression);
        let leaf_value = LeafValue {
            split_expression,
            information_value,
            minimum_bin_size: extracted_bin_size,
        };

        // Compare to optimal value:
        match &optimal_leaf_value {
            Some(optimal)
                if leaf_value.information_value > optimal.information_value
                    && leaf_value.minimum_bin_size > minimum_bin_size =>
            {
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

pub fn get_optimal_leaf_value_of_dataframe(
    df: &LazyFrame,
    minimum_bin_size: f32,
) -> Result<LeafValue, Box<dyn Error>> {
    let schema = df.logical_plan.compute_schema()?;
    let mut optimal_leaf_value: Option<LeafValue> = None;

    for feature_column in schema.iter_names() {
        if feature_column == TARGET_COLUMN {
            continue;
        }

        let leaf_value = get_leaf_value_for_column(df, &schema, feature_column, minimum_bin_size)
            .unwrap_or(LeafValue {
                split_expression: col("Empty"),
                information_value: -99999.0,
                minimum_bin_size: 99999.0,
            });

        match &optimal_leaf_value {
            Some(optimal)
                if leaf_value.information_value > optimal.information_value
                    && leaf_value.minimum_bin_size > minimum_bin_size =>
            {
                optimal_leaf_value = Some(leaf_value);
            }
            None => {
                optimal_leaf_value = Some(leaf_value);
            }
            _ => {}
        }
    }
    optimal_leaf_value.ok_or_else(|| "No optimal leaf value found in the entire dataframe".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::get_preprocessed_test_dataframe;

    #[test]
    fn test_get_leaf_value_of_age() -> Result<(), Box<dyn Error>> {
        let df = get_preprocessed_test_dataframe();
        let schema = df.logical_plan.compute_schema()?;
        let leaf_value = get_leaf_value_for_column(&df, &schema, "Age", 32f32)?;
        assert!(leaf_value.information_value < 1.0);
        println!("{:?}", leaf_value.information_value);

        Ok(())
    }

    #[test]
    fn test_get_leaf_value_of_sex() -> Result<(), Box<dyn Error>> {
        let df = get_preprocessed_test_dataframe();
        let schema = df.logical_plan.compute_schema()?;
        let leaf_value = get_leaf_value_for_column(&df, &schema, "Sex", 32f32)?;
        assert!(leaf_value.information_value > 1.0);
        println!("{:?}", leaf_value.information_value);
        Ok(())
    }

    #[test]
    fn test_get_optimal_leaf_value_of_dataframe() -> Result<(), Box<dyn Error>> {
        let df = get_preprocessed_test_dataframe();
        let schema = df.logical_plan.compute_schema()?;
        let column_names: Vec<&str> = schema.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(column_names.len(), 12);

        // println!("{:?}", column_names);

        let optimal_leaf_value = get_optimal_leaf_value_of_dataframe(&df, 32f32)?;
        let expected_split_expression: Expr = col("Sex").eq(lit("male"));
        assert_eq!(
            expected_split_expression,
            optimal_leaf_value.split_expression
        );
        assert!(optimal_leaf_value.information_value > 1.0);
        println!("{:?}", optimal_leaf_value.information_value);
        println!("{:?}", optimal_leaf_value.split_expression);

        Ok(())
    }
}

use polars::io::SerReader;
use polars::prelude::CsvReadOptions;
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use std::collections::HashSet;
use std::error::Error;

use polars::prelude::*;

mod categorical_columns;
mod categorical_columns_remake;
mod constants;
mod debug;
mod display_tree;
mod empty_tree;
mod extract_values;
mod generic_functions;
mod gini_impurity_working_file;
mod numeric_type_columns;
mod preprocessing;
mod splitting_columns;
mod test_utils;
mod tree;
mod settings;
mod splitting_dataframes;

fn create_split_in_dataframe(
    df: &LazyFrame,
    category: &str,
    feature_column: &str,
) -> (LazyFrame, Expr) {
    // Creates a split in the dataframe and returns the associated split condition
    let not_category = "not_".to_owned() + category;
    let binary_feature_column = "binary_".to_owned() + feature_column;
    let split_condition: Expr = col(feature_column).eq(lit(category));
    (
        df.clone().with_column(
            when(split_condition.clone())
                .then(lit(category))
                .otherwise(lit(not_category))
                .alias(binary_feature_column),
        ),
        split_condition,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    // Specify the path to your CSV file
    let file_path = "Titanic-Dataset.csv";

    let numeric_types: HashSet<DataType> = HashSet::from([
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
    ]);

    let df = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()?
        .lazy();

    // Read the CSV file into a DataFrame
    // let feature_column = "Sex";
    let feature_column = "Sex";

    let schema = df.logical_plan.compute_schema()?;

    let dtype_column = schema.get(feature_column).unwrap();
    if numeric_types.contains(dtype_column) {
        println!("Is numeric!");
    }

    // Define the quantiles you want
    let quantiles = [0.1, 0.25, 0.5, 0.75, 0.9];

    // Create expressions for each quantile
    let quantile_exprs: Vec<Expr> = quantiles
        .iter()
        .map(|&q| {
            col("Age")
                .quantile(lit(q), QuantileInterpolOptions::Nearest)
                .alias(format!("q{}", q))
        })
        .collect();

    let result = df.select(quantile_exprs).collect()?;

    // Extract quantile values into a Vec<f64>
    let quantile_values: Vec<f64> = result
        .get_columns() // Get all Series (columns)
        .iter()
        .map(|series| series.get(0).unwrap().try_extract::<f64>().unwrap())
        .collect();

    println!("Quantile Values: {:?}", quantile_values);

    // let total_information_value = string_type_columns::get_information_value_of_string_type_column(&df, &feature_column)?;

    println!("{:?}", result);

    Ok(())
}

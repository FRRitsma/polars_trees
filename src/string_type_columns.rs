use crate::constants::{BINARIZED_COLUMN, COUNT_COLUMN};
use polars::prelude::col;
use polars::prelude::Expr;
use polars_core::prelude::SortMultipleOptions;
use polars_core::prelude::*;
use polars_lazy::frame::LazyFrame;
use polars_lazy::prelude::*;
use std::error::Error;

fn get_split_expression_for_string_column(
    df: &LazyFrame,
    feature_column: &str,
) -> Result<Expr, Box<dyn Error>> {
    let most_common_string = get_most_common_string_in_column(df, feature_column)?;
    Ok(col(feature_column).eq(lit(most_common_string)))
}

fn get_most_common_string_in_column<'a>(
    df: &'a LazyFrame,
    feature_column: &'a str,
) -> Result<String, Box<dyn Error>> {
    // try_extract doesn't work for string, convert PolarsResult<AnyValue> doesn't support utf8,
    // so the extraction is quite ugly
    let collected_df = df
        .clone()
        .group_by([col(feature_column)])
        .agg([col(feature_column).count().alias("count")])
        .sort(
            ["count"],
            SortMultipleOptions::default().with_order_descending(true), // Sort descending
        )
        .select([col(feature_column)])
        .limit(1)
        .collect()?;

    let mode_value = collected_df
        .column(feature_column)?
        .get(0)?
        .to_string()
        .trim_matches('"')
        .to_string();
    Ok(mode_value)
}

pub fn get_information_value_of_string_type_column(
    df: &LazyFrame,
    feature_column: &&str,
    target_column: &str,
) -> Result<(f32, String), Box<dyn Error>> {
    let category = get_most_common_string_in_column(df, feature_column)?;
    let not_category = "not_".to_owned() + &*category;

    let (df, _) = crate::create_split_in_dataframe(df, &category, feature_column);
    let binarized_feature_column = &("binary_".to_owned() + feature_column);

    let df = df
        .group_by([col(target_column), col(binarized_feature_column)])
        .agg([col(target_column).count().alias(COUNT_COLUMN)])
        .sort(
            [binarized_feature_column, target_column],
            Default::default(),
        );

    let first_information_value =
        crate::extract_combined_information_value(&category, binarized_feature_column, &df)?;
    let second_information_value =
        crate::extract_combined_information_value(&not_category, binarized_feature_column, &df)?;
    let total_information_value = first_information_value + second_information_value;
    Ok((total_information_value, category))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic_functions;
    use crate::generic_functions::add_binary_column_to_dataframe;
    use polars::io::SerReader;
    use polars::prelude::CsvReadOptions;
    use polars_lazy::prelude::IntoLazy;

    // This allows the test module to access the functions in the outer scope
    const FILE_PATH: &str = "Titanic-Dataset.csv";
    const FEATURE_COLUMN: &str = "Sex";
    const TARGET_COLUMN: &str = "Survived";

    fn get_test_dataframe() -> LazyFrame {
        CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(FILE_PATH.into()))
            .unwrap()
            .finish()
            .unwrap()
            .lazy()
    }

    #[test]
    fn test_get_split_expression_for_category_sex() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();
        let split_expression = get_split_expression_for_string_column(&df, FEATURE_COLUMN)?;
        let expected_split_expression = col(FEATURE_COLUMN).eq(lit("male"));
        assert_eq!(split_expression, expected_split_expression);
        Ok(())
    }

    #[test]
    fn test_split_feature_column_into_a_and_b_category() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();
        let split_expression = get_split_expression_for_string_column(&df, FEATURE_COLUMN)?;
        let binary_df = add_binary_column_to_dataframe(&df, &split_expression)
            .select([col(FEATURE_COLUMN), col(BINARIZED_COLUMN)])
            .collect()?;
        println!("{:?}", binary_df);
        Ok(())
    }

    #[test]
    fn test_get_total_information() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();

        let split_expression = get_split_expression_for_string_column(&df, FEATURE_COLUMN)?;
        let total_information_value =
            generic_functions::get_total_information_value(&df, &split_expression, TARGET_COLUMN)?;

        println!("{:?}", total_information_value);
        assert!(total_information_value > 1.0);
        Ok(())
    }

    #[test]
    fn test_get_information_value_for_column_sex() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();

        let information_value_and_category =
            get_information_value_of_string_type_column(&df, &FEATURE_COLUMN, TARGET_COLUMN)?;
        assert!(information_value_and_category.0 > 1.0);
        Ok(())
    }
}

use crate::constants::BINARIZED_COLUMN;
use polars::prelude::col;
use polars::prelude::Expr;
use polars_core::prelude::SortMultipleOptions;
use polars_core::prelude::*;
use polars_lazy::frame::LazyFrame;
use polars_lazy::prelude::*;
use std::error::Error;

pub fn get_mode_split_expression(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic_functions::add_binary_column_to_dataframe;
    use crate::test_utils::get_test_dataframe;
    use crate::{generic_functions, test_utils};

    const FEATURE_COLUMN: &str = "Sex";
    const TARGET_COLUMN: &str = "Survived";

    #[test]
    fn test_get_split_expression_for_category_sex() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();
        let split_expression = get_mode_split_expression(&df, FEATURE_COLUMN)?;
        let expected_split_expression = col(FEATURE_COLUMN).eq(lit("male"));
        assert_eq!(split_expression, expected_split_expression);
        Ok(())
    }

    #[test]
    fn test_split_feature_column_into_a_and_b_category() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();
        let split_expression = get_mode_split_expression(&df, FEATURE_COLUMN)?;
        let binary_df = add_binary_column_to_dataframe(&df, &split_expression)
            .select([col(FEATURE_COLUMN), col(BINARIZED_COLUMN)])
            .collect()?;
        println!("{:?}", binary_df);
        Ok(())
    }

    #[test]
    fn test_get_total_information_for_sex_column() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();

        let split_expression = get_mode_split_expression(&df, FEATURE_COLUMN)?;
        let total_information_value =
            generic_functions::get_total_information_value(&df, &split_expression, TARGET_COLUMN)?;

        println!("{:?}", total_information_value);
        assert!(total_information_value > 1.0);
        Ok(())
    }
}

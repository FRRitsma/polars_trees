use polars_lazy::frame::LazyFrame;
use std::error::Error;
use polars::prelude::col;
use polars_core::prelude::SortMultipleOptions;
use polars::prelude::Expr;
use polars_core::prelude::*;
use polars_lazy::prelude::*;


const CATEGORY_A: &str = "CATEGORY_A";
const CATEGORY_B: &str = "CATEGORY_B";
const BINARIZED_COLUMN: &str = "BINARIZED_COLUMN";

fn get_split_expression_for_string_column(df: &LazyFrame, feature_column: &str) -> Result<Expr, Box<dyn Error>>{
    let most_common_string = get_most_common_string_in_column(df, feature_column)?;
    return Ok(col(feature_column).eq(lit(most_common_string)));
}


fn get_most_common_string_in_column<'a>(df: &'a LazyFrame, feature_column: &'a str) -> Result<String, Box<dyn Error>> {
    // try_extract doesn't work for string, convert PolarsResult<AnyValue> doesn't support utf8,
    // so the extraction is quite ugly
    let collected_df = df
        .clone()
        .group_by([col(feature_column)])
        .agg([col(feature_column).count().alias("count")])
        .sort(
            ["count"],
            SortMultipleOptions::default().with_order_descending(true)  // Sort descending
        )
        .select([col(feature_column)])
        .limit(1)
        .collect()?;

    let mode_value = collected_df.column(feature_column)?.get(0)?.to_string().trim_matches('"').to_string();
    Ok(mode_value)
}


pub fn get_information_value_of_string_type_column(df: &LazyFrame, feature_column: &&str, target_column: &str) -> Result<(f32, String), Box<dyn Error>> {

    let category = get_most_common_string_in_column(&df, feature_column)?;
    let not_category = "not_".to_owned() + &*category;

    let (df, _) = crate::create_split_in_dataframe(&df, &category, feature_column);
    let binarized_feature_column = &("binary_".to_owned() + feature_column);


    let df = df
        .group_by([col(target_column), col(binarized_feature_column)])
        .agg([col(target_column).count().alias("count")])
        .sort([binarized_feature_column, target_column], Default::default());

    let first_information_value = crate::extract_combined_information_value(&category, binarized_feature_column, &df)?;
    let second_information_value = crate::extract_combined_information_value(&not_category, binarized_feature_column, &df)?;
    let total_information_value = first_information_value + second_information_value;
    Ok((total_information_value, category))
}


pub fn create_binary_column(df: &LazyFrame, split_expression: &Expr) -> LazyFrame{
    let df = df.clone().with_column(
        when(split_expression.clone()).then(lit(CATEGORY_A)).otherwise(lit(CATEGORY_B)).alias(BINARIZED_COLUMN)
    );
    return df;
}


pub fn count_binary_column(df: &LazyFrame, target_column: &str) -> LazyFrame{
    let df = df.clone()
        .select([col(target_column), col(BINARIZED_COLUMN)])
        .group_by([col(target_column), col(BINARIZED_COLUMN)])
        .agg([col(target_column).count().alias("count")])
        .sort([BINARIZED_COLUMN, target_column], Default::default());
    return df
}


#[cfg(test)]
mod tests {
    use polars::io::SerReader;
    use polars::prelude::CsvReadOptions;
    use polars_lazy::prelude::IntoLazy;
    use super::*; // This allows the test module to access the functions in the outer scope
    const FILE_PATH: &str = "Titanic-Dataset.csv";

    fn get_test_dataframe() -> LazyFrame{
         CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(FILE_PATH.into())).unwrap()
        .finish()
             .unwrap()
        .lazy()
    }


    #[test]
    fn test_get_split_expression_for_category_sex() -> Result<(), Box<dyn Error>>{
        let df = get_test_dataframe();
        let feature_column = "Sex";
        let split_expression = get_split_expression_for_string_column(&df, feature_column)?;
        let expected_split_expression = col(feature_column).eq(lit("male"));
        assert_eq!(split_expression, expected_split_expression);
        Ok(())
    }

    #[test]
    fn test_split_feature_column_into_a_and_b_category() -> Result<(), Box<dyn Error>>{
        let df = get_test_dataframe();
        let feature_column = "Sex";
        let split_expression = get_split_expression_for_string_column(&df, feature_column)?;
        let binary_df = create_binary_column(&df, &split_expression).select([col("Sex"), col(BINARIZED_COLUMN)]).collect()?;
        println!("{:?}", binary_df);
        Ok(())
    }


    #[test]
    fn test_get_information_value_for_column_sex() -> Result<(), Box<dyn Error>>{
        let df = get_test_dataframe();

        let feature_column = "Sex";
        let target_column = "Survived";
        let information_value_and_category = get_information_value_of_string_type_column(&df, &feature_column, target_column)?;
        assert!(information_value_and_category.0 > 1.0);
        Ok(())
    }


}


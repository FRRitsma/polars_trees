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

fn get_n_unique_values(lf: LazyFrame, column: &str) -> Result<u32, Box<dyn Error>> {
    let result_column = "unique_values";
    let distinct_values = lf
        .select([col(column).n_unique().alias(result_column)])
        .collect()?
        .column(result_column)?
        .u32()?
        .get(0)
        .unwrap();
    Ok(distinct_values)
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

fn get_most_common_values_as_df(
    lf: &LazyFrame,
    column_name: &str,
    top_n: IdxSize,
    minimum_bin_size: i32,
) -> Result<DataFrame, Box<dyn Error>> {
    // Return format example:
    // column_name = Pclass
    // ┌────────┬───────┐
    // │ Pclass ┆ count │
    // │ ---    ┆ ---   │
    // │ i64    ┆ u32   │
    // ╞════════╪═══════╡
    // │ 3      ┆ 491   │
    // │ 1      ┆ 216   │
    // └────────┴───────┘

    let count_column = "count";
    let top_df = lf
        .clone()
        .group_by([column_name])
        .agg([col(column_name).count().alias(count_column)])
        .filter(col(count_column).gt(minimum_bin_size))
        .sort(
            [count_column],
            SortMultipleOptions::default().with_order_descending(true),
        )
        .limit(top_n)
        .collect()?;
    Ok(top_df)
}

fn extract_equality_expressions(
    column_name: &str,
    top_column: &Column,
) -> Result<Vec<Expr>, Box<dyn Error>> {
    let expressions = match top_column.dtype() {
        DataType::Int64 => {
            let top_vector: Vec<i64> = top_column.i64()?.into_no_null_iter().collect();
            top_vector
                .into_iter()
                .map(|i| col(column_name).eq(lit(i)))
                .collect()
        }
        DataType::String => {
            let top_vector: Vec<&str> = top_column.str()?.into_no_null_iter().collect();
            top_vector
                .into_iter()
                .map(|i| col(column_name).eq(lit(i)))
                .collect()
        }
        _ => return Err("Unsupported data type!".into()),
    };
    Ok(expressions)
}

pub fn get_equality_expression_for_common_categories(
    lf: &LazyFrame,
    column_name: &str,
    top_n: IdxSize,
    minimum_bin_size: i32,
) -> Result<Vec<Expr>, Box<dyn Error>> {
    let top_df = get_most_common_values_as_df(&lf, column_name, top_n, minimum_bin_size)?;
    let top_column = top_df.column(column_name)?;
    let expressions = extract_equality_expressions(column_name, top_column)?;
    Ok(expressions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::BINARIZED_COLUMN;
    use crate::generic_functions;
    use crate::generic_functions::add_binary_column_to_dataframe;
    use crate::test_utils::get_preprocessed_test_dataframe;

    const FEATURE_COLUMN: &str = "Sex";

    #[test]
    fn test_get_split_expression_for_category_sex() -> Result<(), Box<dyn Error>> {
        let lf = get_preprocessed_test_dataframe();
        let split_expression = get_mode_split_expression(&lf, FEATURE_COLUMN)?;
        let expected_split_expression = col(FEATURE_COLUMN).eq(lit("male"));
        assert_eq!(split_expression, expected_split_expression);
        Ok(())
    }

    #[test]
    fn test_split_feature_column_into_a_and_b_category() -> Result<(), Box<dyn Error>> {
        let df = get_preprocessed_test_dataframe();
        let split_expression = get_mode_split_expression(&df, FEATURE_COLUMN)?;
        let binary_df = add_binary_column_to_dataframe(&df, &split_expression)
            .select([col(FEATURE_COLUMN), col(BINARIZED_COLUMN)])
            .collect()?;
        println!("{:?}", binary_df);
        Ok(())
    }

    #[test]
    fn test_get_total_information_for_sex_column() -> Result<(), Box<dyn Error>> {
        let df = get_preprocessed_test_dataframe();

        let split_expression = get_mode_split_expression(&df, FEATURE_COLUMN)?;
        let (total_information_value, _) =
            generic_functions::get_total_information_value(&df, &split_expression);

        println!("{:?}", total_information_value);
        assert!(total_information_value > 1.0);
        Ok(())
    }

    #[test]
    fn test_unique_amount_of_names() -> Result<(), Box<dyn Error>> {
        let lf = get_preprocessed_test_dataframe();
        let column = "Sex";
        let distinct_values = get_n_unique_values(lf, column)?;
        assert_eq!(distinct_values, 2);
        Ok(())
    }

    #[test]
    fn test_get_top_n_most_common_values_in_passenger_class() -> Result<(), Box<dyn Error>> {
        let lf = get_preprocessed_test_dataframe();
        let column_name = "Pclass";
        let top_n = 3;
        let minimum_bin_size = 200;

        let top_df = get_most_common_values_as_df(&lf, column_name, top_n, minimum_bin_size)?;
        let top_column = top_df.column(column_name)?;
        let top_vector: Vec<i64> = top_column.i64()?.into_no_null_iter().collect();

        assert_eq!(top_vector.len(), 2);

        Ok(())
    }

    #[test]
    fn test_equality_expressions_for_passenger_class() {
        let lf = get_preprocessed_test_dataframe();
        let column_name = "Pclass";
        let top_n = 3;
        let minimum_bin_size = 200;

        let expressions = get_equality_expression_for_common_categories(
            &lf,
            column_name,
            top_n,
            minimum_bin_size,
        )
        .unwrap();

        assert_eq!(expressions.len(), 2);
    }

    #[test]
    fn test_equality_expressions_for_sex() {
        let lf = get_preprocessed_test_dataframe();
        let column_name = "Sex";
        let top_n = 3;
        let minimum_bin_size = 200;
        let expressions = get_equality_expression_for_common_categories(
            &lf,
            column_name,
            top_n,
            minimum_bin_size,
        )
        .unwrap();
        assert_eq!(expressions.len(), 2);
    }

    #[test]
    fn test_equality_expressions_for_name() {
        let lf = get_preprocessed_test_dataframe();
        let column_name = "Name";
        let top_n = 3;
        let minimum_bin_size = 200;
        let expressions = get_equality_expression_for_common_categories(
            &lf,
            column_name,
            top_n,
            minimum_bin_size,
        )
        .unwrap();
        assert_eq!(expressions.len(), 0);
        let schema = lf.logical_plan.compute_schema().unwrap();
        let column_names: Vec<&str> = schema.iter().map(|(name, _)| name.as_str()).collect();
        assert!(schema.len() > 10);
    }
}

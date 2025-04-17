use polars::prelude::col;
use polars::prelude::Expr;
use polars_core::prelude::SortMultipleOptions;
use polars_core::prelude::*;
use polars_lazy::frame::LazyFrame;
use polars_lazy::prelude::*;
use std::error::Error;

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

pub fn old_get_most_common_values_as_df(
    lf: &LazyFrame,
    column_name: &str,
    top_n: IdxSize,
    minimum_bin_size: i32,
) -> Result<DataFrame, Box<dyn Error>> {
    /*
    Return format example:
    column_name = Pclass
    ┌────────┬───────┐
    │ Pclass ┆ count │
    │ ---    ┆ ---   │
    │ i64    ┆ u32   │
    ╞════════╪═══════╡
    │ 3      ┆ 491   │
    │ 1      ┆ 216   │
    └────────┴───────┘
     */

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

fn extract_equality_expressions_from_column(
    column_name: &str,
    top_column: &Column,
) -> Result<Vec<Expr>, Box<dyn Error>> {
    match top_column.dtype() {
        dt if dt.is_integer() => {
            // Cast to i64 to handle all integer types uniformly
            let casted = top_column.cast(&DataType::Int64)?;
            let top_vector: Vec<i64> = casted.i64()?.into_no_null_iter().collect();
            Ok(top_vector
                .into_iter()
                .map(|i| col(column_name).eq(lit(i)))
                .collect())
        }
        DataType::String => {
            let top_vector: Vec<&str> = top_column.str()?.into_no_null_iter().collect();
            Ok(top_vector
                .into_iter()
                .map(|i| col(column_name).eq(lit(i)))
                .collect())
        }
        _ => Err("Unsupported data type!".into()),
    }
}

pub fn get_equality_expression_categories(
    lf: &LazyFrame,
    column_name: &str,
    top_n: IdxSize,
    minimum_bin_size: i32,
) -> Result<Vec<Expr>, Box<dyn Error>> {
    let top_df = old_get_most_common_values_as_df(lf, column_name, top_n, minimum_bin_size)?;
    let top_column = top_df.column(column_name)?;
    let expressions = extract_equality_expressions_from_column(column_name, top_column)?;
    Ok(expressions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::get_preprocessed_test_dataframe;

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

        let top_df = old_get_most_common_values_as_df(&lf, column_name, top_n, minimum_bin_size)?;
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

        let expressions =
            get_equality_expression_categories(&lf, column_name, top_n, minimum_bin_size).unwrap();

        assert_eq!(expressions.len(), 2);
    }

    #[test]
    fn test_equality_expressions_for_sex() {
        let lf = get_preprocessed_test_dataframe();
        let column_name = "Sex";
        let top_n = 3;
        let minimum_bin_size = 200;
        let expressions =
            get_equality_expression_categories(&lf, column_name, top_n, minimum_bin_size).unwrap();
        assert_eq!(expressions.len(), 2);
    }

    #[test]
    fn test_equality_expressions_for_name() {
        let lf = get_preprocessed_test_dataframe();
        let column_name = "Name";
        let top_n = 3;
        let minimum_bin_size = 200;
        let expressions =
            get_equality_expression_categories(&lf, column_name, top_n, minimum_bin_size).unwrap();
        assert_eq!(expressions.len(), 0);
        let schema = lf.logical_plan.compute_schema().unwrap();
        let column_names: Vec<&str> = schema.iter().map(|(name, _)| name.as_str()).collect();
        assert!(schema.len() > 10);
    }
}

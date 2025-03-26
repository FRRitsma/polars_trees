use polars::prelude::Expr;
use polars::prelude::{col, lit};
use polars_lazy::frame::LazyFrame;
use std::error::Error;

fn get_median_of_column(df: &LazyFrame, feature_column: &str) -> Result<f32, Box<dyn Error>> {
    let median_df = df
        .clone()
        .select([col(feature_column).median().alias("median_value")])
        .collect()?;

    let median_value = median_df
        .column("median_value")?
        .get(0)?
        .try_extract::<f32>()?;

    Ok(median_value)
}

pub fn get_median_split_expression(
    df: &LazyFrame,
    feature_column: &str,
) -> Result<Expr, Box<dyn Error>> {
    let median_value = get_median_of_column(df, feature_column)?;
    Ok(col(feature_column).lt(lit(median_value)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::BINARIZED_COLUMN;
    use crate::generic_functions::{add_binary_column_to_dataframe, get_total_information_value};
    use crate::test_utils::get_test_dataframe;
    use polars::prelude::col;

    #[test]
    fn test_get_median_of_age() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();
        let median_of_age = get_median_of_column(&df, "Age")?;
        assert!(median_of_age > 10.0);
        Ok(())
    }

    #[test]
    fn test_split_age() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();
        let split_expression = get_median_split_expression(&df, "Age")?;
        let binary_df = add_binary_column_to_dataframe(&df, &split_expression)
            .select([col("Age"), col(BINARIZED_COLUMN)])
            .collect()?;
        println!("{:?}", binary_df);
        Ok(())
    }

    #[test]
    fn test_get_total_information_for_age_column() -> Result<(), Box<dyn Error>> {
        let df = get_test_dataframe();

        let split_expression = get_median_split_expression(&df, "Age")?;
        let total_information_value =
            get_total_information_value(&df, &split_expression, "Survived")?;

        println!("{:?}", total_information_value);
        assert!(total_information_value < 1.0);
        Ok(())
    }
}

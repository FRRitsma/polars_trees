use crate::constants::{BINARIZED_COLUMN, COUNT_COLUMN, TARGET_COLUMN};
use polars::prelude::{col, lit, PolarsError, PolarsResult};
use polars_core::frame::DataFrame;
use polars_core::prelude::SortOptions;
use polars_lazy::frame::LazyFrame;
use polars_lazy::prelude::IntoLazy;

pub fn extract_count(df: &DataFrame, label: bool, category: &str) -> f32 {
    // TODO: Conversion to lazy dataframe shouldn't be necessary
    let default = 0.0;
    let output = df
        .clone()
        .lazy()
        .filter(
            col(BINARIZED_COLUMN)
                .eq(lit(category))
                .and(col(TARGET_COLUMN).eq(lit(label))),
        )
        .select([col(COUNT_COLUMN)])
        .collect()
        .unwrap()
        .column(COUNT_COLUMN)
        .and_then(|s| s.get(0))
        .map(|val| val.extract::<f32>().unwrap_or(default))
        .unwrap_or(default);
    output
}

pub fn get_most_common_label(df: &LazyFrame) -> Result<bool, PolarsError> {
    let collected = df.clone().select([col(TARGET_COLUMN).mode()]).collect()?;

    let mode = collected.column(TARGET_COLUMN)?.bool()?.get(0).unwrap(); // or proper error handling
    Ok(mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{CATEGORY_A, CATEGORY_B};
    use crate::extract_values::extract_count;
    use polars_core::df;
    use polars_core::frame::DataFrame;

    fn test_complete_df() -> DataFrame {
        df!(
            BINARIZED_COLUMN => &[CATEGORY_A, CATEGORY_B, CATEGORY_A, CATEGORY_B],
            TARGET_COLUMN => &[true, true, false, false],
            COUNT_COLUMN => &[10, 20, 30, 40]
        )
        .unwrap()
    }

    fn test_empty_df() -> DataFrame {
        df!(
            BINARIZED_COLUMN => &[CATEGORY_A],
            TARGET_COLUMN => &[true],
            COUNT_COLUMN => &[10]
        )
        .unwrap()
    }

    fn test_mostly_false_df() -> DataFrame {
        df!(
            TARGET_COLUMN => &[true, false, false],
        )
        .unwrap()
    }

    #[test]
    pub fn test_get_most_common_label_return_false() -> PolarsResult<()> {
        let df = test_mostly_false_df();
        let mode = get_most_common_label(&df.lazy())?;
        assert_eq!(mode, false);
        Ok(())
    }

    #[test]
    pub fn test_get_category_a_label_true() {
        let df = test_complete_df();
        let label = true;
        let category = CATEGORY_A;
        let output = extract_count(&df, label, category);
        assert_eq!(output, 10.0);
    }

    #[test]
    pub fn test_get_category_a_label_false() {
        let df = test_complete_df();
        let label = false;
        let category = CATEGORY_A;
        let output = extract_count(&df, label, category);
        assert_eq!(output, 30.0);
    }

    #[test]
    pub fn test_get_category_b_label_true() {
        let df = test_complete_df();
        let label = true;
        let category = CATEGORY_B;
        let output = extract_count(&df, label, category);
        assert_eq!(output, 20.0);
    }

    #[test]
    pub fn test_get_category_b_label_false() {
        let df = test_complete_df();
        let label = false;
        let category = CATEGORY_B;
        let output = extract_count(&df, label, category);
        assert_eq!(output, 40.0);
    }

    #[test]
    pub fn test_get_from_empty_dataframe_zero() {
        let df = test_empty_df();
        let label = false;
        let category = CATEGORY_B;
        let output = extract_count(&df, label, category);
        assert_eq!(output, 0.0);
    }

    #[test]
    pub fn test_get_from_empty_dataframe_non_zero() {
        let df = test_empty_df();
        let label = true;
        let category = CATEGORY_A;
        let output = extract_count(&df, label, category);
        assert_eq!(output, 10.0);
    }
}

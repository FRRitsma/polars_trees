use crate::constants::TARGET_COLUMN;
use polars::prelude::col;
use polars_core::datatypes::DataType;
use polars_lazy::frame::LazyFrame;

pub fn add_target_column(df: LazyFrame, target_column: &str) -> LazyFrame {
    df.with_column(
        col(target_column)
            .cast(DataType::Boolean)
            .alias(TARGET_COLUMN),
    )
}

// TODO: Add fail safe to ensure TARGET_COLUMN doesn't already exist in dataframe

#[cfg(test)]
mod tests {
    use crate::preprocessing::add_target_column;
    use crate::test_utils::get_raw_dataframe;

    #[test]
    fn test_add_target_column() {
        let df = get_raw_dataframe();
        let df = add_target_column(df, "Survived");
        let _ = df.collect().unwrap();
    }
}

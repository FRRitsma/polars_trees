use crate::preprocessing::add_target_column;
use polars::io::SerReader;
use polars::prelude::CsvReadOptions;
use polars_lazy::frame::{IntoLazy, LazyFrame};

// This allows the test module to access the functions in the outer scope
pub const FILE_PATH: &str = "Titanic-Dataset.csv";
pub const TITANIC_TARGET_COLUMN: &str = "Survived";

pub fn get_raw_test_dataframe() -> LazyFrame {
    CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(FILE_PATH.into()))
        .unwrap()
        .finish()
        .unwrap()
        .lazy()
}

pub fn get_preprocessed_test_dataframe() -> LazyFrame {
    add_target_column(get_raw_test_dataframe(), TITANIC_TARGET_COLUMN)
}

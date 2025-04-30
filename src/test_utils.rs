use crate::preprocessing::{add_target_column, pre_process_dataframe};
use crate::settings::Settings;
use polars::io::SerReader;
use polars::prelude::CsvReadOptions;
use polars_core::prelude::{DataFrame, DataType};
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
    let raw_lf = get_raw_test_dataframe();
    pre_process_dataframe(raw_lf, Settings::default(), TITANIC_TARGET_COLUMN)
}

pub fn assert_single_row_df_equal(
    df1: &DataFrame,
    df2: &DataFrame,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(df1.shape(), df2.shape(), "DataFrames have different shapes");

    for (col_name, dtype) in df1.schema().iter() {
        let series1 = df1.column(col_name)?;
        let series2 = df2.column(col_name)?;

        match dtype {
            DataType::Float64 => {
                let val1 = series1.f64()?.get(0).unwrap();
                let val2 = series2.f64()?.get(0).unwrap();

                assert!((val1 - val2).abs() < 0.00001f64);
            }
            DataType::String => {
                let val1 = series1.str()?.get(0).unwrap();
                let val2 = series2.str()?.get(0).unwrap();
                assert_eq!(val1, val2);
            }
            _ => {
                panic!("Unexpected DataType")
            }
        }
    }
    Ok(())
}

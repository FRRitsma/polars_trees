use polars::io::SerReader;
use polars::prelude::CsvReadOptions;
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use std::collections::HashSet;
use std::error::Error;

use polars::prelude::*;

mod categorical_columns;
mod constants;
mod display_tree;
mod empty_tree;
mod extract_values;
mod generic_functions;
mod numeric_type_columns;
mod preprocessing;
mod rework;
mod settings;
mod splitting_dataframes;
mod test_utils;
mod tree;


fn main() -> Result<(), Box<dyn Error>> {
    Ok(())
}

/*
Sort type governs how a column can be split left/right.
For ordinal, this is be greater-than/lesser-than a threshold.
For categorical, this equal/not-equal to a category
*/

use polars_core::datatypes::DataType;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum SortType {
    Ordinal,
    Categorical,
}

impl SortType {
    pub fn from_str(string: &str) -> Self {
        match string {
            "ordinal" => SortType::Ordinal,
            "categorical" => SortType::Categorical,
            _ => {
                panic!("Invalid choice for SortType")
            }
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SortType::Ordinal => "ordinal",
            SortType::Categorical => "categorical",
        }
    }
}

pub fn get_sort_type_for_dtype(dtype: &DataType) -> SortType {
    if matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    ) {
        return SortType::Ordinal;
    }
    if matches!(dtype, DataType::String) {
        return SortType::Categorical;
    }
    panic!("Unsupported DataType")
}

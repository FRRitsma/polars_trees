use crate::constants::TARGET_COLUMN;
use crate::settings::Settings;
use polars::prelude::{col, IdxSize, lit, PlSmallStr, when};
use polars_core::datatypes::DataType;
use polars_core::prelude::{NamedFrom, SortMultipleOptions};
use polars_core::series::Series;
use polars_lazy::frame::LazyFrame;
use std::error::Error;

pub fn add_target_column(lf: LazyFrame, target_column: &str) -> LazyFrame {
    lf.with_column(
        col(target_column)
            .cast(DataType::Boolean)
            .alias(TARGET_COLUMN),
    )
    .drop([col(target_column)])
}

pub fn get_string_type_columns(lf: LazyFrame) -> Vec<String> {
    let schema = lf.logical_plan.compute_schema().unwrap();
    let string_cols: Vec<_> = schema
        .iter()
        .filter_map(|(name, dtype)| {
            if let DataType::String = dtype {
                Some(name.to_string())
            } else {
                None
            }
        })
        .collect();
    string_cols
}

pub fn get_most_common_string_values_as_vector(
    lf: &LazyFrame,
    column_name: &str,
    settings: Settings,
) -> Result<Vec<String>, Box<dyn Error>> {
    let count_column = "count";
    let minimum_count = settings.get_min_leave_size() as i32;
    let top_n = settings.get_max_cardinality() as IdxSize;

    let top_df = lf
        .clone()
        .group_by([col(column_name)])
        .agg([col(column_name).count().alias(count_column)])
        .filter(col(count_column).gt(minimum_count))
        .sort(
            [count_column],
            SortMultipleOptions::default().with_order_descending(true),
        )
        .limit(top_n)
        .select([col(column_name)])
        .collect()?;

    let common_values: Vec<String> = top_df
        .column(column_name)?
        .str()?
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect();

    Ok(common_values)
}

pub const REDUNDANT_STRING_VALUE: &str = "FILLER_STRING";
pub fn prune_single_string_column(
    lf: LazyFrame,
    column_name: &str,
    keep_values: Vec<String>,
) -> Result<LazyFrame, Box<dyn Error>> {
    /*
    If a string isn't prevalent enough to be able to be a label rename it
    to simplify further processing
    */

    let pruned_lf: LazyFrame;

    if keep_values.len() == 0 {
        // If there are no values to keep, drop the column entirely:
        pruned_lf = lf.clone().drop([column_name]);
    } else {
        let is_in_expr = col(column_name).is_in(lit(Series::new(
            PlSmallStr::from("keep_values"),
            &keep_values,
        )));

        // Keep only common values, and rename redundant values:
        pruned_lf = lf.clone().with_column(
            when(is_in_expr)
                .then(col(column_name))
                .otherwise(lit(REDUNDANT_STRING_VALUE))
                .alias(column_name),
        );
    }

    Ok(pruned_lf)
}

pub fn pre_process_dataframe(lf: LazyFrame, settings: Settings, target_column: &str) -> LazyFrame {
    let lf = add_target_column(lf, target_column);
    let lf = prune_string_dataframe(lf, settings);
    lf
}

fn prune_string_dataframe(lf: LazyFrame, settings: Settings) -> LazyFrame {
    /*
    Keep in any column only the most prominent string values.
    If there are no prominent string values, remove the entire column.
    */
    let string_columns = get_string_type_columns(lf.clone());
    let mut pruned_df = lf.clone();
    for column in string_columns.iter() {
        let common_string_values =
            get_most_common_string_values_as_vector(&lf, column, settings).unwrap();
        pruned_df = prune_single_string_column(pruned_df, column, common_string_values).unwrap();
    }
    pruned_df
}

fn filter_uncommon_string_values(lf: LazyFrame, column_name: &str) -> LazyFrame {
    todo!()
}

#[cfg(test)]
mod tests {
    use crate::old_preprocessing::{
        add_target_column, get_most_common_string_values_as_vector, get_string_type_columns,
    };
    use crate::settings::Settings;
    use crate::test_utils::get_raw_test_dataframe;
    use polars::prelude::{col, lit, PlSmallStr, when};
    use polars_core::prelude::NamedFrom;
    use polars_core::series::Series;
    use crate::filler_strings::rename_filler_string_full_lazyframe;

    #[test]
    fn test_add_target_column() {
        let df = get_raw_test_dataframe();
        let df = add_target_column(df, "Survived");
        let _ = df.collect().unwrap();
    }

    #[test]
    fn test_get_string_type_columns() {
        let lf = get_raw_test_dataframe();
        assert_eq!(get_string_type_columns(lf).len(), 5);
    }

    #[test]
    fn test_most_common_values() {
        let lf = get_raw_test_dataframe();
        let string_columns = get_string_type_columns(lf.clone());
        let name_vector = get_most_common_string_values_as_vector(
            &lf,
            string_columns.iter().next().unwrap(),
            Settings::default(),
        )
        .unwrap();
        println!("{:?}", string_columns);
        assert_eq!(name_vector.len(), 0);
    }

    #[test]
    fn test_prune_name_column() {
        let lf = get_raw_test_dataframe();
        // let string_columns = get_string_type_columns(lf.clone());
        let keep_values = vec!["male"];

        let column_name = "Sex";
        let replacement = "EMPTY";

        let is_in_expr = col(column_name).is_in(lit(Series::new(
            PlSmallStr::from("keep_values"),
            &keep_values,
        )));

        let new_lf = lf.with_column(
            when(is_in_expr)
                .then(col(column_name))
                .otherwise(lit(replacement))
                .alias(column_name),
        );

        let demo_lf = new_lf.select([col(column_name)]).collect().unwrap();
        println!("{:?}", demo_lf);
    }

    #[test]
    fn test_prune_dataframe_rework() -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
            std::env::set_var("POLARS_FMT_MAX_ROWS", "100");
        }
        let lf = get_raw_test_dataframe();

        let minimum_sample_count = 10;
        let top_n_most_frequent = 5;
        let renamed_lf =
            rename_filler_string_full_lazyframe(lf, minimum_sample_count, top_n_most_frequent)?;
        println!("{:?}", renamed_lf.collect());

        Ok(())
    }
}

use crate::constants::TARGET_COLUMN;
use crate::gini_impurity::constants::{COUNT_LEFT_COL, COUNT_RIGHT_COL, SELECTION_COLUMN};
use crate::gini_impurity::gini_impurity::{
    add_totals_of_in_out_group, add_zero_count, compute_gini_per_feature, extract_best_feature,
    get_optimal_gini_impurity_for_column, normalize_gini_per_group, pre_process_for_gini,
};
use crate::gini_impurity::sort_type::SortType;
use crate::preprocessing::REDUNDANT_STRING_VALUE;
use crate::test_utils::assert_single_row_df_equal;
use polars::prelude::{col, lit};
use polars_core::datatypes::DataType;
use polars_core::df;
use polars_lazy::frame::LazyFrame;

pub fn group_by_for_gini_impurity_categorical(lf: &LazyFrame) -> LazyFrame {
    // Instead of grouping by feature column, should group by selection column.
    let mut grouped_lf = lf
        // Group in and out:
        .clone()
        .group_by([col("*")])
        .agg([col(TARGET_COLUMN)
            .count()
            .alias(COUNT_LEFT_COL)
            .cast(DataType::Float64)]);

    grouped_lf = add_zero_count(SELECTION_COLUMN, TARGET_COLUMN, COUNT_LEFT_COL, &grouped_lf);

    grouped_lf = grouped_lf
        .with_columns([col(COUNT_LEFT_COL)
            .sum()
            .over([col(TARGET_COLUMN)])
            .alias("total_per_target")])
        .with_columns([(col("total_per_target") - col(COUNT_LEFT_COL)).alias(COUNT_RIGHT_COL)])
        .filter(col(SELECTION_COLUMN).neq(lit(REDUNDANT_STRING_VALUE)))
        .drop(["total_per_target"])
        .sort([SELECTION_COLUMN, TARGET_COLUMN], Default::default());
    grouped_lf
}

pub fn get_optimal_gini_impurity_for_categorical_column(
    lf: &LazyFrame,
    feature_column: &str,
) -> LazyFrame {
    let lf = pre_process_for_gini(&lf, SortType::Categorical, feature_column);
    let mut grouped_lf = group_by_for_gini_impurity_categorical(&lf);
    grouped_lf = add_totals_of_in_out_group(&grouped_lf);
    let gini_lf = compute_gini_per_feature(&grouped_lf);
    let normalized_gini_lf = normalize_gini_per_group(&grouped_lf, &gini_lf);
    let final_lf = extract_best_feature(normalized_gini_lf);
    final_lf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::TARGET_COLUMN;
    use crate::gini_impurity::constants::{
        FEATURE_COLUMN_NAME, NORMALIZED_CHILD_GINI, SELECTION_COLUMN, SORT_TYPE_COL,
        TOTAL_LEFT_GROUP_COL, TOTAL_RIGHT_GROUP_COL,
    };
    use crate::test_utils::get_preprocessed_test_dataframe;
    #[test]
    fn debug() {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        }

        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let feature_column = "Embarked";
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);

        let lf = pre_process_for_gini(&lf, SortType::Categorical, feature_column);
        let mut grouped_lf = group_by_for_gini_impurity_categorical(&lf);
        grouped_lf = add_totals_of_in_out_group(&grouped_lf);
        let gini_lf = compute_gini_per_feature(&grouped_lf);
        let normalized_gini_lf = normalize_gini_per_group(&grouped_lf, &gini_lf);

        println!("{:?}", normalized_gini_lf.collect());
    }

    #[test]
    fn test_gini_for_categorical_column() -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        }

        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let feature_column = "Embarked";
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);

        // END OF PRE-PROCESSING, start of Gini computation:
        let final_lf =
            get_optimal_gini_impurity_for_column(&lf, feature_column, SortType::Categorical);
        let collected = final_lf.collect()?;

        let expected_df = df![
            FEATURE_COLUMN_NAME => &["Embarked"],
            SORT_TYPE_COL => &["categorical"],
            SELECTION_COLUMN => &["C"],
            NORMALIZED_CHILD_GINI => &[0.57038_f64],
            TOTAL_LEFT_GROUP_COL => &[168.0],
            TOTAL_RIGHT_GROUP_COL => &[723.0],
        ]?;

        assert_eq!(collected.schema(), expected_df.schema());
        assert_single_row_df_equal(&collected, &expected_df)?;

        Ok(())
    }
}
